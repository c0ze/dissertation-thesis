"""Qreg: state-vector quantum register backed by a 1-D PyTorch tensor.

Surface through Phase 3: construction + accessors (Phase 0+1) and the
single-qubit gate methods (Phase 3). Controlled gates, multi-controlled
gates, measurement (beyond ``prob_of`` / ``norm``), QFT, Grover, and
Shor land in later phases. See the design notes (rounds §1-§5 of the
brainstorm) for the overall architecture and the rationale behind each
choice.

The gate methods are thin wrappers around the function-style API in
:mod:`qubit.gates_single` (and equivalents for controlled / multi-
controlled / measurement / QFT / Grover / Shor in later phases). Both
styles are supported deliberately: ``q.apply_h(0)`` reads naturally for
method-chaining, while ``apply_h(q, 0)`` works for users who prefer the
functional style and matches the lower-level signature used internally.

Key invariants:

* ``_amp`` is a 1-D contiguous tensor of length ``2**_n``, dtype either
  ``complex64`` (MPS) or ``complex128`` (CPU / CUDA / ROCm).
* The amplitude tensor lives on ``_device`` and never migrates. Gates
  operate in-place via tensor views; only ``amplitudes_copy()``
  intentionally crosses to host.
* ``_gen`` is a CPU :class:`torch.Generator` regardless of where ``_amp``
  lives. Measurement crosses to CPU at the readout boundary anyway, so
  keeping the RNG on CPU dodges MPS-generator quirks (reports of
  ``Device type MPS is not supported`` and generator-device-mismatch
  errors in the PyTorch tracker) and gives seeded tests bit-identical
  reproducibility across all devices.
* A :class:`Qreg` is **not** safe for concurrent method calls from
  multiple Python threads on the same instance. Different ``Qreg``
  instances are independent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import torch

from . import _device, _memory
from ._assert import raise_type, raise_value

# Numerical tolerances by dtype. Tests pick the appropriate pair from
# the Qreg's dtype property; gate tests will use ``AMP_TOL_*`` to compare
# amplitudes, measurement tests will use ``PROB_TOL_*`` for probabilities.
AMP_TOL_C64: Final[float] = 1e-5
PROB_TOL_C64: Final[float] = 1e-4
AMP_TOL_C128: Final[float] = 1e-10
PROB_TOL_C128: Final[float] = 1e-9


def amp_tol_for(dtype: torch.dtype) -> float:
    """Amplitude-comparison tolerance for the given dtype."""
    if dtype == torch.complex128:
        return AMP_TOL_C128
    if dtype == torch.complex64:
        return AMP_TOL_C64
    raise ValueError(f"qubit: amp_tol_for: unsupported dtype {dtype}")


def prob_tol_for(dtype: torch.dtype) -> float:
    """Probability-comparison tolerance for the given dtype."""
    if dtype == torch.complex128:
        return PROB_TOL_C128
    if dtype == torch.complex64:
        return PROB_TOL_C64
    raise ValueError(f"qubit: prob_tol_for: unsupported dtype {dtype}")


class Qreg:
    """State-vector quantum register.

    Construct with :class:`Qreg(n_qubits) <Qreg>` for the auto-detected
    device and dtype. Keyword arguments override:

    * ``device`` -- one of ``"cpu"``, ``"cuda"``, ``"mps"``, a
      ``torch.device`` instance, or ``None`` (auto-detect).
    * ``seed`` -- integer for the measurement RNG; ``None`` uses fresh
      OS entropy.
    * ``dtype`` -- ``torch.complex64`` or ``torch.complex128``; ``None``
      picks the highest-precision dtype the device supports. ``mps``
      with ``complex128`` raises ``ValueError``.
    * ``check_memory`` -- ``True`` (default) runs a preflight that
      raises ``MemoryError`` for obviously-impossible requests.
      ``False`` skips, leaving raw allocation failures to PyTorch.
    """

    _amp: torch.Tensor       # shape (2**_n,), dtype complex64 or complex128
    _n: int                  # qubit count; >= 1
    _device: torch.device    # device where _amp lives
    _dtype: torch.dtype      # complex64 or complex128
    _gen: torch.Generator    # CPU RNG for measurement sampling

    def __init__(
        self,
        n_qubits: int,
        *,
        device: torch.device | str | None = None,
        seed: int | None = None,
        dtype: torch.dtype | None = None,
        check_memory: bool = True,
    ) -> None:
        # ---- validate caller-facing arguments ---------------------------
        # n_qubits must be a plain int (booleans are ints in Python; reject
        # them explicitly to avoid Qreg(True) silently making a 1-qubit reg).
        raise_type(
            isinstance(n_qubits, int) and not isinstance(n_qubits, bool),
            "Qreg: n_qubits must be int, got %s",
            type(n_qubits).__name__,
        )
        raise_value(
            n_qubits >= 1,
            "Qreg: n_qubits=%d must be >= 1",
            n_qubits,
        )

        # ---- resolve device and dtype -----------------------------------
        resolved_device = _device.coerce_device(device)
        if dtype is None:
            resolved_dtype = _device.default_dtype(resolved_device)
        else:
            # Explicit caller dtype: validate against the device before
            # we commit to allocating.
            _device.validate_dtype_device(resolved_device, dtype)
            resolved_dtype = dtype

        # ---- preflight memory ------------------------------------------
        if check_memory:
            _memory.preflight(
                n_qubits, resolved_dtype, resolved_device, op="state"
            )

        # ---- allocate state vector -------------------------------------
        # zeros() returns a contiguous tensor; later gate code relies on
        # contiguity for the (2,)*n reshape view.
        amp = torch.zeros(
            1 << n_qubits, dtype=resolved_dtype, device=resolved_device
        )

        # ---- measurement RNG -------------------------------------------
        # CPU generator regardless of where _amp lives. MPS measurement
        # crosses to CPU at the sample step anyway; CUDA measurement
        # likewise pulls the sampled scalar to host before collapsing.
        gen = torch.Generator()
        if seed is None:
            # Default: fresh OS-entropy seed. Different Qregs without a
            # supplied seed produce independent measurement sequences.
            gen.seed()
        else:
            gen.manual_seed(int(seed))

        # ---- commit -----------------------------------------------------
        self._n = n_qubits
        self._device = resolved_device
        self._dtype = resolved_dtype
        self._amp = amp
        self._gen = gen

    # ---- read-only properties --------------------------------------------

    @property
    def n_qubits(self) -> int:
        """Number of qubits this register represents."""
        return self._n

    @property
    def device(self) -> torch.device:
        """Device where the amplitude tensor lives."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Complex dtype of the amplitude tensor."""
        return self._dtype

    # ---- state-mutating accessor ----------------------------------------

    def init_basis(self, basis: int) -> None:
        """Collapse the register to the computational basis state ``|basis>``.

        Zeros every amplitude, then sets ``amp[basis] = 1 + 0j``. Used to
        prepare a known starting state before applying a gate sequence.
        """
        raise_value(
            0 <= basis < (1 << self._n),
            "init_basis: basis=%d out of [0, %d)",
            basis,
            1 << self._n,
        )
        self._amp.zero_()
        # Tensor assignment of a Python complex to a complex-dtype cell
        # works across CPU / CUDA / MPS. No host->device transfer of a
        # full tensor: this is a single-element write.
        self._amp[basis] = 1 + 0j

    # ---- read-only accessors --------------------------------------------

    def amplitude(self, i: int) -> complex:
        """Return ``amp[i]`` as a Python ``complex``.

        Bounds-checked; raises ``ValueError`` for ``i`` outside
        ``[0, 2**n_qubits)``. The single-element lookup synchronises with
        the device (it has to, to materialise a Python scalar); use this
        for spot checks, not for hot loops. For full-vector inspection
        use :meth:`amplitudes_copy`.
        """
        raise_value(
            0 <= i < (1 << self._n),
            "amplitude: i=%d out of [0, %d)",
            i,
            1 << self._n,
        )
        return complex(self._amp[i].item())

    def amplitudes_copy(self) -> torch.Tensor:
        """Return a detached CPU clone of the full amplitude vector.

        Always returns a CPU tensor regardless of where ``_amp`` lives,
        so callers can inspect, pickle, compare amplitude-by-amplitude
        against ``/c`` and ``/go`` outputs, or feed the result to NumPy
        without further conversion. This intentionally synchronises with
        the device (``.cpu()`` is itself a sync point), so it's a
        diagnostic API, not a hot-path one. Mutations to the returned
        tensor do not affect ``_amp``.
        """
        return self._amp.detach().cpu().clone()

    def prob_of(self, basis: int) -> float:
        """Return ``|amp[basis]|^2`` as a Python ``float``.

        Bounds-checked. Computes ``real**2 + imag**2`` on device (one
        multiply-and-add), then materialises the scalar.
        """
        raise_value(
            0 <= basis < (1 << self._n),
            "prob_of: basis=%d out of [0, %d)",
            basis,
            1 << self._n,
        )
        a = self._amp[basis]
        # Use real()/imag() views and square on device; .item() forces
        # the single-scalar materialisation at the end.
        return float((a.real * a.real + a.imag * a.imag).item())

    def norm(self) -> float:
        """Sum of ``|amp[i]|^2`` over the entire state vector.

        For a normalised state this is 1.0 within floating-point
        precision; gate tests assert ``abs(norm() - 1) <= PROB_TOL``.
        Computed in one ``torch.sum`` call on device.
        """
        amp = self._amp
        return float(
            (amp.real * amp.real + amp.imag * amp.imag).sum().item()
        )

    # ---- single-qubit gate methods (Phase 3) ----------------------------
    #
    # These are thin wrappers around the function-style API in
    # :mod:`qubit.gates_single`. Both call shapes are public:
    # ``q.apply_h(0)`` is method-style; ``apply_h(q, 0)`` (imported from
    # ``qubit``) is the function-style equivalent. They share the same
    # underlying implementation -- the method just forwards self.
    #
    # The :mod:`qubit.gates_single` import is deferred inside each method
    # so importing :mod:`qubit.qreg` from the gates module (for static
    # type hints via ``TYPE_CHECKING``) does not create a runtime cycle.

    def apply_u(self, target: int, u: torch.Tensor) -> None:
        """Apply a 2x2 single-qubit unitary to the target qubit.

        ``u`` must be a 2x2 :class:`torch.Tensor` on the same device and
        with the same complex dtype as this register. See
        :func:`qubit.gates_single.apply_u` for the full contract.
        """
        from . import gates_single

        gates_single.apply_u(self, target, u)

    def apply_h(self, target: int) -> None:
        """Apply the Hadamard gate to ``target``."""
        from . import gates_single

        gates_single.apply_h(self, target)

    def apply_x(self, target: int) -> None:
        """Apply the Pauli-X (bit-flip) gate to ``target``."""
        from . import gates_single

        gates_single.apply_x(self, target)

    def apply_y(self, target: int) -> None:
        """Apply the Pauli-Y gate to ``target``."""
        from . import gates_single

        gates_single.apply_y(self, target)

    def apply_z(self, target: int) -> None:
        """Apply the Pauli-Z (phase-flip) gate to ``target``."""
        from . import gates_single

        gates_single.apply_z(self, target)

    def apply_s(self, target: int) -> None:
        """Apply the S (phase pi/2) gate to ``target``."""
        from . import gates_single

        gates_single.apply_s(self, target)

    def apply_t(self, target: int) -> None:
        """Apply the T (phase pi/4) gate to ``target``."""
        from . import gates_single

        gates_single.apply_t(self, target)

    def apply_phase(self, target: int, theta: float) -> None:
        """Apply the general phase gate ``diag(1, e^{i theta})`` to ``target``."""
        from . import gates_single

        gates_single.apply_phase(self, target, theta)

    def apply_rx(self, target: int, theta: float) -> None:
        """Apply the rotation ``RX(theta)`` to ``target``."""
        from . import gates_single

        gates_single.apply_rx(self, target, theta)

    def apply_ry(self, target: int, theta: float) -> None:
        """Apply the rotation ``RY(theta)`` to ``target``."""
        from . import gates_single

        gates_single.apply_ry(self, target, theta)

    def apply_rz(self, target: int, theta: float) -> None:
        """Apply the rotation ``RZ(theta)`` to ``target``."""
        from . import gates_single

        gates_single.apply_rz(self, target, theta)

    # ---- two-qubit controlled gate methods (Phase 4) --------------------
    #
    # Same lazy-import discipline as the single-qubit methods. The
    # function-style equivalents live in :mod:`qubit.gates_controlled`.

    def apply_cu(
        self, control: int, target: int, u: torch.Tensor
    ) -> None:
        """Apply a 2x2 controlled-unitary with ``u`` on the target qubit.

        See :func:`qubit.gates_controlled.apply_cu` for the contract.
        """
        from . import gates_controlled

        gates_controlled.apply_cu(self, control, target, u)

    def apply_cnot(self, control: int, target: int) -> None:
        """Apply controlled-NOT: flip ``target`` when ``control`` is 1."""
        from . import gates_controlled

        gates_controlled.apply_cnot(self, control, target)

    def apply_cz(self, control: int, target: int) -> None:
        """Apply controlled-Z: phase-flip when both qubits are 1."""
        from . import gates_controlled

        gates_controlled.apply_cz(self, control, target)

    def apply_controlled_phase(
        self, control: int, target: int, theta: float
    ) -> None:
        """Apply controlled-Phase: multiply ``|11>`` by ``e^{i theta}``."""
        from . import gates_controlled

        gates_controlled.apply_controlled_phase(
            self, control, target, theta
        )

    def apply_swap(self, a: int, b: int) -> None:
        """Exchange qubits ``a`` and ``b`` via the 3-CNOT decomposition."""
        from . import gates_controlled

        gates_controlled.apply_swap(self, a, b)

    # ---- multi-controlled gate methods (Phase 4) ------------------------

    def apply_multi_controlled_z(
        self, controls: list[int] | tuple[int, ...]
    ) -> None:
        """Phase-flip when every listed control bit is 1."""
        from . import gates_multi

        gates_multi.apply_multi_controlled_z(self, controls)

    def apply_multi_controlled_x(
        self,
        controls: list[int] | tuple[int, ...],
        target: int,
    ) -> None:
        """Flip ``target`` when every listed control is 1 (generalised Toffoli)."""
        from . import gates_multi

        gates_multi.apply_multi_controlled_x(self, controls, target)

    # ---- measurement, sampling, clone, dump (Phase 5) -------------------

    def measure_qubit(self, target: int) -> int:
        """Projective measurement on ``target``; collapse + renormalise.

        Returns ``0`` or ``1``. See :func:`qubit.measure.measure_qubit`.
        """
        from . import measure

        return measure.measure_qubit(self, target)

    def measure_all(self) -> int:
        """Sample a full basis index from ``|amp|^2`` and collapse.

        See :func:`qubit.measure.measure_all`.
        """
        from . import measure

        return measure.measure_all(self)

    def sample_distribution(self, shots: int) -> list[int]:
        """Run ``shots`` independent measurements; original is unchanged.

        See :func:`qubit.measure.sample_distribution`.
        """
        from . import measure

        return measure.sample_distribution(self, shots)

    def clone(self) -> Qreg:
        """Return an independent copy of this register (amp + RNG state).

        See :func:`qubit.measure.clone`.
        """
        from . import measure

        return measure.clone(self)

    def dump(
        self, *, threshold: float = 0.0
    ) -> list[tuple[int, complex]]:
        """Structured list of ``(basis, amplitude)`` with ``|amp| > threshold``.

        See :func:`qubit.measure.dump`.
        """
        from . import measure

        return measure.dump(self, threshold=threshold)

    # ---- QFT (Phase 6) --------------------------------------------------

    def apply_qft(self, start: int = 0, n: int | None = None) -> None:
        """Apply the forward QFT to qubits ``[start, start + n)``.

        See :func:`qubit.qft.apply_qft`.
        """
        from . import qft

        qft.apply_qft(self, start, n)

    def apply_qft_inverse(
        self, start: int = 0, n: int | None = None
    ) -> None:
        """Apply the inverse QFT to qubits ``[start, start + n)``.

        See :func:`qubit.qft.apply_qft_inverse`.
        """
        from . import qft

        qft.apply_qft_inverse(self, start, n)

    # ---- Grover (Phase 7) -----------------------------------------------

    def apply_grover(
        self,
        n_qubits: int,
        oracle: Callable[[Qreg, Any], None],
        user: Any = None,
        iterations: int | None = None,
    ) -> None:
        """Run Grover on qubits ``[0, n_qubits)`` of this register.

        See :func:`qubit.grover.apply_grover`.
        """
        from . import grover

        grover.apply_grover(self, n_qubits, oracle, user, iterations)
