"""Measurement, sampling, clone, and diagnostic dump.

Five public functions, all also exposed as :class:`Qreg` methods:

* :func:`measure_qubit` -- single-qubit projective measurement; the
  register collapses onto the measured branch and renormalises.
* :func:`measure_all` -- sample a full basis index from the
  ``|amp|^2`` distribution and collapse onto it.
* :func:`sample_distribution` -- run ``shots`` independent measurements
  *without* mutating the original register (snapshot + restore).
* :func:`clone` -- independent copy with the same amplitudes and RNG
  state. Mutations on the clone don't reach the original; both
  generators produce the same future sequence until one advances.
* :func:`dump` -- structured diagnostic: list of
  ``(basis_index, amplitude)`` pairs with ``|amp| > threshold``.

Measurement is the documented "device -> host" boundary: we deliberately
``.cpu()`` the probability vector once per call to feed the CPU
:class:`torch.Generator` (the cross-device-portable RNG choice from
Phase 0+1). Gate ops stay on device; only the measurement readout
crosses to host.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ._assert import raise_value
from ._axis import qubit_axis

if TYPE_CHECKING:
    from .qreg import Qreg


# ---------------------------------------------------------------------------
# measure_qubit: single-qubit projective measurement with collapse.
# ---------------------------------------------------------------------------


def measure_qubit(q: Qreg, target: int) -> int:
    """Perform a projective measurement on the ``target`` qubit.

    Returns ``0`` or ``1``. The register is collapsed onto the chosen
    branch and renormalised so :meth:`Qreg.norm` remains 1.

    Uses the per-Qreg CPU RNG ``q._gen`` for the sample. Two scalar
    syncs to host: the two slice-probability sums (so we can sample on
    CPU). Raises :class:`RuntimeError` with the ``qubit:`` prefix if
    the chosen branch has probability ``<= 0``, which only happens if
    the state has drifted from normalisation.
    """
    raise_value(
        0 <= target < q._n,
        "measure_qubit: target=%d out of [0, %d)", target, q._n,
    )

    n = q._n
    axis = qubit_axis(target, n)

    # n-D view; select gives non-contiguous views of the two halves
    # along the target axis. In-place ops on these slices modify
    # q._amp's underlying storage.
    view = q._amp.view((2,) * n)
    slice_0 = view.select(axis, 0)
    slice_1 = view.select(axis, 1)

    # Compute both probabilities from the actual slice contents (rather
    # than p1 = 1 - p0) so the renormalisation is exact relative to the
    # chosen branch even if the state has drifted slightly.
    p0 = float(
        (slice_0.real * slice_0.real + slice_0.imag * slice_0.imag).sum().item()
    )
    p1 = float(
        (slice_1.real * slice_1.real + slice_1.imag * slice_1.imag).sum().item()
    )

    # CPU random draw; q._gen is the per-Qreg CPU generator from Phase 0+1.
    u = float(torch.rand((), generator=q._gen).item())
    outcome = 0 if u < p0 else 1
    p_outcome = p0 if outcome == 0 else p1

    if p_outcome <= 0:
        raise RuntimeError(
            f"qubit: measure_qubit: outcome={outcome} sampled with "
            f"p_outcome={p_outcome:.6g} <= 0; the state has drifted "
            f"from normalisation (call q.norm() to confirm)"
        )

    inv_norm = 1.0 / math.sqrt(p_outcome)

    # Collapse: zero the other half, rescale the chosen half. Both
    # operations are in-place on slices that view q._amp's storage,
    # so q._amp is updated without any reshape/flatten work.
    if outcome == 0:
        slice_1.zero_()
        slice_0.mul_(inv_norm)
    else:
        slice_0.zero_()
        slice_1.mul_(inv_norm)

    return outcome


# ---------------------------------------------------------------------------
# measure_all: sample a full basis index, collapse to it.
# ---------------------------------------------------------------------------


def measure_all(q: Qreg) -> int:
    """Sample a full basis index from the ``|amp|^2`` distribution.

    The register collapses to a pure basis state at the chosen index;
    after this call ``q.norm()`` is 1.0 and exactly one amplitude is
    ``1 + 0j``.

    The probability vector is moved to CPU once so :func:`torch.multinomial`
    can use the per-Qreg CPU generator.
    """
    amp = q._amp
    # Real-valued probability vector on device, then one .cpu() sync
    # so multinomial can pull a sample using q._gen (CPU generator).
    # multinomial accepts non-normalised weights -- it renormalises
    # internally -- so we don't need to assert sum == 1 here.
    probs = (amp.real * amp.real + amp.imag * amp.imag).detach().cpu()
    chosen = int(torch.multinomial(probs, 1, generator=q._gen).item())

    # Collapse on device: zero everything, set the chosen index to 1+0j.
    q._amp.zero_()
    q._amp[chosen] = 1 + 0j
    return chosen


# ---------------------------------------------------------------------------
# sample_distribution: shots independent measurements, original untouched.
# ---------------------------------------------------------------------------


def sample_distribution(q: Qreg, shots: int) -> list[int]:
    """Run ``shots`` independent :func:`measure_all` shots and return the outcomes.

    Does **not** mutate ``q``: the amplitude tensor is restored to its
    pre-call state before this function returns (including on
    exceptions, via ``try / finally``). ``q._gen`` is advanced by
    ``shots`` measurements, which matches calling ``measure_all`` in a
    loop -- the RNG is the one piece of caller-visible state that the
    sampling consumes.

    ``shots=0`` is valid and returns the empty list. ``shots < 0`` raises.
    Returns a plain ``list[int]`` for ergonomic use with
    :class:`collections.Counter`.
    """
    raise_value(
        shots >= 0,
        "sample_distribution: shots=%d must be >= 0", shots,
    )
    if shots == 0:
        return []

    # Snapshot the initial amplitudes; we restore before each shot so
    # each measurement samples from the original distribution.
    snapshot = q._amp.detach().clone()
    out: list[int] = []

    try:
        for _ in range(shots):
            # Fresh copy of the original state for this shot's collapse.
            q._amp = snapshot.detach().clone()
            out.append(measure_all(q))
    finally:
        # Restore q to the pre-call state regardless of whether
        # measure_all succeeded on every shot.
        q._amp = snapshot

    return out


# ---------------------------------------------------------------------------
# clone: independent Qreg with same amplitudes and RNG state.
# ---------------------------------------------------------------------------


def clone(q: Qreg) -> Qreg:
    """Return an independent copy of ``q``.

    The clone has its own ``_amp`` tensor (a detached copy of ``q._amp``)
    and its own ``_gen`` generator initialised with the same state as
    ``q._gen``. From the moment of cloning, the two registers are
    independent: mutating one's amplitudes doesn't affect the other's,
    and advancing one's RNG (by measurement) doesn't advance the
    other's. They produce the **same** future random sequence until
    one of them draws.

    Bypasses :class:`Qreg`'s ``__init__`` (via ``__new__``) to avoid
    the otherwise-wasted ``torch.zeros(2**n)`` allocation that would
    immediately get overwritten by the amplitude copy. Same trick is
    used by :func:`sample_distribution` indirectly via its snapshot
    pattern.
    """
    from .qreg import Qreg as _Qreg

    new_q = _Qreg.__new__(_Qreg)
    new_q._n = q._n
    new_q._device = q._device
    new_q._dtype = q._dtype
    new_q._amp = q._amp.detach().clone()

    # Match RNG state. Both generators are independent objects from
    # here on; only their initial state is shared.
    new_q._gen = torch.Generator()
    new_q._gen.set_state(q._gen.get_state())
    return new_q


# ---------------------------------------------------------------------------
# dump: structured diagnostic list.
# ---------------------------------------------------------------------------


def dump(q: Qreg, *, threshold: float = 0.0) -> list[tuple[int, complex]]:
    """Return a list of ``(basis_index, amplitude)`` pairs with ``|amp| > threshold``.

    Diagnostic API; does no printing (callers can pretty-print the list
    however they like). Threshold defaults to 0, which returns every
    non-zero amplitude. Pass a positive threshold to filter out small
    amplitudes that you don't care about.

    Pulls the amplitude tensor to CPU once via ``amplitudes_copy``-style
    detach/copy. For sparse states the iteration is bounded by the
    non-zero count, not by ``2**n``.
    """
    raise_value(
        threshold >= 0,
        "dump: threshold=%g must be non-negative", threshold,
    )

    cpu_amp = q._amp.detach().cpu()
    # Find indices whose |amp| exceeds the threshold. .abs() on a
    # complex tensor returns magnitudes; the boolean mask is on the
    # real magnitude space.
    mask = cpu_amp.abs() > threshold
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    return [(int(i), complex(cpu_amp[i].item())) for i in indices]
