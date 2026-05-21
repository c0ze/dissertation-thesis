"""Tests for the Qreg class -- construction, properties, and accessors.

These exercise every public method that ships in Phase 0+1. Gate
methods, measurement, QFT, Grover, and Shor land in later phases and
have their own test files.
"""

from __future__ import annotations

import pytest
import torch

from qubit import (
    AMP_TOL_C64,
    AMP_TOL_C128,
    PROB_TOL_C64,
    PROB_TOL_C128,
    Qreg,
    amp_tol_for,
    prob_tol_for,
)

# ---------------------------------------------------------------------------
# Construction: valid and invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_construct_valid_n(n: int) -> None:
    q = Qreg(n)
    assert q.n_qubits == n
    assert len(q._amp) == (1 << n)
    # Default initial state is the all-zeros tensor.
    assert q._amp.abs().sum().item() == pytest.approx(0.0)


def test_construct_rejects_n_zero() -> None:
    with pytest.raises(ValueError, match=r"^qubit: Qreg: n_qubits=0"):
        Qreg(0)


def test_construct_rejects_n_negative() -> None:
    with pytest.raises(ValueError, match=r"^qubit: Qreg: n_qubits=-3"):
        Qreg(-3)


def test_construct_rejects_non_int_n() -> None:
    with pytest.raises(TypeError, match=r"^qubit: Qreg: n_qubits must be int"):
        Qreg(3.5)  # type: ignore[arg-type]


def test_construct_rejects_bool_n() -> None:
    # `bool` is a subclass of `int` in Python (so mypy sees Qreg(True) as
    # type-compatible), but we reject it explicitly at runtime so
    # Qreg(True) doesn't silently make a 1-qubit register.
    with pytest.raises(TypeError, match=r"^qubit: Qreg: n_qubits must be int"):
        Qreg(True)


def test_construct_rejects_oversize_obvious() -> None:
    # Past the 1 TiB sanity ceiling -> MemoryError before allocation.
    with pytest.raises(MemoryError, match=r"^qubit: preflight: "):
        Qreg(50)


def test_construct_check_memory_false_skips_preflight() -> None:
    # The escape hatch should let through requests that the preflight
    # would reject. We still don't actually allocate (would OOM) -- just
    # confirm the MemoryError doesn't come from the preflight layer by
    # catching whatever runtime error PyTorch raises instead.
    with pytest.raises(Exception) as exc_info:
        Qreg(50, check_memory=False)
    # The error MUST NOT be the preflight's MemoryError with our prefix.
    msg = str(exc_info.value)
    assert not msg.startswith("qubit: preflight:"), (
        f"check_memory=False should skip preflight; got our message: {msg}"
    )


# ---------------------------------------------------------------------------
# Device / dtype resolution
# ---------------------------------------------------------------------------


def test_explicit_cpu_device() -> None:
    q = Qreg(3, device="cpu")
    assert q.device.type == "cpu"
    assert q.dtype == torch.complex128


def test_explicit_cpu_complex64() -> None:
    q = Qreg(3, device="cpu", dtype=torch.complex64)
    assert q.dtype == torch.complex64


def test_default_device_matches_helper() -> None:
    from qubit._device import default_device

    q = Qreg(3)
    assert q.device == default_device()


def test_mps_complex128_rejected() -> None:
    # The MPS-banned-dtype check runs regardless of whether MPS is
    # available -- validate_dtype_device is purely a policy function,
    # so this test is portable to CPU-only CI.
    with pytest.raises(ValueError, match=r"MPS backend does not support complex128"):
        Qreg(3, device="mps", dtype=torch.complex128)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_default_dtype_is_complex64() -> None:
    q = Qreg(3, device="mps")
    assert q.dtype == torch.complex64
    assert q.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_explicit_complex64_ok() -> None:
    q = Qreg(3, device="mps", dtype=torch.complex64)
    assert q.dtype == torch.complex64


# ---------------------------------------------------------------------------
# Seed and RNG
# ---------------------------------------------------------------------------


def test_construct_with_seed() -> None:
    q1 = Qreg(2, seed=42)
    q2 = Qreg(2, seed=42)
    # Both generators were seeded with the same int; their next draws
    # must match exactly. This validates that seed= goes through to the
    # RNG and that the RNG lives on CPU (so seeded reproducibility holds
    # regardless of where _amp ends up).
    draw1 = torch.rand((), generator=q1._gen).item()
    draw2 = torch.rand((), generator=q2._gen).item()
    assert draw1 == draw2


def test_construct_without_seed_diversifies() -> None:
    # Two unseeded Qregs constructed back-to-back should produce
    # independent RNG sequences. Not a strict guarantee (extremely
    # unlikely collision is possible) but the practical contract.
    q1 = Qreg(2)
    q2 = Qreg(2)
    draw1 = torch.rand((), generator=q1._gen).item()
    draw2 = torch.rand((), generator=q2._gen).item()
    assert draw1 != draw2


def test_rng_is_cpu_generator() -> None:
    # MPS-generator quirks motivate keeping the RNG on CPU even when
    # _amp lives on GPU. The generator's device should always be CPU.
    q = Qreg(2)
    assert q._gen.device.type == "cpu"


# ---------------------------------------------------------------------------
# init_basis
# ---------------------------------------------------------------------------


def test_init_basis_sets_target_amp_to_one() -> None:
    q = Qreg(3)
    q.init_basis(5)
    # |101> -> amp[5] = 1 + 0j, all else zero.
    for i in range(8):
        want = 1 + 0j if i == 5 else 0 + 0j
        assert q.amplitude(i) == want


def test_init_basis_zero_state() -> None:
    q = Qreg(3)
    q.init_basis(0)
    assert q.amplitude(0) == 1 + 0j
    for i in range(1, 8):
        assert q.amplitude(i) == 0 + 0j


def test_init_basis_zeros_previous_state() -> None:
    # init_basis must clear whatever was there before.
    q = Qreg(3)
    q.init_basis(7)
    q.init_basis(2)
    assert q.amplitude(2) == 1 + 0j
    assert q.amplitude(7) == 0 + 0j


def test_init_basis_rejects_out_of_range() -> None:
    q = Qreg(3)
    with pytest.raises(ValueError, match=r"^qubit: init_basis: basis=8"):
        q.init_basis(8)


def test_init_basis_rejects_negative() -> None:
    q = Qreg(3)
    with pytest.raises(ValueError, match=r"^qubit: init_basis: basis=-1"):
        q.init_basis(-1)


# ---------------------------------------------------------------------------
# amplitude
# ---------------------------------------------------------------------------


def test_amplitude_reads_back_one() -> None:
    q = Qreg(2)
    q.init_basis(2)
    assert q.amplitude(2) == 1 + 0j


def test_amplitude_reads_back_zero() -> None:
    q = Qreg(2)
    q.init_basis(2)
    assert q.amplitude(3) == 0 + 0j


def test_amplitude_returns_complex_type() -> None:
    q = Qreg(2)
    q.init_basis(0)
    assert isinstance(q.amplitude(0), complex)


def test_amplitude_rejects_out_of_range() -> None:
    q = Qreg(2)
    with pytest.raises(ValueError, match=r"^qubit: amplitude: i=4"):
        q.amplitude(4)


def test_amplitude_rejects_negative() -> None:
    q = Qreg(2)
    with pytest.raises(ValueError, match=r"^qubit: amplitude: i=-1"):
        q.amplitude(-1)


# ---------------------------------------------------------------------------
# amplitudes_copy
# ---------------------------------------------------------------------------


def test_amplitudes_copy_is_cpu_tensor() -> None:
    # Even with no GPU available, the contract is "always CPU"; verify
    # that explicitly so the same assertion holds when run on MPS.
    q = Qreg(3)
    copy = q.amplitudes_copy()
    assert copy.device.type == "cpu"


def test_amplitudes_copy_independent_of_qreg() -> None:
    q = Qreg(2)
    q.init_basis(0)
    copy = q.amplitudes_copy()
    # Mutate the copy; the live amp must be unaffected.
    copy[0] = 99 + 0j
    assert q.amplitude(0) == 1 + 0j


def test_amplitudes_copy_length_and_dtype() -> None:
    q = Qreg(3)
    copy = q.amplitudes_copy()
    assert copy.shape == (8,)
    assert copy.dtype == q.dtype


def test_amplitudes_copy_reflects_state_at_call_time() -> None:
    q = Qreg(2)
    q.init_basis(1)
    snap1 = q.amplitudes_copy()
    q.init_basis(2)
    snap2 = q.amplitudes_copy()
    # snap1 captured |01>; snap2 captured |10>. They must differ.
    assert not torch.equal(snap1, snap2)
    assert snap1[1].item() == 1 + 0j
    assert snap2[2].item() == 1 + 0j


# ---------------------------------------------------------------------------
# prob_of and norm
# ---------------------------------------------------------------------------


def test_prob_of_basis_state_one() -> None:
    q = Qreg(2)
    q.init_basis(3)
    assert q.prob_of(3) == pytest.approx(1.0, abs=PROB_TOL_C128)


def test_prob_of_basis_state_zero() -> None:
    q = Qreg(2)
    q.init_basis(3)
    assert q.prob_of(0) == pytest.approx(0.0, abs=PROB_TOL_C128)


def test_prob_of_rejects_out_of_range() -> None:
    q = Qreg(2)
    with pytest.raises(ValueError, match=r"^qubit: prob_of: basis=4"):
        q.prob_of(4)


def test_norm_basis_state_is_one() -> None:
    q = Qreg(4)
    q.init_basis(7)
    assert q.norm() == pytest.approx(1.0, abs=PROB_TOL_C128)


def test_norm_zero_state_is_zero() -> None:
    # The default-constructed Qreg (no init_basis call) is all zeros;
    # norm == 0. Pathological but well-defined.
    q = Qreg(3)
    assert q.norm() == pytest.approx(0.0, abs=PROB_TOL_C128)


# ---------------------------------------------------------------------------
# Tolerance helpers exported alongside Qreg
# ---------------------------------------------------------------------------


def test_amp_tol_complex128() -> None:
    assert amp_tol_for(torch.complex128) == AMP_TOL_C128


def test_amp_tol_complex64() -> None:
    assert amp_tol_for(torch.complex64) == AMP_TOL_C64


def test_amp_tol_unsupported_dtype_raises() -> None:
    with pytest.raises(ValueError, match=r"^qubit: amp_tol_for: "):
        amp_tol_for(torch.float64)


def test_prob_tol_complex128() -> None:
    assert prob_tol_for(torch.complex128) == PROB_TOL_C128


def test_prob_tol_complex64() -> None:
    assert prob_tol_for(torch.complex64) == PROB_TOL_C64


# ---------------------------------------------------------------------------
# Cross-device: construction on every available device
# ---------------------------------------------------------------------------


def test_construct_on_available_device(device: str) -> None:
    # The `device` fixture parametrises over whatever PyTorch can reach
    # on this host (always includes "cpu"; "mps" on Apple Silicon, etc.).
    # Construction must succeed on every available device with the
    # auto-selected dtype.
    q = Qreg(3, device=device)
    assert q.device.type == device
    # Default dtype must match policy.
    if device == "mps":
        assert q.dtype == torch.complex64
    else:
        assert q.dtype == torch.complex128


def test_init_basis_on_available_device(device: str) -> None:
    # The init_basis path must work end-to-end on every device, not just
    # CPU. This catches device-specific subtleties (e.g., the single-cell
    # assignment _amp[basis] = 1+0j has to be supported on the target).
    q = Qreg(3, device=device)
    q.init_basis(4)
    assert q.prob_of(4) == pytest.approx(1.0, abs=prob_tol_for(q.dtype))
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


def test_amplitudes_copy_always_cpu(device: str) -> None:
    # Regardless of the source device, amplitudes_copy must yield CPU.
    q = Qreg(3, device=device)
    q.init_basis(0)
    assert q.amplitudes_copy().device.type == "cpu"
