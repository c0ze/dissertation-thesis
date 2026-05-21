"""Tests for the two-qubit controlled gates.

Covers ``apply_cu`` (the workhorse) plus the named controlled gates
(``apply_cnot`` / ``apply_cz`` / ``apply_controlled_phase``) and
``apply_swap``. The two highest-signal tests are:

* the Bell state ``H(0) -> CNOT(0, 1)`` producing
  ``(|00> + |11>) / sqrt(2)`` -- locks in the 4x4 layout convention and
  the permute / matmul / inverse-permute sequence at once;
* the non-adjacent ``CNOT(0, 2)`` on a 3-qubit register -- locks in the
  inverse-permutation accounting that adjacency would mask.
"""

from __future__ import annotations

import cmath
import math

import pytest
import torch

from qubit import (
    Qreg,
    amp_tol_for,
    apply_cnot,
    apply_controlled_phase,
    apply_cu,
    apply_cz,
    apply_h,
    apply_swap,
    prob_tol_for,
)

INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _approx(q: Qreg, expected: complex) -> object:
    return pytest.approx(expected, abs=amp_tol_for(q.dtype))


# ---------------------------------------------------------------------------
# apply_cu workhorse
# ---------------------------------------------------------------------------


def _x_matrix(q: Qreg) -> torch.Tensor:
    return torch.tensor(
        [[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]],
        dtype=q.dtype, device=q.device,
    )


def _identity_matrix(q: Qreg) -> torch.Tensor:
    return torch.eye(2, dtype=q.dtype, device=q.device)


def test_apply_cu_identity_is_noop() -> None:
    q = Qreg(3, device="cpu")
    q.init_basis(5)  # |101>
    apply_cu(q, 0, 1, _identity_matrix(q))
    assert q.amplitude(5) == _approx(q, 1 + 0j)
    for i in range(8):
        if i != 5:
            assert q.amplitude(i) == _approx(q, 0 + 0j)


def test_apply_cu_custom_x_matches_apply_cnot() -> None:
    qf = Qreg(3, device="cpu")
    qm = Qreg(3, device="cpu")
    qf.init_basis(1)
    qm.init_basis(1)
    apply_cu(qf, 0, 1, _x_matrix(qf))
    apply_cnot(qm, 0, 1)
    for i in range(8):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_apply_cu_rejects_out_of_range_control() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: apply_cu: control=5"):
        apply_cu(q, 5, 0, _identity_matrix(q))


def test_apply_cu_rejects_out_of_range_target() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: apply_cu: target=9"):
        apply_cu(q, 0, 9, _identity_matrix(q))


def test_apply_cu_rejects_equal_control_target() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: apply_cu: control == target == 1"):
        apply_cu(q, 1, 1, _identity_matrix(q))


def test_apply_cu_rejects_non_tensor() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(
        TypeError, match=r"^qubit: apply_cu: matrix must be a torch.Tensor"
    ):
        apply_cu(q, 0, 1, [[1, 0], [0, 1]])


def test_apply_cu_rejects_wrong_shape() -> None:
    q = Qreg(2, device="cpu")
    wrong = torch.eye(3, dtype=q.dtype, device=q.device)
    with pytest.raises(
        ValueError, match=r"^qubit: apply_cu: matrix shape must be \(2, 2\)"
    ):
        apply_cu(q, 0, 1, wrong)


def test_apply_cu_rejects_wrong_dtype() -> None:
    q = Qreg(2, device="cpu")  # complex128
    wrong = torch.eye(2, dtype=torch.complex64, device=q.device)
    with pytest.raises(
        ValueError, match=r"^qubit: apply_cu: matrix dtype must be"
    ):
        apply_cu(q, 0, 1, wrong)


# ---------------------------------------------------------------------------
# CNOT: bit flip, LSB / MSB axis, control=0 no-op, Bell state
# ---------------------------------------------------------------------------


def test_apply_cnot_control_zero_is_noop() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(2)  # |10> = q1=1, q0=0
    apply_cnot(q, 0, 1)  # control q0=0, target q1; no flip
    assert q.amplitude(2) == _approx(q, 1 + 0j)


def test_apply_cnot_lsb_mapping_flips_target() -> None:
    # |01> = amp[1] with control q0=1; flip target q1 -> |11> = amp[3].
    q = Qreg(2, device="cpu")
    q.init_basis(1)
    apply_cnot(q, 0, 1)
    assert q.amplitude(3) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_cnot_non_adjacent_control_target() -> None:
    # 3 qubits, control=0, target=2. |001> = amp[1]; control bit q0=1,
    # flip target q2 -> |101> = amp[5].
    q = Qreg(3, device="cpu")
    q.init_basis(1)
    apply_cnot(q, 0, 2)
    assert q.amplitude(5) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_cnot_msb_control_lsb_target() -> None:
    # 3 qubits, control=2 (MSB), target=0 (LSB). |100> = amp[4];
    # control q2=1, flip target q0 -> |101> = amp[5].
    q = Qreg(3, device="cpu")
    q.init_basis(4)
    apply_cnot(q, 2, 0)
    assert q.amplitude(5) == _approx(q, 1 + 0j)
    assert q.amplitude(4) == _approx(q, 0 + 0j)


def test_bell_state_from_h_then_cnot() -> None:
    # The canonical 4x4-layout-and-permutation acceptance test.
    q = Qreg(2, device="cpu")
    q.init_basis(0)
    apply_h(q, 0)
    apply_cnot(q, 0, 1)
    assert q.amplitude(0) == _approx(q, INV_SQRT2 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)
    assert q.amplitude(2) == _approx(q, 0 + 0j)
    assert q.amplitude(3) == _approx(q, INV_SQRT2 + 0j)


def test_cnot_preserves_norm() -> None:
    q = Qreg(3, device="cpu")
    q.init_basis(0)
    apply_h(q, 0)
    apply_h(q, 1)
    apply_cnot(q, 0, 2)
    apply_cnot(q, 1, 2)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


# ---------------------------------------------------------------------------
# Controlled-Z and controlled-Phase
# ---------------------------------------------------------------------------


def test_apply_cz_flips_only_eleven() -> None:
    # CZ adds a sign to |11> and leaves the other three computational
    # basis states alone. Build (|0> + |1>)^2 / 2 and check.
    q = Qreg(2, device="cpu")
    q.init_basis(0)
    apply_h(q, 0)
    apply_h(q, 1)
    apply_cz(q, 0, 1)
    half = 0.5
    assert q.amplitude(0) == _approx(q, half + 0j)
    assert q.amplitude(1) == _approx(q, half + 0j)
    assert q.amplitude(2) == _approx(q, half + 0j)
    assert q.amplitude(3) == _approx(q, -half + 0j)


def test_apply_cz_on_eleven_basis_state_negates() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(3)  # |11>
    apply_cz(q, 0, 1)
    assert q.amplitude(3) == _approx(q, -1 + 0j)


def test_apply_controlled_phase_pi_equals_cz() -> None:
    qf = Qreg(2, device="cpu")
    qm = Qreg(2, device="cpu")
    qf.init_basis(3)
    qm.init_basis(3)
    apply_controlled_phase(qf, 0, 1, math.pi)
    apply_cz(qm, 0, 1)
    for i in range(4):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_apply_controlled_phase_arbitrary_angle() -> None:
    # CPhase(theta)|11> = e^{i theta}|11>.
    q = Qreg(2, device="cpu")
    q.init_basis(3)
    apply_controlled_phase(q, 0, 1, math.pi / 5.0)
    expected = cmath.exp(complex(0, math.pi / 5.0))
    assert q.amplitude(3) == _approx(q, expected)


def test_apply_controlled_phase_no_effect_on_other_branches() -> None:
    # Only the |11> branch picks up the phase. Verify for the other three.
    for basis in (0, 1, 2):
        q = Qreg(2, device="cpu")
        q.init_basis(basis)
        apply_controlled_phase(q, 0, 1, math.pi / 3.0)
        assert q.amplitude(basis) == _approx(q, 1 + 0j)


# ---------------------------------------------------------------------------
# SWAP
# ---------------------------------------------------------------------------


def test_apply_swap_exchanges_01_and_10() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(1)  # |01>
    apply_swap(q, 0, 1)
    # |01> <-> |10>
    assert q.amplitude(2) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_swap_leaves_00_and_11_alone() -> None:
    for basis in (0, 3):
        q = Qreg(2, device="cpu")
        q.init_basis(basis)
        apply_swap(q, 0, 1)
        assert q.amplitude(basis) == _approx(q, 1 + 0j)


def test_apply_swap_non_adjacent() -> None:
    # 3 qubits, swap LSB and MSB. |100> (amp[4]) -> |001> (amp[1]).
    q = Qreg(3, device="cpu")
    q.init_basis(4)
    apply_swap(q, 0, 2)
    assert q.amplitude(1) == _approx(q, 1 + 0j)
    assert q.amplitude(4) == _approx(q, 0 + 0j)


def test_apply_swap_rejects_equal() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: apply_swap: a == b == 1"):
        apply_swap(q, 1, 1)


def test_apply_swap_rejects_out_of_range() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: apply_swap: a=5"):
        apply_swap(q, 5, 0)
    with pytest.raises(ValueError, match=r"^qubit: apply_swap: b=8"):
        apply_swap(q, 0, 8)


# ---------------------------------------------------------------------------
# Method-style API
# ---------------------------------------------------------------------------


def test_method_apply_cnot_matches_function() -> None:
    qf = Qreg(2, device="cpu")
    qm = Qreg(2, device="cpu")
    qf.init_basis(0)
    qm.init_basis(0)
    apply_h(qf, 0)
    qf_function_path = qf  # alias for clarity
    qm.apply_h(0)
    apply_cnot(qf_function_path, 0, 1)
    qm.apply_cnot(0, 1)
    for i in range(4):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_method_apply_cu_matches_function() -> None:
    qf = Qreg(2, device="cpu")
    qm = Qreg(2, device="cpu")
    qf.init_basis(1)
    qm.init_basis(1)
    # Build the X matrix separately for each path to ensure they don't
    # alias the same tensor; identical contents though.
    apply_cu(qf, 0, 1, _x_matrix(qf))
    qm.apply_cu(0, 1, _x_matrix(qm))
    for i in range(4):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_method_apply_swap_matches_function() -> None:
    qf = Qreg(3, device="cpu")
    qm = Qreg(3, device="cpu")
    qf.init_basis(1)
    qm.init_basis(1)
    apply_swap(qf, 0, 2)
    qm.apply_swap(0, 2)
    for i in range(8):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_method_apply_cz_matches_function() -> None:
    qf = Qreg(2, device="cpu")
    qm = Qreg(2, device="cpu")
    qf.init_basis(3)
    qm.init_basis(3)
    apply_cz(qf, 0, 1)
    qm.apply_cz(0, 1)
    for i in range(4):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_method_apply_controlled_phase_matches_function() -> None:
    qf = Qreg(2, device="cpu")
    qm = Qreg(2, device="cpu")
    qf.init_basis(3)
    qm.init_basis(3)
    apply_controlled_phase(qf, 0, 1, 0.7)
    qm.apply_controlled_phase(0, 1, 0.7)
    for i in range(4):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


# ---------------------------------------------------------------------------
# Device-parametrised sanity: Bell state and non-adjacent CNOT on every device
# ---------------------------------------------------------------------------


def test_bell_state_across_devices(device: str) -> None:
    q = Qreg(2, device=device)
    q.init_basis(0)
    apply_h(q, 0)
    apply_cnot(q, 0, 1)
    tol = amp_tol_for(q.dtype)
    assert q.amplitude(0) == pytest.approx(INV_SQRT2 + 0j, abs=tol)
    assert q.amplitude(3) == pytest.approx(INV_SQRT2 + 0j, abs=tol)
    assert q.amplitude(1) == pytest.approx(0 + 0j, abs=tol)
    assert q.amplitude(2) == pytest.approx(0 + 0j, abs=tol)


def test_non_adjacent_cnot_across_devices(device: str) -> None:
    # The non-adjacent case is the one most likely to expose a
    # permutation-accounting bug, so we run it on every device.
    q = Qreg(3, device=device)
    q.init_basis(1)
    apply_cnot(q, 0, 2)
    assert q.prob_of(5) == pytest.approx(1.0, abs=prob_tol_for(q.dtype))
    assert q.prob_of(1) == pytest.approx(0.0, abs=prob_tol_for(q.dtype))


def test_swap_across_devices(device: str) -> None:
    q = Qreg(3, device=device)
    q.init_basis(4)  # |100>
    apply_swap(q, 0, 2)  # exchange LSB and MSB
    assert q.prob_of(1) == pytest.approx(1.0, abs=prob_tol_for(q.dtype))
    assert q.prob_of(4) == pytest.approx(0.0, abs=prob_tol_for(q.dtype))
