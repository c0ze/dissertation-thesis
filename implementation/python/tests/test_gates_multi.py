"""Tests for the multi-controlled gate primitives.

Covers :func:`apply_multi_controlled_z` (Grover diffusion's phase flip)
and :func:`apply_multi_controlled_x` (generalised Toffoli). Both target
the highest-likelihood failure mode (the MCX target-bit=0 mask) and
exhaustive validation.
"""

from __future__ import annotations

import math

import pytest

from qubit import (
    Qreg,
    amp_tol_for,
    apply_h,
    apply_multi_controlled_x,
    apply_multi_controlled_z,
    prob_tol_for,
)


def _approx(q: Qreg, expected: complex) -> object:
    return pytest.approx(expected, abs=amp_tol_for(q.dtype))


# ===========================================================================
# apply_multi_controlled_z
# ===========================================================================


def test_mcz_three_controls_flips_only_all_ones_in_superposition() -> None:
    # Prepare (|0> + |1>)^3 / sqrt(8) and apply MCZ over the three
    # qubits. Only the |111> amplitude should be negated; the other
    # seven stay at +1/sqrt(8).
    q = Qreg(3, device="cpu")
    q.init_basis(0)
    for i in range(3):
        apply_h(q, i)
    apply_multi_controlled_z(q, [0, 1, 2])
    inv8 = 1.0 / math.sqrt(8.0)
    for i in range(8):
        expected = -inv8 if i == 7 else inv8
        assert q.amplitude(i) == _approx(q, expected + 0j)


def test_mcz_two_controls_negates_only_q0_q1_both_one() -> None:
    # 3 qubits, MCZ on controls [0, 1]. Should negate amp[3] (|011>) and
    # amp[7] (|111>) -- both have q0=q1=1. Other six are untouched.
    q = Qreg(3, device="cpu")
    q.init_basis(0)
    for i in range(3):
        apply_h(q, i)
    apply_multi_controlled_z(q, [0, 1])
    inv8 = 1.0 / math.sqrt(8.0)
    expected_signs = [+1, +1, +1, -1, +1, +1, +1, -1]
    for i in range(8):
        assert q.amplitude(i) == _approx(q, expected_signs[i] * inv8 + 0j)


def test_mcz_single_control_is_z_on_that_qubit() -> None:
    # With one control [c], MCZ degenerates to "negate any basis state
    # with bit c set" -- exactly apply_z(q, c) up to global phase.
    q = Qreg(2, device="cpu")
    q.init_basis(3)  # |11>
    apply_multi_controlled_z(q, [0])
    # Negates because q0 is 1.
    assert q.amplitude(3) == _approx(q, -1 + 0j)


def test_mcz_non_contiguous_controls() -> None:
    # 4 qubits, controls=[0, 3] (skipping qubits 1 and 2). Only
    # amplitudes with both q0=1 AND q3=1 should be negated.
    q = Qreg(4, device="cpu")
    q.init_basis(0)
    for i in range(4):
        apply_h(q, i)
    apply_multi_controlled_z(q, [0, 3])
    inv16 = 1.0 / 4.0  # 1/sqrt(16)
    for i in range(16):
        q0 = i & 1
        q3 = (i >> 3) & 1
        sign = -1 if (q0 == 1 and q3 == 1) else +1
        assert q.amplitude(i) == _approx(q, sign * inv16 + 0j)


def test_mcz_leaves_norm_unchanged() -> None:
    q = Qreg(3, device="cpu")
    q.init_basis(0)
    for i in range(3):
        apply_h(q, i)
    apply_multi_controlled_z(q, [0, 1, 2])
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


def test_mcz_rejects_empty_controls() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(
        ValueError,
        match=r"^qubit: apply_multi_controlled_z: controls must be a non-empty",
    ):
        apply_multi_controlled_z(q, [])


def test_mcz_rejects_out_of_range_control() -> None:
    q = Qreg(3, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_multi_controlled_z: control=5"
    ):
        apply_multi_controlled_z(q, [0, 5])


def test_mcz_rejects_negative_control() -> None:
    q = Qreg(3, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_multi_controlled_z: control=-1"
    ):
        apply_multi_controlled_z(q, [0, -1])


def test_mcz_rejects_duplicate_controls() -> None:
    q = Qreg(3, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_multi_controlled_z: duplicate control=1"
    ):
        apply_multi_controlled_z(q, [1, 1])


def test_mcz_accepts_tuple_controls() -> None:
    # The signature is list[int] | tuple[int, ...]; both work.
    q = Qreg(2, device="cpu")
    q.init_basis(3)
    apply_multi_controlled_z(q, (0, 1))
    assert q.amplitude(3) == _approx(q, -1 + 0j)


# ===========================================================================
# apply_multi_controlled_x (generalised Toffoli)
# ===========================================================================


def test_mcx_toffoli_flips_target_when_both_controls_on() -> None:
    # |011> -> Toffoli with controls=[0,1], target=2 -> |111>.
    q = Qreg(3, device="cpu")
    q.init_basis(3)
    apply_multi_controlled_x(q, [0, 1], 2)
    assert q.amplitude(7) == _approx(q, 1 + 0j)
    assert q.amplitude(3) == _approx(q, 0 + 0j)


def test_mcx_no_flip_when_any_control_zero() -> None:
    # |001> has q0=1 but q1=0; Toffoli leaves it alone.
    q = Qreg(3, device="cpu")
    q.init_basis(1)
    apply_multi_controlled_x(q, [0, 1], 2)
    assert q.amplitude(1) == _approx(q, 1 + 0j)


def test_mcx_single_control_is_cnot() -> None:
    # Toffoli with one control degenerates to CNOT(control, target).
    # |001> with controls=[0], target=2 -> flip q2 -> |101>.
    q = Qreg(3, device="cpu")
    q.init_basis(1)
    apply_multi_controlled_x(q, [0], 2)
    assert q.amplitude(5) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_mcx_inverse_is_self() -> None:
    # X^2 = I and the controls are a permutation, so MCX^2 must be I.
    q = Qreg(3, device="cpu")
    q.init_basis(0)
    for i in range(3):
        apply_h(q, i)
    snapshot = q.amplitudes_copy()
    apply_multi_controlled_x(q, [0, 1], 2)
    apply_multi_controlled_x(q, [0, 1], 2)
    after = q.amplitudes_copy()
    for i in range(8):
        diff = abs(complex(snapshot[i].item()) - complex(after[i].item()))
        assert diff <= amp_tol_for(q.dtype), (
            f"MCX^2 != identity at amp[{i}]: diff={diff}"
        )


def test_mcx_three_controls_flips_only_when_all_on() -> None:
    # 4 qubits, controls=[0,1,2], target=3. |0111> = amp[7] has
    # q0=q1=q2=1 -> flip q3 -> |1111> = amp[15]. Other near-misses
    # (one control off) should be untouched.
    for basis, want in [
        (7, 15),    # all three controls on -> flip
        (3, 3),     # only q0,q1 on -> no flip
        (5, 5),     # only q0,q2 on -> no flip
        (6, 6),     # only q1,q2 on -> no flip
        (1, 1),     # only q0 on   -> no flip
    ]:
        q = Qreg(4, device="cpu")
        q.init_basis(basis)
        apply_multi_controlled_x(q, [0, 1, 2], 3)
        assert q.amplitude(want) == _approx(q, 1 + 0j), (
            f"MCX on basis={basis} -> want amp[{want}]=1, got {q.amplitude(want)}"
        )


def test_mcx_lsb_target_mapping() -> None:
    # Target the LSB: controls=[1, 2], target=0. |110> = amp[6];
    # q1=q2=1 -> flip q0 -> |111> = amp[7].
    q = Qreg(3, device="cpu")
    q.init_basis(6)
    apply_multi_controlled_x(q, [1, 2], 0)
    assert q.amplitude(7) == _approx(q, 1 + 0j)
    assert q.amplitude(6) == _approx(q, 0 + 0j)


def test_mcx_preserves_norm_under_superposition() -> None:
    q = Qreg(4, device="cpu")
    q.init_basis(0)
    for i in range(4):
        apply_h(q, i)
    apply_multi_controlled_x(q, [0, 1], 3)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


def test_mcx_rejects_empty_controls() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(
        ValueError,
        match=r"^qubit: apply_multi_controlled_x: controls must be a non-empty",
    ):
        apply_multi_controlled_x(q, [], 0)


def test_mcx_rejects_out_of_range_target() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_multi_controlled_x: target=5"
    ):
        apply_multi_controlled_x(q, [0], 5)


def test_mcx_rejects_out_of_range_control() -> None:
    q = Qreg(3, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_multi_controlled_x: control=7"
    ):
        apply_multi_controlled_x(q, [0, 7], 2)


def test_mcx_rejects_duplicate_controls() -> None:
    q = Qreg(3, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_multi_controlled_x: duplicate control=1"
    ):
        apply_multi_controlled_x(q, [1, 1], 2)


def test_mcx_rejects_target_in_controls() -> None:
    q = Qreg(3, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_multi_controlled_x: control 2 == target"
    ):
        apply_multi_controlled_x(q, [0, 2], 2)


def test_mcx_accepts_tuple_controls() -> None:
    q = Qreg(3, device="cpu")
    q.init_basis(3)
    apply_multi_controlled_x(q, (0, 1), 2)
    assert q.amplitude(7) == _approx(q, 1 + 0j)


# ===========================================================================
# Method-style API
# ===========================================================================


def test_method_apply_mcz_matches_function() -> None:
    qf = Qreg(3, device="cpu")
    qm = Qreg(3, device="cpu")
    qf.init_basis(0)
    qm.init_basis(0)
    for i in range(3):
        apply_h(qf, i)
        qm.apply_h(i)
    apply_multi_controlled_z(qf, [0, 1, 2])
    qm.apply_multi_controlled_z([0, 1, 2])
    for i in range(8):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_method_apply_mcx_matches_function() -> None:
    qf = Qreg(3, device="cpu")
    qm = Qreg(3, device="cpu")
    qf.init_basis(3)
    qm.init_basis(3)
    apply_multi_controlled_x(qf, [0, 1], 2)
    qm.apply_multi_controlled_x([0, 1], 2)
    for i in range(8):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


# ===========================================================================
# Device-parametrised sanity
# ===========================================================================


def test_mcz_all_controls_across_devices(device: str) -> None:
    # Same uniform-superposition MCZ test on every available device.
    q = Qreg(3, device=device)
    q.init_basis(0)
    for i in range(3):
        apply_h(q, i)
    apply_multi_controlled_z(q, [0, 1, 2])
    inv8 = 1.0 / math.sqrt(8.0)
    tol = amp_tol_for(q.dtype)
    assert q.amplitude(7) == pytest.approx(-inv8 + 0j, abs=tol)
    assert q.amplitude(0) == pytest.approx(inv8 + 0j, abs=tol)


def test_mcx_toffoli_across_devices(device: str) -> None:
    q = Qreg(3, device=device)
    q.init_basis(3)
    apply_multi_controlled_x(q, [0, 1], 2)
    assert q.prob_of(7) == pytest.approx(1.0, abs=prob_tol_for(q.dtype))
    assert q.prob_of(3) == pytest.approx(0.0, abs=prob_tol_for(q.dtype))
