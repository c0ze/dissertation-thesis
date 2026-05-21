"""Tests for the single-qubit gate primitives.

Covers ``apply_u`` (the workhorse) plus every named gate
(``apply_h`` / ``apply_x`` / ``apply_y`` / ``apply_z`` / ``apply_s`` /
``apply_t`` / ``apply_phase`` / ``apply_rx`` / ``apply_ry`` /
``apply_rz``), in both function-style ``apply_h(q, 0)`` and method-style
``q.apply_h(0)`` invocations.

The LSB-first qubit-to-axis mapping is the single most likely place to
introduce a silent endian flip; ``test_apply_x_lsb_axis_mapping`` and
``test_apply_x_msb_axis_mapping`` lock the convention in alongside
``tests/test_axis.py``.
"""

from __future__ import annotations

import cmath
import math

import pytest
import torch

from qubit import (
    Qreg,
    amp_tol_for,
    apply_h,
    apply_phase,
    apply_rx,
    apply_ry,
    apply_rz,
    apply_s,
    apply_t,
    apply_u,
    apply_x,
    apply_y,
    apply_z,
    prob_tol_for,
)

# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------


def _approx(q: Qreg, expected: complex, **_: object) -> object:
    """Return ``pytest.approx(expected)`` with the right dtype tolerance."""
    return pytest.approx(expected, abs=amp_tol_for(q.dtype))


# Convenient complex literal
INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# apply_u workhorse: shape, dtype, device, and target validation
# ---------------------------------------------------------------------------


def _eye2(q: Qreg) -> torch.Tensor:
    """Build a 2x2 identity tensor on q's device/dtype."""
    return torch.eye(2, dtype=q.dtype, device=q.device)


def test_apply_u_identity_is_noop() -> None:
    q = Qreg(3, device="cpu")
    q.init_basis(5)  # |101>
    apply_u(q, 1, _eye2(q))
    assert q.amplitude(5) == _approx(q, 1 + 0j)
    for i in range(8):
        if i != 5:
            assert q.amplitude(i) == _approx(q, 0 + 0j)


def test_apply_u_with_explicit_x_matches_apply_x() -> None:
    # Custom u = X; result must match the named apply_x exactly.
    q1 = Qreg(2, device="cpu")
    q2 = Qreg(2, device="cpu")
    q1.init_basis(0)
    q2.init_basis(0)
    x = torch.tensor(
        [[0 + 0j, 1 + 0j],
         [1 + 0j, 0 + 0j]],
        dtype=q1.dtype, device=q1.device,
    )
    apply_u(q1, 0, x)
    apply_x(q2, 0)
    for i in range(4):
        assert q1.amplitude(i) == _approx(q1, q2.amplitude(i))


def test_apply_u_rejects_out_of_range_target() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: qubit_axis: target=5"):
        apply_u(q, 5, _eye2(q))


def test_apply_u_rejects_negative_target() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: qubit_axis: target=-1"):
        apply_u(q, -1, _eye2(q))


def test_apply_u_rejects_non_tensor() -> None:
    q = Qreg(2, device="cpu")
    bad = [[1.0, 0.0], [0.0, 1.0]]  # a plain Python list
    # No `# type: ignore` here: our mypy config follow-skips torch stubs,
    # so torch.Tensor resolves as Any and the type checker can't catch
    # this bad call statically. The runtime guard is the only defence.
    with pytest.raises(
        TypeError, match=r"^qubit: apply_u: matrix must be a torch.Tensor"
    ):
        apply_u(q, 0, bad)


def test_apply_u_rejects_wrong_shape() -> None:
    q = Qreg(2, device="cpu")
    wrong_shape = torch.eye(3, dtype=q.dtype, device=q.device)
    with pytest.raises(
        ValueError, match=r"^qubit: apply_u: matrix shape must be \(2, 2\)"
    ):
        apply_u(q, 0, wrong_shape)


def test_apply_u_rejects_wrong_dtype() -> None:
    q = Qreg(2, device="cpu")  # complex128
    wrong_dtype = torch.eye(2, dtype=torch.complex64, device=q.device)
    with pytest.raises(
        ValueError, match=r"^qubit: apply_u: matrix dtype must be"
    ):
        apply_u(q, 0, wrong_dtype)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="device-mismatch test needs a second device; only MPS is locally available",
)
def test_apply_u_rejects_wrong_device() -> None:
    # Qreg on MPS, matrix on CPU -- caller bug; must raise.
    q = Qreg(2, device="mps")  # complex64
    cpu_matrix = torch.eye(2, dtype=q.dtype, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_u: matrix device must be"
    ):
        apply_u(q, 0, cpu_matrix)


# ---------------------------------------------------------------------------
# Hadamard
# ---------------------------------------------------------------------------


def test_apply_h_on_zero_produces_plus_state() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_h(q, 0)
    assert q.amplitude(0) == _approx(q, INV_SQRT2 + 0j)
    assert q.amplitude(1) == _approx(q, INV_SQRT2 + 0j)


def test_apply_h_on_one_produces_minus_state() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_h(q, 0)
    # H|1> = (|0> - |1>) / sqrt(2)
    assert q.amplitude(0) == _approx(q, INV_SQRT2 + 0j)
    assert q.amplitude(1) == _approx(q, -INV_SQRT2 + 0j)


def test_apply_h_twice_is_identity() -> None:
    q = Qreg(3, device="cpu")
    q.init_basis(5)  # |101>
    apply_h(q, 1)
    apply_h(q, 1)
    for i in range(8):
        expected = 1 + 0j if i == 5 else 0 + 0j
        assert q.amplitude(i) == _approx(q, expected)


def test_apply_h_preserves_norm() -> None:
    q = Qreg(4, device="cpu")
    q.init_basis(7)
    apply_h(q, 0)
    apply_h(q, 1)
    apply_h(q, 2)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


# ---------------------------------------------------------------------------
# Pauli-X: bit flip + LSB axis check
# ---------------------------------------------------------------------------


def test_apply_x_on_zero_gives_one() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_x(q, 0)
    assert q.amplitude(0) == _approx(q, 0 + 0j)
    assert q.amplitude(1) == _approx(q, 1 + 0j)


def test_apply_x_on_one_gives_zero() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_x(q, 0)
    assert q.amplitude(0) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_x_lsb_axis_mapping() -> None:
    # |001> = amp[1]; X on qubit 0 (LSB) -> |000> = amp[0].
    q = Qreg(3, device="cpu")
    q.init_basis(1)
    apply_x(q, 0)
    assert q.amplitude(0) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_x_msb_axis_mapping() -> None:
    # |001> = amp[1]; X on qubit 2 (MSB on 3 qubits) -> |101> = amp[5].
    q = Qreg(3, device="cpu")
    q.init_basis(1)
    apply_x(q, 2)
    assert q.amplitude(5) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_x_middle_qubit() -> None:
    # |001> = amp[1]; X on qubit 1 -> |011> = amp[3].
    q = Qreg(3, device="cpu")
    q.init_basis(1)
    apply_x(q, 1)
    assert q.amplitude(3) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


# ---------------------------------------------------------------------------
# Pauli-Y
# ---------------------------------------------------------------------------


def test_apply_y_on_zero_gives_i_times_one() -> None:
    # Y|0> = i|1>
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_y(q, 0)
    assert q.amplitude(0) == _approx(q, 0 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 1j)


def test_apply_y_on_one_gives_negative_i_times_zero() -> None:
    # Y|1> = -i|0>
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_y(q, 0)
    assert q.amplitude(0) == _approx(q, 0 - 1j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


# ---------------------------------------------------------------------------
# Pauli-Z
# ---------------------------------------------------------------------------


def test_apply_z_on_zero_is_identity() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_z(q, 0)
    assert q.amplitude(0) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_z_on_one_negates() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_z(q, 0)
    assert q.amplitude(0) == _approx(q, 0 + 0j)
    assert q.amplitude(1) == _approx(q, -1 + 0j)


# ---------------------------------------------------------------------------
# S and T: S**2 == Z, T**4 == Z
# ---------------------------------------------------------------------------


def test_apply_s_squared_equals_z_on_one() -> None:
    # S|1> = i|1>; S(S|1>) = i*i|1> = -|1>. Same as Z|1>.
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_s(q, 0)
    apply_s(q, 0)
    assert q.amplitude(1) == _approx(q, -1 + 0j)


def test_apply_t_fourth_power_equals_z_on_one() -> None:
    # T|1> = e^(i pi/4)|1>; T^4|1> = e^(i pi)|1> = -|1>.
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    for _ in range(4):
        apply_t(q, 0)
    assert q.amplitude(1) == _approx(q, -1 + 0j)


# ---------------------------------------------------------------------------
# General phase gate
# ---------------------------------------------------------------------------


def test_apply_phase_pi_equals_z_on_one() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_phase(q, 0, math.pi)
    assert q.amplitude(1) == _approx(q, -1 + 0j)


def test_apply_phase_zero_is_identity() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_phase(q, 0, 0.0)
    assert q.amplitude(1) == _approx(q, 1 + 0j)


def test_apply_phase_arbitrary_angle() -> None:
    # Phase(pi/3)|1> = e^(i pi/3)|1>.
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_phase(q, 0, math.pi / 3.0)
    expected = cmath.exp(complex(0, math.pi / 3.0))
    assert q.amplitude(1) == _approx(q, expected)


# ---------------------------------------------------------------------------
# Rotations: RX, RY, RZ
# ---------------------------------------------------------------------------


def test_apply_rx_2pi_on_zero_is_minus_one() -> None:
    # RX(2 pi)|0> = -|0>. Classic sign flip after a full rotation.
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_rx(q, 0, 2 * math.pi)
    assert q.amplitude(0) == _approx(q, -1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_rx_pi_on_zero_gives_minus_i_one() -> None:
    # RX(pi)|0> = -i|1>: cos(pi/2)=0, -i sin(pi/2) = -i.
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_rx(q, 0, math.pi)
    assert q.amplitude(0) == _approx(q, 0 + 0j)
    assert q.amplitude(1) == _approx(q, 0 - 1j)


def test_apply_ry_4pi_on_zero_is_identity() -> None:
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_ry(q, 0, 4 * math.pi)
    assert q.amplitude(0) == _approx(q, 1 + 0j)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_ry_pi_on_zero_gives_one() -> None:
    # RY(pi)|0> = cos(pi/2)|0> + sin(pi/2)|1> = |1>.
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_ry(q, 0, math.pi)
    assert q.amplitude(0) == _approx(q, 0 + 0j)
    assert q.amplitude(1) == _approx(q, 1 + 0j)


def test_apply_rz_diagonal_phase_on_zero() -> None:
    # RZ(theta)|0> = e^(-i theta/2)|0>. Test at theta = pi/2.
    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_rz(q, 0, math.pi / 2.0)
    expected = cmath.exp(complex(0, -math.pi / 4.0))
    assert q.amplitude(0) == _approx(q, expected)
    assert q.amplitude(1) == _approx(q, 0 + 0j)


def test_apply_rz_diagonal_phase_on_one() -> None:
    # RZ(theta)|1> = e^(+i theta/2)|1>.
    q = Qreg(1, device="cpu")
    q.init_basis(1)
    apply_rz(q, 0, math.pi / 2.0)
    expected = cmath.exp(complex(0, math.pi / 4.0))
    assert q.amplitude(0) == _approx(q, 0 + 0j)
    assert q.amplitude(1) == _approx(q, expected)


# ---------------------------------------------------------------------------
# Method-style API: q.apply_h(0) must agree with apply_h(q, 0)
# ---------------------------------------------------------------------------


def test_method_apply_h_matches_function() -> None:
    qf = Qreg(2, device="cpu")
    qm = Qreg(2, device="cpu")
    qf.init_basis(0)
    qm.init_basis(0)
    apply_h(qf, 0)
    qm.apply_h(0)
    for i in range(4):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_method_apply_x_matches_function() -> None:
    qf = Qreg(3, device="cpu")
    qm = Qreg(3, device="cpu")
    qf.init_basis(1)  # |001>
    qm.init_basis(1)
    apply_x(qf, 0)
    qm.apply_x(0)
    for i in range(8):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_method_apply_u_matches_function() -> None:
    qf = Qreg(2, device="cpu")
    qm = Qreg(2, device="cpu")
    qf.init_basis(0)
    qm.init_basis(0)
    u = torch.tensor(
        [[INV_SQRT2 + 0j, INV_SQRT2 + 0j],
         [INV_SQRT2 + 0j, -INV_SQRT2 + 0j]],
        dtype=qf.dtype, device=qf.device,
    )
    apply_u(qf, 0, u)
    # Build a fresh tensor for the method-style call too (otherwise the
    # method-style would alias the same tensor; same result, but the
    # test is sharper when each path uses its own matrix).
    u2 = torch.tensor(
        [[INV_SQRT2 + 0j, INV_SQRT2 + 0j],
         [INV_SQRT2 + 0j, -INV_SQRT2 + 0j]],
        dtype=qm.dtype, device=qm.device,
    )
    qm.apply_u(0, u2)
    for i in range(4):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


def test_method_apply_rz_matches_function() -> None:
    qf = Qreg(1, device="cpu")
    qm = Qreg(1, device="cpu")
    qf.init_basis(0)
    qm.init_basis(0)
    apply_rz(qf, 0, 0.7)
    qm.apply_rz(0, 0.7)
    for i in range(2):
        assert qf.amplitude(i) == _approx(qf, qm.amplitude(i))


# ---------------------------------------------------------------------------
# Norm preservation: every named gate is unitary, so norm stays at 1.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate",
    ["h", "x", "y", "z", "s", "t"],
)
def test_named_gate_preserves_norm_on_basis_state(gate: str) -> None:
    q = Qreg(3, device="cpu")
    q.init_basis(3)
    {
        "h": apply_h,
        "x": apply_x,
        "y": apply_y,
        "z": apply_z,
        "s": apply_s,
        "t": apply_t,
    }[gate](q, 1)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


@pytest.mark.parametrize("theta", [0.0, 0.5, math.pi / 3.0, math.pi, 2.5])
def test_rotations_preserve_norm(theta: float) -> None:
    for apply_r in (apply_rx, apply_ry, apply_rz):
        q = Qreg(2, device="cpu")
        # Use a superposition starting state so the rotation actually
        # exercises both components, not just a basis-state phase.
        q.init_basis(0)
        apply_h(q, 0)
        apply_r(q, 0, theta)
        assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


# ---------------------------------------------------------------------------
# Device-parametrised sanity check: gates work on every available device
# ---------------------------------------------------------------------------


def test_apply_h_on_zero_across_devices(device: str) -> None:
    q = Qreg(1, device=device)
    q.init_basis(0)
    apply_h(q, 0)
    assert q.amplitude(0) == pytest.approx(
        INV_SQRT2 + 0j, abs=amp_tol_for(q.dtype)
    )
    assert q.amplitude(1) == pytest.approx(
        INV_SQRT2 + 0j, abs=amp_tol_for(q.dtype)
    )


def test_apply_x_lsb_across_devices(device: str) -> None:
    # Repeats the LSB-axis test on every available device to catch any
    # device-specific tensordot/movedim/reshape interaction.
    q = Qreg(3, device=device)
    q.init_basis(1)
    apply_x(q, 0)
    assert q.prob_of(0) == pytest.approx(1.0, abs=prob_tol_for(q.dtype))
    assert q.prob_of(1) == pytest.approx(0.0, abs=prob_tol_for(q.dtype))


def test_apply_u_custom_matrix_across_devices(device: str) -> None:
    # Builds the matrix on the same device, then applies it. This is the
    # contract: caller pre-places the matrix; gate doesn't auto-transfer.
    q = Qreg(2, device=device)
    q.init_basis(0)
    x = torch.tensor(
        [[0 + 0j, 1 + 0j],
         [1 + 0j, 0 + 0j]],
        dtype=q.dtype, device=q.device,
    )
    apply_u(q, 0, x)
    assert q.prob_of(1) == pytest.approx(1.0, abs=prob_tol_for(q.dtype))
    assert q.prob_of(0) == pytest.approx(0.0, abs=prob_tol_for(q.dtype))
