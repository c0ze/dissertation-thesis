"""Tests for the Quantum Fourier Transform.

The five highest-signal tests:

* ``test_qft_on_zero_state_is_uniform`` -- QFT|0> = uniform 1/sqrt(N).
* ``test_qft_on_basis_state_matches_analytic`` -- locks the
  ``exp(2 pi i x y / N) / sqrt(N)`` formula across multiple
  ``(n, x)`` pairs and incidentally locks the bit-reversal-swap
  convention (without the final swaps the output would land at
  bit-reversed indices).
* ``test_qft_then_qft_inverse_round_trip`` -- exhaustive over basis
  states for n = 1..4, locks the inverse is a TRUE inverse.
* ``test_qft_inverse_then_qft_round_trip`` -- same, opposite order.
* ``test_qft_subregister_preserves_outer_qubits`` -- sub-register
  ``start=1, n=2`` on a 3-qubit register leaves the outer qubit
  coherent.
"""

from __future__ import annotations

import cmath
import math

import pytest

from qubit import (
    Qreg,
    amp_tol_for,
    apply_qft,
    apply_qft_inverse,
    prob_tol_for,
)

# ===========================================================================
# QFT on |0> = uniform superposition
# ===========================================================================


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_on_zero_state_is_uniform(n: int) -> None:
    q = Qreg(n, device="cpu")
    q.init_basis(0)
    apply_qft(q)
    expected = 1.0 / math.sqrt(1 << n)
    tol = amp_tol_for(q.dtype)
    for i in range(1 << n):
        assert q.amplitude(i) == pytest.approx(expected + 0j, abs=tol), (
            f"QFT|0> n={n} amp[{i}] = {q.amplitude(i)}, expect {expected}"
        )


# ===========================================================================
# QFT on |x> matches the analytic formula
# ===========================================================================


@pytest.mark.parametrize(
    "n,x",
    [
        # n=2: every basis state
        (2, 0), (2, 1), (2, 2), (2, 3),
        # n=3: a representative sweep
        (3, 0), (3, 1), (3, 3), (3, 5), (3, 7),
        # n=4: a couple of cases (full sweep is 16 -- covered by
        # round-trip tests)
        (4, 1), (4, 11),
    ],
)
def test_qft_on_basis_state_matches_analytic(n: int, x: int) -> None:
    # amp[y] = exp(2 pi i * x * y / N) / sqrt(N) for input |x>.
    # The final bit-reversal swap is what puts the output in natural
    # binary order -- without it amp would land at bit_reverse(y).
    q = Qreg(n, device="cpu")
    q.init_basis(x)
    apply_qft(q)
    big_n = 1 << n
    tol = amp_tol_for(q.dtype)
    for y in range(big_n):
        expected = cmath.exp(complex(0, 2.0 * math.pi * x * y / big_n)) / math.sqrt(big_n)
        assert q.amplitude(y) == pytest.approx(expected, abs=tol), (
            f"QFT|{x}> n={n} amp[{y}]={q.amplitude(y)}, expect {expected}"
        )


# ===========================================================================
# Round-trip: QFT then inverse, and the other order
# ===========================================================================


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_then_qft_inverse_round_trip(n: int) -> None:
    # Every basis state x must map back to |x> after QFT followed by
    # inverse-QFT. Locks that the inverse is a TRUE inverse, not an
    # equivalent "QFT-three-times" (which would also work, since QFT^4
    # = I, but at three times the gate cost -- documented in qft.py).
    tol = amp_tol_for(Qreg(1, device="cpu").dtype)
    for x in range(1 << n):
        q = Qreg(n, device="cpu")
        q.init_basis(x)
        apply_qft(q)
        apply_qft_inverse(q)
        for i in range(1 << n):
            expected = 1 + 0j if i == x else 0 + 0j
            assert q.amplitude(i) == pytest.approx(expected, abs=tol), (
                f"round-trip n={n} basis={x} amp[{i}]={q.amplitude(i)}"
            )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_qft_inverse_then_qft_round_trip(n: int) -> None:
    tol = amp_tol_for(Qreg(1, device="cpu").dtype)
    for x in range(1 << n):
        q = Qreg(n, device="cpu")
        q.init_basis(x)
        apply_qft_inverse(q)
        apply_qft(q)
        for i in range(1 << n):
            expected = 1 + 0j if i == x else 0 + 0j
            assert q.amplitude(i) == pytest.approx(expected, abs=tol), (
                f"inverse-then-forward n={n} basis={x} amp[{i}]={q.amplitude(i)}"
            )


def test_qft_preserves_norm() -> None:
    # Unitarity check: norm stays 1 through QFT, inverse, and a
    # third application (which by F^4 = I should equal F^-1 again,
    # but here we just care that norm doesn't drift).
    q = Qreg(4, device="cpu")
    q.init_basis(7)
    apply_qft(q)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))
    apply_qft_inverse(q)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


# ===========================================================================
# Sub-register
# ===========================================================================


def test_qft_subregister_preserves_outer_qubits() -> None:
    # 3-qubit register in |001> (q0=1, q1=0, q2=0). Apply QFT to the
    # sub-register [start=1, n=2] (qubits 1 and 2). The sub-state is
    # |q2 q1> = |00>, so the sub-QFT produces uniform over the four
    # 2-qubit basis states. Qubit 0 stays at 1, so the full register
    # ends up uniform over indices with q0=1: {1, 3, 5, 7}.
    q = Qreg(3, device="cpu")
    q.init_basis(1)
    apply_qft(q, start=1, n=2)
    tol = amp_tol_for(q.dtype)
    for i in range(8):
        if i & 1 == 1:  # q0 (LSB) must be 1
            assert q.amplitude(i) == pytest.approx(0.5 + 0j, abs=tol), (
                f"sub-QFT amp[{i}] = {q.amplitude(i)}, expect 0.5"
            )
        else:
            assert q.amplitude(i) == pytest.approx(0 + 0j, abs=tol), (
                f"sub-QFT amp[{i}] = {q.amplitude(i)}, expect 0 (q0=0)"
            )


def test_qft_subregister_round_trip() -> None:
    # Inverse must work on sub-registers too. Start with a random-ish
    # superposition on the outer qubit, then QFT + inverse on the
    # inner range; the result should equal the starting state.
    q = Qreg(3, device="cpu")
    q.init_basis(0)
    q.apply_h(0)  # outer qubit in superposition
    q.apply_h(2)  # one inner qubit in superposition
    snapshot = q.amplitudes_copy()
    apply_qft(q, start=1, n=2)
    apply_qft_inverse(q, start=1, n=2)
    tol = amp_tol_for(q.dtype)
    for i in range(8):
        assert q.amplitude(i) == pytest.approx(
            complex(snapshot[i].item()), abs=tol
        ), f"sub-QFT round-trip amp[{i}]"


def test_qft_default_n_covers_full_register() -> None:
    # apply_qft(q) with no start/n should cover the whole register,
    # so QFT|0...0> should give uniform 1/sqrt(2^n).
    q = Qreg(3, device="cpu")
    q.init_basis(0)
    apply_qft(q)  # equivalent to apply_qft(q, start=0, n=3)
    expected = 1.0 / math.sqrt(8)
    tol = amp_tol_for(q.dtype)
    for i in range(8):
        assert q.amplitude(i) == pytest.approx(expected + 0j, abs=tol)


# ===========================================================================
# Validation
# ===========================================================================


def test_apply_qft_rejects_negative_start() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: apply_qft: start=-1"):
        apply_qft(q, start=-1)


def test_apply_qft_rejects_zero_n() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: apply_qft: n=0"):
        apply_qft(q, n=0)


def test_apply_qft_rejects_negative_n() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(ValueError, match=r"^qubit: apply_qft: n=-1"):
        apply_qft(q, n=-1)


def test_apply_qft_rejects_range_overflow() -> None:
    q = Qreg(3, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_qft: start=1 \+ n=3 > q.n_qubits=3"
    ):
        apply_qft(q, start=1, n=3)


def test_apply_qft_inverse_rejects_range_overflow() -> None:
    # Same validation surface as the forward direction; the inverse
    # uses the same helper.
    q = Qreg(3, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_qft: start=2 \+ n=2 > q.n_qubits=3"
    ):
        apply_qft_inverse(q, start=2, n=2)


# ===========================================================================
# Method-style parity
# ===========================================================================


def test_method_apply_qft_matches_function() -> None:
    q1 = Qreg(3, device="cpu")
    q2 = Qreg(3, device="cpu")
    q1.init_basis(5)
    q2.init_basis(5)
    apply_qft(q1)
    q2.apply_qft()
    tol = amp_tol_for(q1.dtype)
    for i in range(8):
        assert q1.amplitude(i) == pytest.approx(q2.amplitude(i), abs=tol)


def test_method_apply_qft_inverse_matches_function() -> None:
    q1 = Qreg(3, device="cpu")
    q2 = Qreg(3, device="cpu")
    q1.init_basis(5)
    q2.init_basis(5)
    apply_qft_inverse(q1)
    q2.apply_qft_inverse()
    tol = amp_tol_for(q1.dtype)
    for i in range(8):
        assert q1.amplitude(i) == pytest.approx(q2.amplitude(i), abs=tol)


def test_method_qft_subregister_args() -> None:
    # Confirms that start= and n= flow through the method wrapper.
    q1 = Qreg(3, device="cpu")
    q2 = Qreg(3, device="cpu")
    q1.init_basis(1)
    q2.init_basis(1)
    apply_qft(q1, start=1, n=2)
    q2.apply_qft(start=1, n=2)
    tol = amp_tol_for(q1.dtype)
    for i in range(8):
        assert q1.amplitude(i) == pytest.approx(q2.amplitude(i), abs=tol)


# ===========================================================================
# Device-parametrised sanity (small n to keep MPS path cheap)
# ===========================================================================


def test_qft_uniform_across_devices(device: str) -> None:
    n = 3
    q = Qreg(n, device=device)
    q.init_basis(0)
    apply_qft(q)
    expected = 1.0 / math.sqrt(1 << n)
    tol = amp_tol_for(q.dtype)
    for i in range(1 << n):
        assert q.amplitude(i) == pytest.approx(expected + 0j, abs=tol)


def test_qft_round_trip_across_devices(device: str) -> None:
    n = 2
    q = Qreg(n, device=device)
    q.init_basis(1)
    apply_qft(q)
    apply_qft_inverse(q)
    tol = amp_tol_for(q.dtype)
    assert q.amplitude(1) == pytest.approx(1 + 0j, abs=tol)
    for i in (0, 2, 3):
        assert q.amplitude(i) == pytest.approx(0 + 0j, abs=tol)
