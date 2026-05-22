"""Tests for Shor's algorithm: modular exponentiation, period finding, factoring.

The four highest-signal tests:

* ``test_modular_exp_orbit_a2_mod5`` -- locks the (x, y) -> (x, a^x*y mod N)
  permutation against a hand-computed orbit table for the smallest
  non-trivial case.
* ``test_modular_exp_y_ge_N_passes_through`` -- locks the reversibility
  pass-through for y >= N.
* ``test_shor_factor_15_seeded`` -- end-to-end: factor 15 returns
  {3, 5} reproducibly with a fixed seed.
* ``test_shor_period_a2_mod21`` (gated by ``RUN_SHOR_21=1``) -- the
  canonical 16-qubit acceptance test: period of 2 mod 21 must divide 6.
"""

from __future__ import annotations

import os

import pytest

from qubit import (
    Qreg,
    amp_tol_for,
    apply_modular_exp,
    apply_shor_period,
    prob_tol_for,
    shor_factor,
)

# ===========================================================================
# apply_modular_exp: deterministic orbit table on basis states
# ===========================================================================


def test_modular_exp_orbit_a2_mod5() -> None:
    # a=2 mod 5 has period 4: 2^0=1, 2^1=2, 2^2=4, 2^3=3 (then 2^4=1).
    # Layout: counting at [0..1] (t=2), target at [2..4] (n=3), total=5.
    # For y=1 and each x in [0, 4), the modexp gate must move the
    # amplitude from (x, y=1) to (x, y=2^x mod 5).
    expected = [1, 2, 4, 3]
    for x in range(4):
        q = Qreg(5, device="cpu")
        initial = (1 << 2) | (x << 0)  # y=1, x=x
        q.init_basis(initial)
        apply_modular_exp(q, 0, 2, 2, 3, 2, 5)
        final = (expected[x] << 2) | (x << 0)
        assert q.amplitude(final) == pytest.approx(
            1 + 0j, abs=amp_tol_for(q.dtype)
        ), f"a=2 mod 5: (x={x}, y=1) -> (x, y={expected[x]})"


def test_modular_exp_y_ge_N_passes_through() -> None:
    # Layout: 1 counting + 5 target. N=5, so y in [5, 31] passes through.
    # Place amplitude at (x=0, y=10); after modexp, it should remain there.
    q = Qreg(6, device="cpu")
    q.init_basis(10 << 1)  # y=10, x=0
    apply_modular_exp(q, 0, 1, 1, 5, 2, 5)
    assert q.amplitude(10 << 1) == pytest.approx(
        1 + 0j, abs=amp_tol_for(q.dtype)
    )


@pytest.mark.parametrize("y", [5, 6, 7, 15, 31])
def test_modular_exp_y_ge_N_passes_through_each(y: int) -> None:
    # Sweep more y >= N values to ensure the pass-through region is
    # actually identity, not just the one tested point.
    q = Qreg(6, device="cpu")
    q.init_basis(y << 1)  # x=0, target = y
    apply_modular_exp(q, 0, 1, 1, 5, 2, 5)
    assert q.amplitude(y << 1) == pytest.approx(
        1 + 0j, abs=amp_tol_for(q.dtype)
    )


def test_modular_exp_preserves_norm_on_superposition() -> None:
    # Build a uniform superposition over the counting register, leave
    # y=1 in the target, then apply modexp. The result is a uniform
    # superposition over (x, y=2^x mod N) for x in [0, 2^t), so the
    # norm must still be 1.
    n = 4
    t = 4
    q = Qreg(t + n, device="cpu")
    q.init_basis(1 << t)  # x=0, y=1
    for i in range(t):
        q.apply_h(i)
    apply_modular_exp(q, 0, t, t, n, 7, 15)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


# ===========================================================================
# apply_modular_exp: validation
# ===========================================================================


def test_modular_exp_rejects_t_zero() -> None:
    q = Qreg(5, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_modular_exp: t=0"
    ):
        apply_modular_exp(q, 0, 0, 2, 3, 2, 5)


def test_modular_exp_rejects_n_zero() -> None:
    q = Qreg(5, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_modular_exp: n=0"
    ):
        apply_modular_exp(q, 0, 2, 2, 0, 2, 5)


def test_modular_exp_rejects_counting_overflow() -> None:
    q = Qreg(5, device="cpu")
    with pytest.raises(
        ValueError,
        match=r"^qubit: apply_modular_exp: counting range \[3, 7\) out of \[0, 5\)",
    ):
        apply_modular_exp(q, 3, 4, 0, 1, 2, 5)


def test_modular_exp_rejects_target_overflow() -> None:
    q = Qreg(5, device="cpu")
    with pytest.raises(
        ValueError,
        match=r"^qubit: apply_modular_exp: target range \[3, 7\) out of \[0, 5\)",
    ):
        apply_modular_exp(q, 0, 1, 3, 4, 2, 5)


def test_modular_exp_rejects_overlap() -> None:
    q = Qreg(8, device="cpu")
    with pytest.raises(
        ValueError,
        match=r"^qubit: apply_modular_exp: counting \[0, 4\) and target \[2, 6\) overlap",
    ):
        apply_modular_exp(q, 0, 4, 2, 4, 2, 5)


def test_modular_exp_rejects_small_N() -> None:
    q = Qreg(5, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: apply_modular_exp: N=1 must be >= 2"
    ):
        apply_modular_exp(q, 0, 2, 2, 3, 2, 1)


def test_modular_exp_rejects_oversized_N() -> None:
    q = Qreg(5, device="cpu")
    # n=3 target -> N must be <= 8. N=9 is over.
    with pytest.raises(
        ValueError,
        match=r"^qubit: apply_modular_exp: N=9 exceeds target capacity 2\^3",
    ):
        apply_modular_exp(q, 0, 2, 2, 3, 2, 9)


def test_modular_exp_rejects_non_coprime() -> None:
    q = Qreg(5, device="cpu")
    # gcd(6, 15) = 3 != 1.
    with pytest.raises(
        ValueError,
        match=r"^qubit: apply_modular_exp: gcd\(a=6, N=15\) != 1",
    ):
        apply_modular_exp(q, 0, 1, 1, 4, 6, 15)


# ===========================================================================
# apply_shor_period: stochastic but seeded
# ===========================================================================


def test_shor_period_a7_mod15_seeded() -> None:
    # True period of 7 mod 15 is 4. The recovered period must be a
    # divisor of 4, i.e., in {1, 2, 4}. With a fixed seed the run is
    # bit-for-bit reproducible.
    n = 4
    t = 9
    q = Qreg(t + n, device="cpu", seed=42)
    res = apply_shor_period(q, n, t, 0, n, 7, 15)
    assert res.period in {1, 2, 4}, (
        f"recovered period {res.period} not a divisor of true r=4 "
        f"(raw measured = {res.measured})"
    )


def test_shor_period_a7_mod15_multiple_seeds() -> None:
    # The function is stochastic; sweep a handful of seeds and confirm
    # every run produces either a divisor of the true order (4 for
    # a=7 mod 15) OR the period=0 "outright failure" sentinel that
    # apply_shor_period returns on a c=0 readout. If the implementation
    # were wrong (e.g. swapped LSB/MSB on the counting register,
    # off-by-one in the modexp permutation), at least one seed would
    # surface a value outside this set.
    n = 4
    t = 9
    for seed in (1, 7, 42, 100, 2026):
        q = Qreg(t + n, device="cpu", seed=seed)
        res = apply_shor_period(q, n, t, 0, n, 7, 15)
        assert res.period in {0, 1, 2, 4}, (
            f"seed={seed}: period={res.period} (measured={res.measured})"
        )


# ===========================================================================
# shor_factor: end-to-end
# ===========================================================================


def test_shor_factor_15_seeded() -> None:
    res = shor_factor(15, seed=42)
    # Both factors must multiply to 15 and lie in {3, 5}.
    assert {res.p, res.q} == {3, 5}, (
        f"shor_factor(15, seed=42) -> p={res.p}, q={res.q}; "
        f"attempts={res.attempts}"
    )
    assert res.p * res.q == 15
    assert res.attempts >= 1


def test_shor_factor_15_multiple_seeds() -> None:
    # The algorithm is stochastic; with N=15 and 20 attempts, success
    # is essentially guaranteed. Verify across several seeds.
    for seed in (1, 7, 42, 99, 2026):
        res = shor_factor(15, seed=seed)
        assert {res.p, res.q} == {3, 5}, (
            f"seed={seed}: p={res.p}, q={res.q}, attempts={res.attempts}"
        )


def test_shor_factor_even_N_short_circuits() -> None:
    # Even N is factored without any quantum step (attempts == 0).
    for N, expected_p, expected_q in [
        (4, 2, 2),
        (6, 2, 3),
        (14, 2, 7),
        (100, 2, 50),
    ]:
        res = shor_factor(N)
        assert (res.p, res.q) == (expected_p, expected_q), (
            f"shor_factor({N}): p={res.p}, q={res.q}"
        )
        assert res.attempts == 0


def test_shor_factor_seeded_is_deterministic() -> None:
    # Same seed -> same factor pair, same attempt count.
    res_a = shor_factor(15, seed=42)
    res_b = shor_factor(15, seed=42)
    assert res_a == res_b


def test_shor_factor_rejects_small_N() -> None:
    with pytest.raises(
        ValueError, match=r"^qubit: shor_factor: N=1 must be >= 2"
    ):
        shor_factor(1)


def test_shor_factor_rejects_negative_max_attempts() -> None:
    with pytest.raises(
        ValueError, match=r"^qubit: shor_factor: max_attempts=0"
    ):
        shor_factor(15, max_attempts=0)


# ===========================================================================
# Shor-21: 16-qubit period-finding test, gated by RUN_SHOR_21
# ===========================================================================


@pytest.mark.skipif(
    os.environ.get("RUN_SHOR_21") != "1",
    reason="set RUN_SHOR_21=1 to run the 16-qubit Shor-21 test",
)
def test_shor_period_a2_mod21_gated() -> None:
    # The 2 mod 21 orbit is {1, 2, 4, 8, 16, 11, 22%21=1}, so the true
    # period is 6. Continued-fraction recovery returns the denominator
    # of the best rational approximation of c/2^t with denominator <= 21,
    # which must divide 6 -- i.e., be in {1, 2, 3, 6}. The test uses
    # a FIXED a=2 (rather than running shor_factor) so the only
    # stochasticity is the quantum measurement; gated by RUN_SHOR_21=1
    # to keep the default suite fast.
    n = 5
    t = 11
    q = Qreg(t + n, device="cpu", seed=2026)
    res = apply_shor_period(q, n, t, 0, n, 2, 21)
    assert res.period in {1, 2, 3, 6}, (
        f"recovered period {res.period} not a divisor of true r=6 "
        f"(raw measured = {res.measured})"
    )


# ===========================================================================
# Method-style parity
# ===========================================================================


def test_method_apply_modular_exp_matches_function() -> None:
    # The deterministic modexp orbit case, run through both APIs.
    q1 = Qreg(5, device="cpu")
    q2 = Qreg(5, device="cpu")
    q1.init_basis((1 << 2) | (3 << 0))  # y=1, x=3
    q2.init_basis((1 << 2) | (3 << 0))
    apply_modular_exp(q1, 0, 2, 2, 3, 2, 5)
    q2.apply_modular_exp(0, 2, 2, 3, 2, 5)
    tol = amp_tol_for(q1.dtype)
    for i in range(1 << 5):
        assert q1.amplitude(i) == pytest.approx(q2.amplitude(i), abs=tol)


def test_method_apply_shor_period_matches_function() -> None:
    # Both call paths share the same RNG state initially, so the
    # measurement outcome (and therefore the recovered period) must
    # match exactly.
    n = 4
    t = 9
    q1 = Qreg(t + n, device="cpu", seed=42)
    q2 = Qreg(t + n, device="cpu", seed=42)
    res_function = apply_shor_period(q1, n, t, 0, n, 7, 15)
    res_method = q2.apply_shor_period(n, t, 0, n, 7, 15)
    assert res_function == res_method


# ===========================================================================
# Device-parametrised: modular_exp on small register on every device
# ===========================================================================


def test_modular_exp_orbit_across_devices(device: str) -> None:
    # The small a=2 mod 5 orbit on every available device. Verifies
    # the permutation tensor transfers correctly and gather works on
    # both CPU and MPS (and CUDA if present).
    expected = [1, 2, 4, 3]
    for x in range(4):
        q = Qreg(5, device=device)
        initial = (1 << 2) | (x << 0)
        q.init_basis(initial)
        apply_modular_exp(q, 0, 2, 2, 3, 2, 5)
        final = (expected[x] << 2) | (x << 0)
        assert q.prob_of(final) == pytest.approx(
            1.0, abs=prob_tol_for(q.dtype)
        )
