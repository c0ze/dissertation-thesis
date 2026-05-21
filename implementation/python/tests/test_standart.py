"""Tests for the ``qubit.standart`` arithmetic helpers.

These mirror the corresponding ``/c`` and ``/go`` tests for the shared
function set. The Shor-relevant continued-fraction cases (period of 7
mod 15, period of 2 mod 21) are exercised in their actual readout form
``(c, 2**t, N)`` so a future Shor implementation can drop these helpers
in without rewriting the test contracts.
"""

from __future__ import annotations

import pytest

from qubit import (
    continued_fraction,
    gcd_u64,
    ilog2_u32,
    is_power_of_two,
    mod_pow,
)

# ---------------------------------------------------------------------------
# gcd_u64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,want",
    [
        (0, 5, 5),
        (5, 0, 5),
        (12, 8, 4),
        (7, 13, 1),
        (15, 21, 3),
        (100, 75, 25),
        (0, 0, 0),
        # Beyond uint64: Python ints don't overflow, so gcd_u64 works on
        # arbitrarily large inputs. The function name still says "u64"
        # for parity with /c+/go, but the contract is "non-negative int".
        (10**20, 10**20 - 1, 1),
    ],
)
def test_gcd_basic(a: int, b: int, want: int) -> None:
    assert gcd_u64(a, b) == want


def test_gcd_rejects_negative_a() -> None:
    with pytest.raises(ValueError, match=r"^qubit: gcd_u64: a=-1"):
        gcd_u64(-1, 5)


def test_gcd_rejects_negative_b() -> None:
    with pytest.raises(ValueError, match=r"^qubit: gcd_u64: b=-3"):
        gcd_u64(7, -3)


def test_gcd_rejects_both_negative() -> None:
    # Either error is acceptable as long as one fires; the validation
    # is order-sensitive (a before b), so we expect the a-side message.
    with pytest.raises(ValueError, match=r"^qubit: gcd_u64: a=-5"):
        gcd_u64(-5, -7)


# ---------------------------------------------------------------------------
# mod_pow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base,exp,mod,want",
    [
        # Basic cases from the /go suite.
        (2, 10, 1000, 24),  # 1024 mod 1000
        (7, 0, 15, 1),
        (0, 0, 7, 1),  # 0**0 = 1 convention
        # Large exp -- pow handles square-and-multiply automatically.
        (2, 1 << 10, 1000000007, 812734592),
        # Mod over uint32.
        ((1 << 32) + 1, 5, (1 << 33) - 1, 5100273671),
        # mod == 1 -- everything reduces to 0.
        (5, 3, 1, 0),
        # Negative base (Python normalises): -2**3 mod 5 = -8 mod 5 = 2.
        (-2, 3, 5, 2),
    ],
)
def test_mod_pow_basic(base: int, exp: int, mod: int, want: int) -> None:
    assert mod_pow(base, exp, mod) == want


def test_mod_pow_rejects_negative_exp() -> None:
    with pytest.raises(ValueError, match=r"^qubit: mod_pow: exp=-1"):
        mod_pow(2, -1, 5)


def test_mod_pow_rejects_zero_mod() -> None:
    # mod == 0 would mean "anything mod 0", which Python's pow raises
    # ZeroDivisionError for. We surface this with the qubit: prefix.
    with pytest.raises(ValueError, match=r"^qubit: mod_pow: mod=0"):
        mod_pow(2, 5, 0)


def test_mod_pow_rejects_negative_mod() -> None:
    with pytest.raises(ValueError, match=r"^qubit: mod_pow: mod=-3"):
        mod_pow(2, 5, -3)


# ---------------------------------------------------------------------------
# continued_fraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,q,max_denom,want",
    [
        # 3/8 with a generous max-denom -- the exact denominator survives.
        (3, 8, 100, 8),
        # 1/3 with max-denom >= 3.
        (1, 3, 10, 3),
        # x == 0: the trivial 0/1 approximation, matching /go's x<=0 hatch.
        (0, 256, 15, 1),
        (0, 2048, 21, 1),
        # The fraction reduces internally: 2/4 is identical to 1/2 inside
        # Fraction(), so the returned denominator is the reduced one.
        (2, 4, 100, 2),
        # max_denom exactly equal to the fraction's reduced denominator.
        (3, 7, 7, 7),
    ],
)
def test_continued_fraction_basic(
    x: int, q: int, max_denom: int, want: int
) -> None:
    assert continued_fraction(x, q, max_denom) == want


@pytest.mark.parametrize(
    "c,want",
    [
        # /go's test_shor_period_a7_mod15 uses t=8 counting + n=4 target;
        # the true period of 7 mod 15 is 4, so c is one of {0, 64, 128,
        # 192} after the QFT. continued_fraction recovers a divisor of 4.
        (0, 1),
        (64, 4),
        (128, 2),
        (192, 4),
    ],
)
def test_continued_fraction_shor_a7_mod15(c: int, want: int) -> None:
    # Shor's period finder: c is the measured counting register, q is
    # 2**t (t=8 here), max_denominator is N (=15). The returned
    # denominator is the candidate period r.
    assert continued_fraction(c, 256, 15) == want


@pytest.mark.parametrize(
    "c,want",
    [
        # /go's test_shor_period_a2_mod21: t=11, true period r=6.
        # The QFT readout c is approximately k * 2048 / 6 for some k;
        # continued_fraction should recover a divisor of 6.
        (0, 1),
        (341, 6),   # 1/6
        (683, 3),   # 1/3
        (1024, 2),  # 1/2
        (1365, 3),  # 2/3
        (1707, 6),  # 5/6
    ],
)
def test_continued_fraction_shor_a2_mod21(c: int, want: int) -> None:
    assert continued_fraction(c, 2048, 21) == want


def test_continued_fraction_irrational_via_rational() -> None:
    # /go tests continued_fraction(sqrt(2), 10) -> denominator 5
    # (convergent 7/5). Python's signature doesn't take floats, but
    # we can express sqrt(2) ~ 14142135 / 10000000 and recover 7/5
    # via limit_denominator. Confirms the Stern-Brocot search matches
    # /go's hand-rolled convergent loop on a non-trivial irrational.
    assert continued_fraction(14142135, 10000000, 10) == 5


def test_continued_fraction_rejects_zero_q() -> None:
    with pytest.raises(ValueError, match=r"^qubit: continued_fraction: q=0"):
        continued_fraction(3, 0, 100)


def test_continued_fraction_rejects_negative_q() -> None:
    with pytest.raises(ValueError, match=r"^qubit: continued_fraction: q=-8"):
        continued_fraction(3, -8, 100)


def test_continued_fraction_rejects_zero_max_denominator() -> None:
    with pytest.raises(
        ValueError, match=r"^qubit: continued_fraction: max_denominator=0"
    ):
        continued_fraction(3, 8, 0)


def test_continued_fraction_rejects_negative_max_denominator() -> None:
    with pytest.raises(
        ValueError, match=r"^qubit: continued_fraction: max_denominator=-5"
    ):
        continued_fraction(3, 8, -5)


# ---------------------------------------------------------------------------
# is_power_of_two
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,want",
    [
        (1, True),
        (2, True),
        (4, True),
        (8, True),
        (1024, True),
        (1 << 30, True),
        (1 << 60, True),  # Python ints are unbounded; still detects.
        (0, False),
        (3, False),
        (6, False),
        (-1, False),
        (-2, False),
        (-1024, False),
    ],
)
def test_is_power_of_two(x: int, want: bool) -> None:
    assert is_power_of_two(x) is want


# ---------------------------------------------------------------------------
# ilog2_u32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,want",
    [
        (1, 0),
        (2, 1),
        (3, 1),  # floor(log2(3)) = 1
        (4, 2),
        (7, 2),  # floor(log2(7)) = 2
        (8, 3),
        (1024, 10),
        (1 << 30, 30),
        (1 << 62, 62),  # Python ints have no width limit; the name is just parity.
    ],
)
def test_ilog2(x: int, want: int) -> None:
    assert ilog2_u32(x) == want


def test_ilog2_rejects_zero() -> None:
    with pytest.raises(ValueError, match=r"^qubit: ilog2_u32: x=0"):
        ilog2_u32(0)


def test_ilog2_rejects_negative() -> None:
    with pytest.raises(ValueError, match=r"^qubit: ilog2_u32: x=-3"):
        ilog2_u32(-3)


# ---------------------------------------------------------------------------
# Public re-export surface
# ---------------------------------------------------------------------------


def test_all_helpers_exported_from_qubit() -> None:
    # Sanity: every standart helper must be importable from the top-level
    # qubit package, not only from qubit.standart.
    import qubit

    for name in (
        "gcd_u64",
        "mod_pow",
        "continued_fraction",
        "is_power_of_two",
        "ilog2_u32",
    ):
        assert hasattr(qubit, name), f"qubit.{name} missing from re-exports"
        assert name in qubit.__all__, f"qubit.{name} missing from __all__"
