"""Arithmetic helpers shared by gates and algorithms.

The misspelled filename ``standart`` is parity with the original 2004
``/c`` thesis code and the ``/go`` sibling; keeping the misspelling
makes the cross-implementation correspondence visually obvious.

Unlike ``/c`` and ``/go`` -- both of which need ``add_mod``/``mul_mod``
gymnastics to keep ``uint64`` arithmetic from overflowing -- Python's
``int`` is arbitrary-precision, so the operations here are thin wrappers
around stdlib builtins (``math.gcd``, ``pow``,
``fractions.Fraction.limit_denominator``) with a uniform validation
layer that raises ``qubit:``-prefixed exceptions.

Function names keep the ``_u64`` / ``_u32`` suffixes from ``/c``+``/go``
even though Python ints have no such limit; the suffix preserves the
visual parallel and signals the intended non-negative input domain.

Public surface re-exported from :mod:`qubit`:

* :func:`gcd_u64`
* :func:`mod_pow`
* :func:`continued_fraction`
* :func:`is_power_of_two`
* :func:`ilog2_u32`
"""

from __future__ import annotations

import math
from fractions import Fraction

from ._assert import raise_value


def gcd_u64(a: int, b: int) -> int:
    """Return ``gcd(a, b)`` for non-negative ``a`` and ``b``.

    Thin wrapper around :func:`math.gcd` with explicit validation that
    both inputs are non-negative -- ``math.gcd`` accepts negatives but
    the ``/c``+``/go`` siblings work over ``uint64`` and the algorithm
    is only meaningful here for non-negative integers. ``gcd(0, n)``
    returns ``n``; ``gcd(0, 0)`` returns ``0``.
    """
    raise_value(a >= 0, "gcd_u64: a=%d must be non-negative", a)
    raise_value(b >= 0, "gcd_u64: b=%d must be non-negative", b)
    return math.gcd(a, b)


def mod_pow(base: int, exp: int, mod: int) -> int:
    """Return ``base**exp mod mod`` using Python's three-argument :func:`pow`.

    Python's builtin is the canonical fast modular exponentiation; this
    function adds the ``qubit:``-prefixed validation that gates and
    Shor's period finder rely on.

    Validates ``exp >= 0`` and ``mod >= 1``. Negative ``base`` is
    accepted (Python normalises via ``base % mod`` internally), and the
    ``0**0 = 1`` convention from :func:`pow` is preserved.
    """
    raise_value(exp >= 0, "mod_pow: exp=%d must be >= 0", exp)
    raise_value(mod >= 1, "mod_pow: mod=%d must be >= 1", mod)
    return pow(base, exp, mod)


def continued_fraction(x: int, q: int, max_denominator: int) -> int:
    """Return the denominator of the best rational approximation of
    ``x / q`` whose denominator is at most ``max_denominator``.

    Used by Shor's period-finding step: after measuring the counting
    register to get ``c`` in ``[0, 2**t)``, the candidate period ``r``
    is the denominator of the best rational ``p / r`` approximation of
    ``c / 2**t`` with ``r <= N``. This function takes the
    ``(numerator, denominator)`` pair directly instead of a float, so it
    avoids the float-precision tap-dance that ``/c`` and ``/go`` have to
    work around with their early-exit ``frac < 1e-12`` guards.

    Implementation delegates to :meth:`fractions.Fraction.limit_denominator`,
    which performs the same Stern-Brocot convergent search that ``/c``+
    ``/go`` spell out by hand. Validated to match the siblings on every
    Shor case we have (see :file:`tests/test_standart.py`).

    Returns ``1`` for ``x == 0`` (the trivial ``0 / 1`` approximation),
    which matches the ``/go`` ``x <= 0`` safety hatch used at the
    Shor-readout boundary.
    """
    raise_value(q > 0, "continued_fraction: q=%d must be > 0", q)
    raise_value(
        max_denominator > 0,
        "continued_fraction: max_denominator=%d must be > 0",
        max_denominator,
    )
    return Fraction(x, q).limit_denominator(max_denominator).denominator


def is_power_of_two(x: int) -> bool:
    """Return ``True`` iff ``x`` is a positive power of two (1, 2, 4, 8, ...).

    Returns ``False`` for zero and for negative integers. Matches the
    ``/c``+``/go`` ``IsPowerOfTwo`` semantics exactly.
    """
    return x > 0 and (x & (x - 1)) == 0


def ilog2_u32(x: int) -> int:
    """Return ``floor(log2(x))`` for ``x >= 1``.

    Uses :meth:`int.bit_length` minus one -- the canonical Python idiom
    for integer log-base-2. The ``_u32`` suffix preserves parity with
    ``/c``+``/go``, where the argument was constrained to unsigned
    32-bit; Python ints have no such bound, but the name keeps the
    cross-implementation correspondence intact.

    Raises ``ValueError`` for ``x <= 0`` (``log2(0)`` is undefined and
    the function is meaningless on negative inputs).
    """
    raise_value(x > 0, "ilog2_u32: x=%d must be > 0", x)
    return x.bit_length() - 1
