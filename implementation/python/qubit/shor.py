"""Shor's algorithm: modular exponentiation, period finding, factoring.

Three public entry points:

* :func:`apply_modular_exp` -- the reversible map
  ``(x, y) -> (x, (a**x * y) mod N)`` for ``y < N``, identity otherwise.
  Implemented as a single ``torch.gather`` with a precomputed integer
  permutation. The permutation is built on CPU because PyTorch's
  bitwise integer ops have inconsistent backend coverage (CUDA / ROCm /
  MPS); a contiguous device tensor is materialised once and gathered.
* :func:`apply_shor_period` -- the quantum period-finding subroutine.
  Initialises ``q`` to ``|0>_counting |1>_target``, runs the Shor
  circuit (``H^t -> modular_exp -> QFT^-1 -> measure``), and recovers
  a candidate period via continued-fraction expansion of ``c / 2^t``.
* :func:`shor_factor` -- end-to-end factoring with retry loop. Picks
  random ``a`` coprime to ``N``, invokes the quantum period finder,
  derives a factor via ``gcd(a**(r/2) +/- 1, N)``. Allocates its own
  ``Qreg``; size ``t + n`` with ``n = ceil(log2 N)``, ``t = 2n + 1``
  per the standard Shor recipe.

Scope note: v1 covers Shor-15 (12-qubit register) and Shor-21
(16-qubit register) comfortably -- both well under a second on CPU
including the CPU-side permutation build. 25-qubit modular
exponentiation is NOT a practical target for this implementation: the
permutation tensor would be 2^25 * 8 = 256 MiB on CPU before transfer
to device, and the gather output is another 2^25 * 16 = 512 MiB. The
arithmetic is correct at that scale, but call sites should consider
the memory cost.
"""

from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING

import torch

from ._assert import raise_value
from .gates_single import apply_h
from .qft import apply_qft_inverse
from .standart import continued_fraction, gcd_u64, mod_pow

if TYPE_CHECKING:
    from .qreg import Qreg


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ShorPeriodResult:
    """Result of one run of :func:`apply_shor_period`.

    Attributes:
        period: candidate period ``r`` recovered from the continued
            fraction expansion of ``measured / 2**t``. Zero if the
            recovery fell off the edge (degenerate measurement, unlikely
            for a well-sized counting register).
        measured: the raw measurement of the counting register (an
            integer in ``[0, 2**t)``). Exposed for debugging and for
            tests that want to inspect the underlying randomness rather
            than the recovered period.
    """

    period: int
    measured: int


@dataclasses.dataclass(frozen=True)
class ShorFactorResult:
    """Result of :func:`shor_factor`.

    Attributes:
        p, q: the two factors of ``N``. On success, ``p * q == N`` and
            both ``p, q > 1``. On failure (after exhausting
            ``max_attempts``), both fields are ``0``.
        attempts: number of period-finding rounds consumed. ``0`` for
            the trivial-even-``N`` short-circuit path; otherwise the
            attempt number on which a factor was found (or
            ``max_attempts`` on total failure).
    """

    p: int
    q: int
    attempts: int


# ---------------------------------------------------------------------------
# apply_modular_exp: the reversible (x, y) -> (x, a^x * y mod N) map.
# ---------------------------------------------------------------------------


def _build_modexp_perm(
    n_total: int,
    counting_start: int,
    t: int,
    target_start: int,
    n: int,
    a: int,
    N: int,
) -> torch.Tensor:
    """Build the gather permutation for one apply_modular_exp call.

    Returns a 1-D ``int64`` tensor of length ``2**n_total`` such that
    ``perm[i_new] == i_old`` for every input/output pair.

    The build is vectorised on CPU. Memory is one int64 tensor of size
    ``2**n_total``; for Shor-21 (n_total=16) that's 512 KiB, trivial.
    The ``pow(a, x, N)`` precompute is ``2**t`` Python int calls and
    runs in microseconds at our sizes.
    """
    size = 1 << n_total
    t_mask = (1 << t) - 1
    n_mask = (1 << n) - 1
    target_clear_mask = n_mask << target_start
    # Mask of bits to keep when reconstructing i_new: everything except
    # the target-register bits, within the n_total-bit window.
    keep_mask = (size - 1) & ~target_clear_mask

    # Precompute a^x mod N for x in [0, 2**t). Python's three-argument
    # pow is fast (C-level) and exact.
    a_pow_table = torch.tensor(
        [pow(a, x, N) for x in range(1 << t)],
        dtype=torch.int64,
    )

    idx = torch.arange(size, dtype=torch.int64)
    x = (idx >> counting_start) & t_mask
    y = (idx >> target_start) & n_mask

    # Use the precomputed table; advanced indexing into a 1-D tensor
    # is portable across backends.
    a_pow_x = a_pow_table[x]
    y_new_in_ring = (a_pow_x * y) % N
    # Pass-through for y >= N keeps the gate a bijection on the full
    # 2**n target space (required for reversibility / unitarity).
    y_new = torch.where(y < N, y_new_in_ring, y)

    i_new = (idx & keep_mask) | (y_new << target_start)

    # Scatter: perm[i_new[k]] = idx[k]. The map is a bijection because
    # gcd(a, N) == 1 makes y -> (a^x * y) mod N invertible on [0, N),
    # extended by identity on [N, 2^n). So i_new contains every value
    # in [0, size) exactly once and the fancy-indexing assignment is
    # well-defined (no duplicate destinations).
    perm = torch.empty(size, dtype=torch.int64)
    perm[i_new] = idx
    return perm


def apply_modular_exp(
    q: Qreg,
    counting_start: int,
    t: int,
    target_start: int,
    n: int,
    a: int,
    N: int,
) -> None:
    """Apply the reversible modular-exponentiation gate to ``q``.

    The counting register lives at qubits ``[counting_start, counting_start + t)``
    and stores ``x``. The target register lives at qubits
    ``[target_start, target_start + n)`` and stores ``y``. The action is

        (x, y) -> (x, (a**x * y) mod N)    for y < N
        (x, y) -> (x, y)                    for y >= N (pass-through)

    The two registers must not overlap, and ``gcd(a, N) == 1`` is
    required (otherwise the map is not a bijection and the operation
    isn't unitary).

    Implementation: builds the permutation tensor on CPU via
    :func:`_build_modexp_perm`, transfers it to ``q._device``, and
    applies a single ``torch.gather``. One state-vector allocation
    (the gather output) per call.
    """
    raise_value(t >= 1, "apply_modular_exp: t=%d must be >= 1", t)
    raise_value(n >= 1, "apply_modular_exp: n=%d must be >= 1", n)
    raise_value(
        counting_start >= 0,
        "apply_modular_exp: counting_start=%d must be >= 0", counting_start,
    )
    raise_value(
        target_start >= 0,
        "apply_modular_exp: target_start=%d must be >= 0", target_start,
    )
    raise_value(
        counting_start + t <= q._n,
        "apply_modular_exp: counting range [%d, %d) out of [0, %d)",
        counting_start, counting_start + t, q._n,
    )
    raise_value(
        target_start + n <= q._n,
        "apply_modular_exp: target range [%d, %d) out of [0, %d)",
        target_start, target_start + n, q._n,
    )
    c_end = counting_start + t
    t_end = target_start + n
    raise_value(
        c_end <= target_start or t_end <= counting_start,
        "apply_modular_exp: counting [%d, %d) and target [%d, %d) overlap",
        counting_start, c_end, target_start, t_end,
    )
    raise_value(N >= 2, "apply_modular_exp: N=%d must be >= 2", N)
    raise_value(
        N <= 1 << n,
        "apply_modular_exp: N=%d exceeds target capacity 2^%d", N, n,
    )
    raise_value(a >= 0, "apply_modular_exp: a=%d must be >= 0", a)
    raise_value(
        gcd_u64(a, N) == 1,
        "apply_modular_exp: gcd(a=%d, N=%d) != 1 (not coprime)", a, N,
    )

    perm = _build_modexp_perm(
        q._n, counting_start, t, target_start, n, a, N
    ).to(q._device)
    q._amp = q._amp.gather(0, perm)


# ---------------------------------------------------------------------------
# apply_shor_period: one period-finding round.
# ---------------------------------------------------------------------------


def apply_shor_period(
    q: Qreg,
    counting_start: int,
    t: int,
    target_start: int,
    n: int,
    a: int,
    N: int,
) -> ShorPeriodResult:
    """Run one period-finding round of Shor's algorithm on ``q``.

    Re-initialises ``q`` (so the caller doesn't need to set it up
    first): the counting register is zeroed, the target register is set
    to ``|1>``, and any other qubits are zeroed. Then runs the standard
    Shor circuit:

    1. ``H`` on every counting qubit (uniform superposition over ``x``).
    2. :func:`apply_modular_exp` (correlates ``y`` with ``a**x mod N``).
    3. :func:`qubit.qft.apply_qft_inverse` on the counting register
       (concentrates probability mass at multiples of ``2**t / r``).
    4. Measure all qubits; extract the counting register's value.
    5. Recover candidate period via :func:`qubit.continued_fraction`
       applied to ``measured / 2**t`` with denominator bound ``N``.

    Caller validation: ``a`` and ``N`` must satisfy
    :func:`apply_modular_exp`'s preconditions (in particular
    ``gcd(a, N) == 1``). Counting and target ranges must not overlap.

    Returns a :class:`ShorPeriodResult`. The recovered ``period`` may be
    a divisor of the true order rather than the order itself; the
    caller (typically :func:`shor_factor`) is responsible for filtering
    odd / unlucky periods.
    """
    # Initialise state. apply_modular_exp validates a, N, ranges, so we
    # don't pre-check here -- delegation keeps the message provenance
    # clear if something is wrong.
    q.init_basis(1 << target_start)

    # Step 1: uniform superposition over the counting register.
    for i in range(t):
        apply_h(q, counting_start + i)

    # Step 2: modular-exponentiation oracle.
    apply_modular_exp(q, counting_start, t, target_start, n, a, N)

    # Step 3: inverse QFT on the counting register.
    apply_qft_inverse(q, counting_start, t)

    # Step 4: measure. We measure ALL qubits and then bit-extract the
    # counting register's value, which is simpler than implementing a
    # sub-register measurement helper. The full measurement collapses
    # everything but Shor doesn't need the post-state -- the period
    # comes from the counting bits' value alone.
    full = q.measure_all()
    t_mask = (1 << t) - 1
    measured = (full >> counting_start) & t_mask

    # Step 5: continued-fraction recovery of the candidate period.
    # Already shipping: continued_fraction(c, q, max_denom) returns the
    # denominator of the best rational approximation of c/q with
    # denominator <= max_denom. For Shor that bound is N (the modulus
    # being factored).
    period = continued_fraction(measured, 1 << t, N)
    return ShorPeriodResult(period=period, measured=measured)


# ---------------------------------------------------------------------------
# shor_factor: end-to-end factoring with retry loop.
# ---------------------------------------------------------------------------


def shor_factor(
    N: int,
    max_attempts: int = 20,
    seed: int | None = None,
) -> ShorFactorResult:
    """Factor ``N`` end-to-end via Shor's algorithm.

    Args:
        N: the number to factor. Must be >= 2.
        max_attempts: maximum number of period-finding rounds before
            declaring failure. Default 20 is comfortable for Shor-15;
            ~10 attempts is the realistic median success threshold.
        seed: optional integer for reproducibility. Threads through to
            both the Python ``random.Random`` driving the choice of
            ``a`` AND each constructed Qreg's measurement RNG, so
            seeded runs are bit-for-bit reproducible.

    Returns a :class:`ShorFactorResult`. On the trivial-even-``N``
    short-circuit ``attempts == 0``; otherwise ``attempts`` is the
    attempt number on which a factor was found, or ``max_attempts`` on
    failure.

    Scope caveat: this function does NOT detect prime powers
    (``N = p**k`` for prime ``p``, ``k >= 2``). For those inputs, the
    period-finding step has no useful order to recover, every attempt
    fails the ``a**(r/2) != N - 1`` check or returns ``period == 0``,
    and the function exhausts ``max_attempts`` then reports failure.
    The classical pre-check (try ``round(N**(1/k)) ** k == N`` for
    every ``k`` in ``[2, log2 N]``) belongs in the caller, matching
    ``/c`` and ``/go``'s scope. Even ``N`` is handled here (short-
    circuit).

    Algorithm:

    1. Validate ``N >= 2`` and ``max_attempts >= 1``.
    2. If ``N`` is even, return ``(2, N // 2)`` without invoking the
       quantum subroutine.
    3. Otherwise pick a random ``a`` in ``[2, N - 1]``. If
       ``gcd(a, N) > 1`` we already have a non-trivial factor.
    4. Run :func:`apply_shor_period` with that ``a`` on a freshly
       allocated ``Qreg(t + n)`` where ``n = ceil(log2 N)`` and
       ``t = 2n + 1`` (standard Shor recipe).
    5. Check the recovered period ``r``: it must be non-zero and even,
       and ``a**(r/2) mod N`` must not equal ``N - 1`` (otherwise the
       gcd step yields only trivial factors).
    6. Compute ``gcd(a**(r/2) +/- 1, N)``; if either gives a
       non-trivial factor, return it.
    7. On failure, loop back to step 3 with a fresh ``a`` and a fresh
       Qreg.
    """
    raise_value(N >= 2, "shor_factor: N=%d must be >= 2", N)
    raise_value(
        max_attempts >= 1,
        "shor_factor: max_attempts=%d must be >= 1", max_attempts,
    )

    # Even-N short-circuit. Returns sorted (small, large) for
    # determinism.
    if N % 2 == 0:
        return ShorFactorResult(p=2, q=N // 2, attempts=0)

    # Register sizing: n is the smallest number of bits that can
    # represent every value in [0, N). For N=15 that's 4 (since
    # 14 = 0b1110). For N=21 that's 5 (since 20 = 0b10100). The
    # counting register gets t = 2n + 1 bits, the standard Shor recipe
    # that guarantees >= 1/2 success probability per attempt.
    n = (N - 1).bit_length()
    t_bits = 2 * n + 1
    n_total = t_bits + n

    # The python `random.Random(seed)` drives BOTH the choice of `a`
    # AND the seed for each constructed Qreg's measurement RNG. Seeded
    # runs are bit-for-bit reproducible.
    rng = random.Random(seed)

    # Lazy import: shor.py is at the same level as qreg.py and qreg.py
    # has method wrappers that import shor lazily. Doing it the other
    # direction with TYPE_CHECKING + lazy here keeps both directions
    # cycle-free at module load.
    from .qreg import Qreg

    for attempt in range(1, max_attempts + 1):
        a = rng.randint(2, N - 1)
        # Derive a Qreg seed from the same RNG so the full call is
        # reproducible given just `seed`. Use a wide enough range to
        # exercise the full 64-bit measurement RNG space.
        qreg_seed = rng.randint(0, (1 << 63) - 1)

        # Lucky-shortcut: if a is not coprime to N, gcd itself gives a
        # non-trivial factor (because gcd(a, N) divides N and 1 < gcd
        # < N when gcd != 1).
        g = gcd_u64(a, N)
        if g != 1:
            return ShorFactorResult(p=g, q=N // g, attempts=attempt)

        # Allocate a fresh register; allow seeding for reproducibility.
        # check_memory=False because we already validated n_total
        # implicitly above (Shor-15 = 13, Shor-21 = 16, both small).
        q = Qreg(n_total, seed=qreg_seed)
        res = apply_shor_period(q, n, t_bits, 0, n, a, N)
        r = res.period

        if r == 0 or r % 2 != 0:
            continue

        # Verify r is actually a period of a mod N. Continued-fraction
        # recovery can return a divisor of the true order; the gcd
        # extraction below would just produce trivial factors in that
        # case and we'd retry, but it's cheaper to filter upfront and
        # matches the verification promise documented on apply_shor_period.
        if mod_pow(a, r, N) != 1:
            continue

        half = mod_pow(a, r // 2, N)
        if half == N - 1:
            # a^(r/2) ≡ -1 (mod N) -- the gcd step would only yield
            # trivial factors (1 or N). Retry with a different `a`.
            continue

        # gcd(a^(r/2) + 1, N) or gcd(a^(r/2) - 1, N) gives a factor.
        candidate = gcd_u64(half + 1, N)
        if 1 < candidate < N:
            return ShorFactorResult(
                p=candidate, q=N // candidate, attempts=attempt
            )
        if half > 0:
            candidate = gcd_u64(half - 1, N)
            if 1 < candidate < N:
                return ShorFactorResult(
                    p=candidate, q=N // candidate, attempts=attempt
                )

    return ShorFactorResult(p=0, q=0, attempts=max_attempts)
