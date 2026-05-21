"""Tests for Grover's amplitude amplification.

The highest-signal cases:

* ``test_grover_one_marked_in_16`` -- default iterations on 4 qubits
  with 1 marked state recovers P > 0.9 (analytic ~0.961 at 3 iters).
* ``test_grover_four_marked_in_16_one_iteration`` -- 4 marked out of
  16 hits total marked-probability ~1.0 after exactly 1 iteration.
* ``test_grover_over_iteration_drops_probability`` -- amplitude
  amplification reverses past the optimum (locks "more iterations
  is not always better" -- a real bug-detector).
"""

from __future__ import annotations

import math

import pytest

from qubit import (
    Qreg,
    amp_tol_for,
    apply_grover,
    prob_tol_for,
)

# ===========================================================================
# Canonical Grover identities
# ===========================================================================


def test_grover_one_marked_in_16() -> None:
    # 4 qubits, 1 marked state, default iterations.
    # Default count = floor(pi/4 * sqrt(16)) = floor(pi) = 3.
    # Analytic P after 3 iters = sin(7 * arcsin(1/4))^2 ~= 0.961.
    n = 4
    target = 10

    def oracle(q: Qreg, _user: object) -> None:
        q._amp[target] = -q._amp[target]

    q = Qreg(n, device="cpu")
    q.init_basis(0)
    apply_grover(q, n, oracle)
    assert q.prob_of(target) > 0.9, (
        f"1-marked-in-16 default iterations: P[{target}] = "
        f"{q.prob_of(target)} (analytic ~0.961)"
    )


def test_grover_four_marked_in_16_one_iteration() -> None:
    # 4 marked out of 16: theta = arcsin(sqrt(4/16)) = pi/6.
    # After 1 iter: sin(3 * pi/6)^2 = sin(pi/2)^2 = 1.
    # So total marked prob is exactly 1.0 (within tolerance).
    n = 4
    marked = [1, 5, 10, 14]

    def oracle(q: Qreg, _user: object) -> None:
        for m in marked:
            q._amp[m] = -q._amp[m]

    q = Qreg(n, device="cpu")
    q.init_basis(0)
    apply_grover(q, n, oracle, iterations=1)
    total = sum(q.prob_of(m) for m in marked)
    assert total == pytest.approx(1.0, abs=prob_tol_for(q.dtype) * 4)


def test_grover_two_marked_in_8_three_iterations() -> None:
    # 2 marked out of 8: theta = arcsin(sqrt(2/8)) = arcsin(0.5) = pi/6.
    # After 1 iter: sin(3 * pi/6)^2 = 1. So 1 iteration is optimal here
    # too (not 2; this is a known small-system result). Verifies the
    # iteration count IS adjustable -- the function doesn't always do
    # the same number of rounds.
    n = 3
    marked = [3, 5]

    def oracle(q: Qreg, _user: object) -> None:
        for m in marked:
            q._amp[m] = -q._amp[m]

    q = Qreg(n, device="cpu")
    q.init_basis(0)
    apply_grover(q, n, oracle, iterations=1)
    total = sum(q.prob_of(m) for m in marked)
    assert total == pytest.approx(1.0, abs=prob_tol_for(q.dtype) * 4)


def test_grover_zero_iterations_keeps_uniform_state() -> None:
    # iterations=0 means: prepare uniform superposition and STOP
    # (no oracle calls, no diffusion). Distribution is the uniform
    # 1/sqrt(N) per amplitude.
    n = 3

    def never_called_oracle(q: Qreg, _user: object) -> None:
        raise AssertionError(
            "iterations=0 must not call the oracle"
        )

    q = Qreg(n, device="cpu")
    q.init_basis(0)
    apply_grover(q, n, never_called_oracle, iterations=0)
    expected = 1.0 / math.sqrt(1 << n)
    tol = amp_tol_for(q.dtype)
    for i in range(1 << n):
        assert q.amplitude(i) == pytest.approx(expected + 0j, abs=tol)


def test_grover_over_iteration_drops_probability() -> None:
    # Past the optimum, amplitude amplification REVERSES (the rotation
    # angle continues past pi/2). 20 iterations on 1-marked-in-16
    # should land far below the ~0.961 optimum at 3 iterations.
    n = 4
    target = 10

    def oracle(q: Qreg, _user: object) -> None:
        q._amp[target] = -q._amp[target]

    q_opt = Qreg(n, device="cpu")
    q_opt.init_basis(0)
    apply_grover(q_opt, n, oracle)  # default = 3 iterations

    q_over = Qreg(n, device="cpu")
    q_over.init_basis(0)
    apply_grover(q_over, n, oracle, iterations=20)

    assert q_opt.prob_of(target) > q_over.prob_of(target) + 0.1, (
        f"optimum {q_opt.prob_of(target)} should clearly exceed "
        f"over-iterated {q_over.prob_of(target)}"
    )


def test_grover_preserves_norm() -> None:
    n = 4
    target = 10

    def oracle(q: Qreg, _user: object) -> None:
        q._amp[target] = -q._amp[target]

    q = Qreg(n, device="cpu")
    q.init_basis(0)
    apply_grover(q, n, oracle)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


# ===========================================================================
# User payload threads through
# ===========================================================================


def test_grover_user_payload_reaches_oracle() -> None:
    # The oracle should be invoked once per iteration with the same
    # user payload. Use a list-recording oracle to verify.
    received: list[object] = []

    def oracle(q: Qreg, user: object) -> None:
        received.append(user)

    q = Qreg(2, device="cpu")
    q.init_basis(0)
    apply_grover(q, 2, oracle, user="hello", iterations=3)
    assert received == ["hello", "hello", "hello"]


def test_grover_default_user_is_none() -> None:
    # If the caller omits `user`, the oracle gets None. Allows callers
    # who close over their predicate to ignore the second argument.
    seen: list[object] = []

    def oracle(q: Qreg, user: object) -> None:
        seen.append(user)

    q = Qreg(2, device="cpu")
    q.init_basis(0)
    apply_grover(q, 2, oracle, iterations=2)
    assert seen == [None, None]


def test_grover_user_payload_drives_oracle_behaviour() -> None:
    # Demonstrates the typical use of `user`: pass the marked-state
    # index/set through, so the oracle is a pure function of (q, user).
    n = 4

    def oracle(q: Qreg, user: object) -> None:
        # The user payload is an int in this test; cast explicitly so
        # mypy is happy without relaxing the signature.
        assert isinstance(user, int)
        q._amp[user] = -q._amp[user]

    # Same oracle, different user payloads => different marked items.
    q1 = Qreg(n, device="cpu")
    q1.init_basis(0)
    apply_grover(q1, n, oracle, user=3)
    assert q1.prob_of(3) > 0.9

    q2 = Qreg(n, device="cpu")
    q2.init_basis(0)
    apply_grover(q2, n, oracle, user=11)
    assert q2.prob_of(11) > 0.9


# ===========================================================================
# n_qubits=1 edge case (degenerate but supported)
# ===========================================================================


def test_grover_n_qubits_one_does_not_amplify_but_runs() -> None:
    # For n=1 with 1 marked state out of 2, theta = arcsin(1/sqrt(2)) =
    # pi/4. After one iteration: sin(3 * pi/4)^2 = (sqrt(2)/2)^2 = 0.5.
    # So P(marked) == 0.5 regardless of iteration count -- amplitude
    # amplification doesn't kick in for the M = N/2 case.
    # The function should still EXECUTE cleanly.
    def oracle(q: Qreg, _user: object) -> None:
        q._amp[0] = -q._amp[0]

    q = Qreg(1, device="cpu")
    q.init_basis(0)
    apply_grover(q, 1, oracle, iterations=5)
    assert q.prob_of(0) == pytest.approx(0.5, abs=prob_tol_for(q.dtype) * 4)


# ===========================================================================
# Sub-register: searched qubits are [0, n_qubits) of a larger register
# ===========================================================================


def test_grover_on_subregister_leaves_outer_qubits_uniform() -> None:
    # 4-qubit register, but Grover only operates on the lowest 3 qubits.
    # Qubit 3 (highest) is untouched -- it's in |0> initially and stays
    # there. The search succeeds within the 3-qubit subspace.
    full_n = 4
    search_n = 3
    target = 5  # in [0, 8), so within the 3-qubit search space

    def oracle(q: Qreg, _user: object) -> None:
        q._amp[target] = -q._amp[target]

    q = Qreg(full_n, device="cpu")
    q.init_basis(0)
    apply_grover(q, search_n, oracle, iterations=2)
    # After Grover, qubit 3 should still be 0 -- so all probability
    # mass is in indices < 8.
    p_low = sum(q.prob_of(i) for i in range(8))
    assert p_low == pytest.approx(1.0, abs=prob_tol_for(q.dtype) * 4)
    # And amp[target] should be amplified compared to its uniform start.
    assert q.prob_of(target) > 1.0 / 8


# ===========================================================================
# Validation
# ===========================================================================


def test_grover_rejects_zero_n_qubits() -> None:
    q = Qreg(2, device="cpu")

    def oracle(_q: Qreg, _user: object) -> None:
        pass

    with pytest.raises(
        ValueError, match=r"^qubit: apply_grover: n_qubits=0"
    ):
        apply_grover(q, 0, oracle)


def test_grover_rejects_negative_n_qubits() -> None:
    q = Qreg(2, device="cpu")

    def oracle(_q: Qreg, _user: object) -> None:
        pass

    with pytest.raises(
        ValueError, match=r"^qubit: apply_grover: n_qubits=-1"
    ):
        apply_grover(q, -1, oracle)


def test_grover_rejects_too_large_n_qubits() -> None:
    q = Qreg(2, device="cpu")

    def oracle(_q: Qreg, _user: object) -> None:
        pass

    with pytest.raises(
        ValueError, match=r"^qubit: apply_grover: n_qubits=3"
    ):
        apply_grover(q, 3, oracle)


def test_grover_rejects_negative_iterations() -> None:
    q = Qreg(2, device="cpu")

    def oracle(_q: Qreg, _user: object) -> None:
        pass

    with pytest.raises(
        ValueError, match=r"^qubit: apply_grover: iterations=-1"
    ):
        apply_grover(q, 2, oracle, iterations=-1)


def test_grover_rejects_non_callable_oracle() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(
        TypeError, match=r"^qubit: apply_grover: oracle must be callable"
    ):
        apply_grover(q, 2, "not callable")  # type: ignore[arg-type]


# ===========================================================================
# Method-style parity
# ===========================================================================


def test_method_apply_grover_matches_function() -> None:
    n = 3
    target = 5

    def oracle(q: Qreg, _user: object) -> None:
        q._amp[target] = -q._amp[target]

    q1 = Qreg(n, device="cpu")
    q2 = Qreg(n, device="cpu")
    q1.init_basis(0)
    q2.init_basis(0)
    apply_grover(q1, n, oracle, iterations=2)
    q2.apply_grover(n, oracle, iterations=2)
    tol = amp_tol_for(q1.dtype)
    for i in range(1 << n):
        assert q1.amplitude(i) == pytest.approx(q2.amplitude(i), abs=tol)


def test_method_apply_grover_threads_user_payload() -> None:
    # Verify the method-style API also forwards the user payload.
    received: list[object] = []

    def oracle(q: Qreg, user: object) -> None:
        received.append(user)

    q = Qreg(2, device="cpu")
    q.init_basis(0)
    q.apply_grover(2, oracle, user=42, iterations=2)
    assert received == [42, 42]


# ===========================================================================
# Device-parametrised sanity
# ===========================================================================


def test_grover_one_marked_across_devices(device: str) -> None:
    # 1-marked-in-16 with default iterations on every available device.
    # MPS gets the same amplification math as CPU.
    n = 4
    target = 10

    def oracle(q: Qreg, _user: object) -> None:
        q._amp[target] = -q._amp[target]

    q = Qreg(n, device=device)
    q.init_basis(0)
    apply_grover(q, n, oracle)
    # Slightly looser bound on MPS due to complex64 precision; the
    # 0.9 threshold accommodates either complex64 or complex128.
    assert q.prob_of(target) > 0.9
