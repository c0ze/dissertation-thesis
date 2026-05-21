"""Tests for measurement, sampling, clone, and dump.

Probabilistic tests use a pinned seed and the per-Qreg CPU RNG; the
sampling helpers are deterministic given a seed because measurement
sampling crosses to CPU at the readout boundary (see Phase 0+1's
design rationale).
"""

from __future__ import annotations

import math
from collections import Counter

import pytest
import torch

from qubit import (
    Qreg,
    amp_tol_for,
    apply_h,
    apply_x,
    clone,
    dump,
    measure_all,
    measure_qubit,
    prob_tol_for,
    sample_distribution,
)

INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ===========================================================================
# measure_qubit -- deterministic on basis states
# ===========================================================================


def test_measure_qubit_basis_state_each_qubit() -> None:
    # |101> = amp[5]: q0=1, q1=0, q2=1. Measure each in turn; on a
    # basis state the outcome is fully determined.
    for target, want in [(0, 1), (1, 0), (2, 1)]:
        q = Qreg(3, device="cpu", seed=42)
        q.init_basis(5)
        outcome = measure_qubit(q, target)
        assert outcome == want, (
            f"measure_qubit(target={target}) on |101> = {outcome}, want {want}"
        )


def test_measure_qubit_collapse_preserves_norm() -> None:
    q = Qreg(2, device="cpu", seed=7)
    q.init_basis(0)
    apply_h(q, 0)
    measure_qubit(q, 0)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


def test_measure_qubit_collapse_zeros_other_branch() -> None:
    # After measuring qubit 0 of |+>, the state must be exactly a
    # basis state of qubit 0: either amp[0]=1, amp[1]=0 or vice versa.
    q = Qreg(1, device="cpu", seed=99)
    q.init_basis(0)
    apply_h(q, 0)
    outcome = measure_qubit(q, 0)
    tol = amp_tol_for(q.dtype)
    if outcome == 0:
        assert q.amplitude(0) == pytest.approx(1 + 0j, abs=tol)
        assert q.amplitude(1) == pytest.approx(0 + 0j, abs=tol)
    else:
        assert q.amplitude(0) == pytest.approx(0 + 0j, abs=tol)
        assert q.amplitude(1) == pytest.approx(1 + 0j, abs=tol)


def test_measure_qubit_distribution_on_plus_state() -> None:
    # H|0> -> measure qubit 0 in a loop; expect roughly 50/50 outcomes.
    # We re-prepare the +-state each iteration (cheap, no RNG cost), so
    # each measurement consumes one RNG draw from the same q._gen.
    n_shots = 1000
    q = Qreg(1, device="cpu", seed=2026)
    counts = [0, 0]
    for _ in range(n_shots):
        q.init_basis(0)
        apply_h(q, 0)
        counts[measure_qubit(q, 0)] += 1
    # 3-sigma window for binomial(1000, 0.5) is ~ 500 +/- 47. Use
    # +/-100 for safety; the test must not flake on slow CI.
    assert 400 < counts[0] < 600, (
        f"H|0> 1000 shots: {counts[0]} zeros (expected ~500)"
    )
    assert 400 < counts[1] < 600


def test_measure_qubit_rejects_out_of_range() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: measure_qubit: target=5"
    ):
        measure_qubit(q, 5)


def test_measure_qubit_rejects_negative() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: measure_qubit: target=-1"
    ):
        measure_qubit(q, -1)


def test_measure_qubit_raises_qubit_prefix_on_zero_norm() -> None:
    # A fresh Qreg has an all-zero amplitude vector (init_basis hasn't
    # been called). measure_qubit must raise a "qubit:"-prefixed
    # RuntimeError rather than crashing with a bare math error or
    # producing an unbiased result.
    q = Qreg(2, device="cpu")
    with pytest.raises(
        RuntimeError, match=r"^qubit: measure_qubit: total probability"
    ):
        measure_qubit(q, 0)


def test_measure_qubit_handles_unnormalised_state() -> None:
    # Hand-scale H|0> by a factor of 2 so |amp|^2 sums to 4 rather than 1.
    # The sampling step must use p0/total (not the raw p0), and the
    # post-collapse state must still be unit-norm. Old code with the
    # un-normalised `u < p0` comparison would always pick outcome=0 here
    # because u in [0, 1) is always < p0=2.
    q = Qreg(1, device="cpu", seed=42)
    q.init_basis(0)
    apply_h(q, 0)
    q._amp.mul_(2.0)
    # Pre-condition: state is scaled to norm 4.
    assert q.norm() == pytest.approx(4.0, abs=prob_tol_for(q.dtype))
    outcome = measure_qubit(q, 0)
    assert outcome in (0, 1)
    # Post-condition: collapse to a basis state with unit norm.
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))
    tol = amp_tol_for(q.dtype)
    if outcome == 0:
        assert q.amplitude(0) == pytest.approx(1 + 0j, abs=tol)
        assert q.amplitude(1) == pytest.approx(0 + 0j, abs=tol)
    else:
        assert q.amplitude(0) == pytest.approx(0 + 0j, abs=tol)
        assert q.amplitude(1) == pytest.approx(1 + 0j, abs=tol)


# ===========================================================================
# measure_all
# ===========================================================================


def test_measure_all_basis_state_is_deterministic() -> None:
    q = Qreg(3, device="cpu", seed=11)
    q.init_basis(6)
    assert measure_all(q) == 6


def test_measure_all_collapses_to_chosen_basis() -> None:
    q = Qreg(2, device="cpu", seed=7)
    q.init_basis(0)
    apply_h(q, 0)
    apply_h(q, 1)
    chosen = measure_all(q)
    # After collapse, only amp[chosen] is non-zero.
    tol = amp_tol_for(q.dtype)
    for i in range(4):
        expected = 1 + 0j if i == chosen else 0 + 0j
        assert q.amplitude(i) == pytest.approx(expected, abs=tol)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


def test_measure_all_raises_qubit_prefix_on_zero_norm() -> None:
    # A fresh Qreg with no init_basis has all-zero amplitudes. Calling
    # measure_all must raise a "qubit:"-prefixed RuntimeError rather
    # than leaking torch.multinomial's bare RuntimeError.
    q = Qreg(2, device="cpu")
    with pytest.raises(
        RuntimeError, match=r"^qubit: measure_all: total probability"
    ):
        measure_all(q)


def test_measure_all_distribution_on_uniform_superposition() -> None:
    # Apply H on each qubit of a 3-qubit |0> -> uniform 1/sqrt(8) over
    # all 8 basis states. Use sample_distribution to count outcomes;
    # each bin should be ~125 out of 1000.
    q = Qreg(3, device="cpu", seed=2026)
    q.init_basis(0)
    for i in range(3):
        apply_h(q, i)
    outcomes = sample_distribution(q, 1000)
    counts = Counter(outcomes)
    # 3-sigma window for binomial(1000, 1/8) is 125 +/- ~33. Use a
    # generous +/-50 to avoid flakes.
    for k in range(8):
        assert 75 < counts[k] < 175, (
            f"uniform 1000 shots: amp[{k}] count = {counts[k]}"
        )


# ===========================================================================
# sample_distribution
# ===========================================================================


def test_sample_distribution_does_not_mutate_original() -> None:
    q = Qreg(2, device="cpu", seed=42)
    q.init_basis(0)
    apply_h(q, 0)  # |+> on qubit 0; uniform over {0, 1}
    snapshot_before = q.amplitudes_copy()
    sample_distribution(q, 50)
    snapshot_after = q.amplitudes_copy()
    # The full amplitude vector must be unchanged.
    assert torch.equal(snapshot_before, snapshot_after)


def test_sample_distribution_returns_list_of_ints() -> None:
    q = Qreg(2, device="cpu", seed=1)
    q.init_basis(3)
    out = sample_distribution(q, 5)
    assert isinstance(out, list)
    assert all(isinstance(x, int) for x in out)
    # |11> is deterministic; every shot must return 3.
    assert out == [3, 3, 3, 3, 3]


def test_sample_distribution_zero_shots_returns_empty() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(0)
    assert sample_distribution(q, 0) == []


def test_sample_distribution_rejects_negative_shots() -> None:
    q = Qreg(2, device="cpu")
    with pytest.raises(
        ValueError, match=r"^qubit: sample_distribution: shots=-1"
    ):
        sample_distribution(q, -1)


def test_sample_distribution_preserves_norm() -> None:
    # The function must restore the original (norm=1) state, including
    # after sampling has internally collapsed the register.
    q = Qreg(3, device="cpu", seed=2026)
    q.init_basis(0)
    for i in range(3):
        apply_h(q, i)
    sample_distribution(q, 100)
    assert q.norm() == pytest.approx(1.0, abs=prob_tol_for(q.dtype))


# ===========================================================================
# clone
# ===========================================================================


def test_clone_amp_is_independent() -> None:
    q1 = Qreg(2, device="cpu", seed=42)
    q1.init_basis(1)
    q2 = clone(q1)
    # Mutate the clone: apply X to flip the LSB of basis 1 -> basis 0.
    apply_x(q2, 0)
    # Original is unchanged.
    tol = amp_tol_for(q1.dtype)
    assert q1.amplitude(1) == pytest.approx(1 + 0j, abs=tol)
    assert q1.amplitude(0) == pytest.approx(0 + 0j, abs=tol)
    # Clone has the mutation.
    assert q2.amplitude(0) == pytest.approx(1 + 0j, abs=tol)
    assert q2.amplitude(1) == pytest.approx(0 + 0j, abs=tol)


def test_clone_amp_matches_initially() -> None:
    q1 = Qreg(3, device="cpu", seed=2026)
    q1.init_basis(0)
    for i in range(3):
        apply_h(q1, i)
    q2 = clone(q1)
    snap1 = q1.amplitudes_copy()
    snap2 = q2.amplitudes_copy()
    assert torch.equal(snap1, snap2)


def test_clone_preserves_metadata() -> None:
    q1 = Qreg(4, device="cpu", seed=1, dtype=torch.complex128)
    q2 = clone(q1)
    assert q2.n_qubits == q1.n_qubits
    assert q2.device == q1.device
    assert q2.dtype == q1.dtype


def test_clone_rng_state_starts_equal() -> None:
    # The clone copies q's RNG state. Until either advances, both
    # generators draw identical random sequences -- including the same
    # measurement outcome on identical superpositions.
    q1 = Qreg(1, device="cpu", seed=42)
    q1.init_basis(0)
    apply_h(q1, 0)
    q2 = clone(q1)
    # Both clones are |+> with the same RNG state. The first
    # measurement on each must give the same outcome.
    outcome1 = measure_qubit(q1, 0)
    outcome2 = measure_qubit(q2, 0)
    assert outcome1 == outcome2


def test_clone_rng_independence_after_first_draw() -> None:
    # After clone, one register advancing its RNG must not advance the
    # other's. Verify by drawing from each and checking they STILL
    # match (because both were drawing from independent generators
    # that happened to have identical states at the start).
    q1 = Qreg(1, device="cpu", seed=42)
    q2 = clone(q1)
    # Advance q1's generator three times.
    for _ in range(3):
        torch.rand((), generator=q1._gen)
    # q2's generator is still at the initial state. Draw a value from
    # each and check that q2's value is what q1's FIRST value was.
    # That requires resetting; cleaner test: verify q1._gen.get_state()
    # is different from q2._gen.get_state() after q1 advanced.
    state1 = q1._gen.get_state()
    state2 = q2._gen.get_state()
    assert not torch.equal(state1, state2), (
        "after advancing q1's RNG, the two generators' states must "
        "differ -- they're independent"
    )


# ===========================================================================
# dump
# ===========================================================================


def test_dump_on_basis_state() -> None:
    q = Qreg(3, device="cpu")
    q.init_basis(5)
    result = dump(q)
    assert result == [(5, 1 + 0j)]


def test_dump_on_superposition_returns_all_branches() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(0)
    apply_h(q, 0)
    result = dump(q)
    # Should have two entries: amp[0] and amp[1].
    assert len(result) == 2
    indices = sorted(idx for idx, _ in result)
    assert indices == [0, 1]
    for _, amp in result:
        assert abs(amp - (INV_SQRT2 + 0j)) <= amp_tol_for(q.dtype)


def test_dump_threshold_filters() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(0)
    apply_h(q, 0)
    # H|0> amplitudes are 1/sqrt(2) ~= 0.707. Threshold 0.8 filters all.
    assert dump(q, threshold=0.8) == []
    # Threshold 0.5 keeps both.
    assert len(dump(q, threshold=0.5)) == 2


def test_dump_on_zero_state_returns_empty() -> None:
    # A fresh Qreg with no init_basis is all zeros.
    q = Qreg(3, device="cpu")
    assert dump(q) == []


def test_dump_returns_tuples_of_int_and_complex() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(2)
    result = dump(q)
    assert len(result) == 1
    basis, amp = result[0]
    assert isinstance(basis, int)
    assert isinstance(amp, complex)
    assert basis == 2


def test_dump_rejects_negative_threshold() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(0)
    with pytest.raises(ValueError, match=r"^qubit: dump: threshold=-0.5"):
        dump(q, threshold=-0.5)


# ===========================================================================
# Method-style API: q.X() must match the function-style X(q)
# ===========================================================================


def test_method_measure_qubit_matches_function() -> None:
    # Both call paths share the same RNG state initially, so the
    # outcome must match.
    q1 = Qreg(2, device="cpu", seed=42)
    q1.init_basis(0)
    apply_h(q1, 0)
    q2 = clone(q1)
    out_function = measure_qubit(q1, 0)
    out_method = q2.measure_qubit(0)
    assert out_function == out_method


def test_method_measure_all_matches_function() -> None:
    q1 = Qreg(3, device="cpu", seed=2026)
    q1.init_basis(0)
    for i in range(3):
        apply_h(q1, i)
    q2 = clone(q1)
    out_function = measure_all(q1)
    out_method = q2.measure_all()
    assert out_function == out_method


def test_method_sample_distribution_matches_function() -> None:
    q1 = Qreg(2, device="cpu", seed=11)
    q1.init_basis(0)
    apply_h(q1, 0)
    q2 = clone(q1)
    out_function = sample_distribution(q1, 50)
    out_method = q2.sample_distribution(50)
    assert out_function == out_method


def test_method_dump_matches_function() -> None:
    q = Qreg(2, device="cpu")
    q.init_basis(0)
    apply_h(q, 0)
    assert dump(q) == q.dump()


def test_method_clone_matches_function() -> None:
    q1 = Qreg(2, device="cpu", seed=42)
    q1.init_basis(1)
    c_function = clone(q1)
    c_method = q1.clone()
    # Both clones have the same amp and RNG state; their first draw
    # must match.
    assert torch.equal(c_function.amplitudes_copy(), c_method.amplitudes_copy())


# ===========================================================================
# Device-parametrised: basis measurement + dump on every available device
# ===========================================================================


def test_measure_qubit_basis_across_devices(device: str) -> None:
    q = Qreg(3, device=device, seed=42)
    q.init_basis(5)  # |101>
    # Deterministic measurement on every device.
    assert measure_qubit(q, 0) == 1


def test_measure_all_basis_across_devices(device: str) -> None:
    q = Qreg(3, device=device, seed=42)
    q.init_basis(6)
    assert measure_all(q) == 6


def test_dump_across_devices(device: str) -> None:
    q = Qreg(3, device=device)
    q.init_basis(5)
    result = dump(q)
    assert result == [(5, 1 + 0j)]
