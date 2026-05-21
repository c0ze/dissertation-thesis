"""Tests for the qubit-to-tensor-axis convention.

The whole package's correctness depends on the LSB-first mapping
``axis = n - 1 - target``. These tests lock the convention so any future
refactor that flips qubit ordering fails loudly here before the gate
tests start producing mysteriously bit-reversed outputs.
"""

from __future__ import annotations

import pytest

from qubit import qubit_axis


def test_qubit_0_is_last_axis() -> None:
    # Qubit 0 is LSB -> rightmost tensor axis when state is reshaped to (2,)*n.
    assert qubit_axis(0, 1) == 0
    assert qubit_axis(0, 2) == 1
    assert qubit_axis(0, 4) == 3
    assert qubit_axis(0, 10) == 9


def test_qubit_n_minus_1_is_first_axis() -> None:
    # The MSB lives at axis 0 (slowest-varying axis).
    assert qubit_axis(0, 1) == 0  # n=1: only one axis, sits at index 0
    assert qubit_axis(1, 2) == 0
    assert qubit_axis(3, 4) == 0
    assert qubit_axis(9, 10) == 0


def test_full_axis_sweep_is_a_permutation() -> None:
    # For every n, the set of qubit_axis values must be a permutation
    # of [0, n) -- no qubit collides with another, no axis goes unused.
    for n in (1, 2, 3, 4, 8, 16):
        axes = [qubit_axis(q, n) for q in range(n)]
        assert sorted(axes) == list(range(n))


def test_axis_mapping_is_self_inverse() -> None:
    # qubit_axis is its own inverse: axis(axis(q, n), n) == q.
    for n in (1, 4, 8):
        for q in range(n):
            assert qubit_axis(qubit_axis(q, n), n) == q


@pytest.mark.parametrize(
    "target,n",
    [(-1, 1), (1, 1), (4, 4), (100, 4), (-5, 3)],
)
def test_out_of_range_raises(target: int, n: int) -> None:
    with pytest.raises(ValueError, match=r"^qubit: qubit_axis: "):
        qubit_axis(target, n)
