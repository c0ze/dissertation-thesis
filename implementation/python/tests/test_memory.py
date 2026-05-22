"""Tests for the memory-preflight helpers."""

from __future__ import annotations

import pytest
import torch

from qubit._memory import (
    dtype_bytes,
    estimate_peak_bytes,
    estimate_state_bytes,
    free_bytes,
    preflight,
)


def test_dtype_bytes_complex64() -> None:
    assert dtype_bytes(torch.complex64) == 8


def test_dtype_bytes_complex128() -> None:
    assert dtype_bytes(torch.complex128) == 16


def test_dtype_bytes_unsupported() -> None:
    with pytest.raises(ValueError, match=r"^qubit: dtype_bytes: "):
        dtype_bytes(torch.float64)


def test_estimate_state_bytes_minimum() -> None:
    # 1 qubit -> 2 amplitudes -> 32 bytes at complex128, 16 at complex64.
    assert estimate_state_bytes(1, torch.complex128) == 32
    assert estimate_state_bytes(1, torch.complex64) == 16


def test_estimate_state_bytes_at_25_qubits() -> None:
    # The /go thesis target; 512 MiB at complex128.
    assert estimate_state_bytes(25, torch.complex128) == (1 << 25) * 16
    assert estimate_state_bytes(25, torch.complex64) == (1 << 25) * 8


def test_estimate_state_bytes_rejects_zero() -> None:
    with pytest.raises(ValueError, match=r"^qubit: estimate_state_bytes: "):
        estimate_state_bytes(0, torch.complex128)


def test_estimate_state_bytes_rejects_negative() -> None:
    with pytest.raises(ValueError):
        estimate_state_bytes(-3, torch.complex128)


def test_peak_state_equals_state_bytes() -> None:
    # The "state" op is the baseline; no extra allocation.
    n = 10
    dt = torch.complex128
    assert estimate_peak_bytes(n, dt, op="state") == estimate_state_bytes(n, dt)


def test_peak_qft_is_four_state_bytes() -> None:
    # QFT is NOT actually in-place: it decomposes into Hadamards and
    # controlled-phase gates, each of which allocates a fresh state
    # vector via PyTorch tensor ops. We treat 4*state as a defensive
    # upper bound on a single QFT gate's transient peak.
    n = 10
    dt = torch.complex128
    assert estimate_peak_bytes(n, dt, op="qft") == 4 * estimate_state_bytes(n, dt)


def test_peak_single_gate_is_two_state_bytes() -> None:
    # apply_u runs tensordot + movedim, which holds the source state
    # vector and the freshly-allocated output simultaneously.
    n = 10
    dt = torch.complex128
    assert estimate_peak_bytes(n, dt, op="single_gate") == 2 * estimate_state_bytes(n, dt)


def test_peak_controlled_gate_is_four_state_bytes() -> None:
    # apply_cu does permute + reshape + matmul + reshape + inverse-permute;
    # the transient peak can hold up to four state-sized tensors.
    n = 10
    dt = torch.complex128
    assert estimate_peak_bytes(n, dt, op="controlled_gate") == 4 * estimate_state_bytes(n, dt)


def test_peak_modexp_counts_extra_tensors() -> None:
    # ModularExp peak = 2*state (gather output + original) + 2*int64
    # permutation (CPU side + device side transiently coexist).
    n = 10
    dt = torch.complex128
    state = estimate_state_bytes(n, dt)
    perm = (1 << n) * 8
    assert estimate_peak_bytes(n, dt, op="modexp") == 2 * state + 2 * perm


def test_peak_unknown_op_raises() -> None:
    # Deliberate: passing a string outside the _Op Literal union to exercise
    # the runtime branch. The `type: ignore` on the call itself suppresses
    # mypy's complaint about the bad Literal argument.
    with pytest.raises(ValueError, match=r"^qubit: estimate_peak_bytes: unknown op"):
        estimate_peak_bytes(4, torch.complex128, op="bogus")  # type: ignore[arg-type]


def test_free_bytes_cpu_returns_none() -> None:
    # CPU has no portable free-memory query in stable PyTorch, so the
    # function reports None to let preflight skip the precise check.
    assert free_bytes(torch.device("cpu")) is None


def test_free_bytes_mps_returns_none() -> None:
    assert free_bytes(torch.device("mps")) is None


def test_preflight_accepts_small_request() -> None:
    # Small requests under the sanity ceiling pass without raising on
    # any device (free_bytes is None on CPU/MPS so the precise check
    # is skipped; the CUDA branch has plenty of headroom for 8 qubits).
    preflight(8, torch.complex128, torch.device("cpu"), op="state")


def test_preflight_rejects_obviously_impossible_request() -> None:
    # 50 qubits at complex128 = 18 PiB. Far above the 1-TiB sanity
    # ceiling regardless of device. Must raise MemoryError before any
    # allocation is attempted.
    with pytest.raises(MemoryError, match=r"^qubit: preflight: requested n_qubits=50"):
        preflight(50, torch.complex128, torch.device("cpu"), op="state")


def test_preflight_rejects_modexp_oversize() -> None:
    # ModularExp at 40 qubits would need ~32 TiB peak; reject.
    with pytest.raises(MemoryError):
        preflight(40, torch.complex128, torch.device("cpu"), op="modexp")
