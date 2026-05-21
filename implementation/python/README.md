# implementation/python -- PyTorch state-vector quantum simulator

A pure-Python state-vector quantum-circuit simulator using PyTorch for
tensor ops and device-agnostic GPU acceleration. Sibling of
`implementation/c` (MPI) and `implementation/go` (goroutines), covering
the same thesis claims via a third parallelism model: tensor ops on
whatever GPU PyTorch can reach (NVIDIA CUDA, AMD ROCm, Apple Metal/MPS)
with a CPU fallback.

## Status

**Phase 0+1 complete.** This ships:

- The package scaffold (`pyproject.toml`, `uv`-managed env, `ruff` +
  `mypy` + `pytest` configured)
- The `Qreg` class with construction, accessors, and the qubit-axis
  helper
- Memory-preflight helpers
- Tests for everything above

**Not yet implemented:** gates (`apply_*`), measurement
(`measure_qubit`, `measure_all`, `sample_distribution`), QFT, Grover,
Shor, the CLI demo.

## Quickstart

```bash
cd implementation/python
make sync           # creates .venv, installs torch + pytest + ruff + mypy
make test           # runs the test suite
make check          # lint + typecheck + test
```

## Key design choices

- **Device-agnostic via `torch.device`.** `Qreg(n_qubits)` auto-detects
  `cuda` > `mps` > `cpu`; override with `Qreg(n_qubits, device='cpu')`.
- **Tensor-native gates.** State is a flat `(2**n,)` complex tensor;
  gate code will reshape into the `(2,) * n` view and `einsum`/`tensordot`
  the unitary against the target axis. No per-amplitude loops.
- **Dtype policy.** `complex128` on CPU/CUDA/ROCm, `complex64` on MPS
  (the MPS backend lacks `float64`). Explicit
  `Qreg(..., device='mps', dtype=torch.complex128)` raises `ValueError`
  with a guidance message; the simulator never silently downgrades.
- **Qubit 0 is LSB.** Same as `/c` and `/go`. The tensor axis for qubit
  `q` is `n_qubits - 1 - q`, centralised in `qubit/_axis.py`.
- **CPU RNG.** Even when `_amp` lives on GPU, the measurement RNG is a
  CPU `torch.Generator` — MPS-generator quirks are real and measurement
  is a readout boundary anyway. Seeded tests are bit-identical across
  CPU / CUDA / MPS.
- **No `destroy()` / `close()` / context manager.** Python's GC reclaims
  PyTorch tensors when the `Qreg` goes out of scope. No `with` block,
  no manual cleanup.

## API at a glance (Phase 0+1)

```python
import torch
from qubit import Qreg, qubit_axis

# Auto-detect device + dtype.
q = Qreg(n_qubits=4)
print(q.device, q.dtype, q.n_qubits)

# Or explicit:
q = Qreg(4, device='cpu', seed=42, dtype=torch.complex128)

# Seeded measurement-RNG is reproducible across same-seed Qregs.
q.init_basis(5)  # |0101>
print(q.amplitude(5))    # (1+0j)
print(q.prob_of(5))      # 1.0
print(q.norm())          # 1.0

# Defensive CPU clone for inspection / cross-implementation comparison:
amps = q.amplitudes_copy()   # 1-D CPU tensor, len 16
assert amps[5] == 1+0j

# Qubit-to-axis helper (used by future gate code):
assert qubit_axis(0, 4) == 3   # qubit 0 is the LSB / rightmost axis
assert qubit_axis(3, 4) == 0   # qubit 3 is the MSB / leftmost axis
```

## Layout

```
qubit/
  __init__.py            # public API re-exports
  _axis.py               # qubit_axis (LSB-first convention lives here)
  _device.py             # default_device, default_dtype, validate_dtype_device
  _memory.py             # estimate_state_bytes, estimate_peak_bytes, preflight
  _assert.py             # qubit:-prefixed ValueError/TypeError helpers
  qreg.py                # Qreg class
tests/
  conftest.py            # device fixture (parametrises over available devices)
  test_assert.py
  test_axis.py
  test_device.py
  test_memory.py
  test_qreg.py
  test_import.py         # phase-0 smoke test: package imports
pyproject.toml
Makefile
README.md
```

The underscore-prefixed modules are package-private; external callers
should import only from `qubit` (the top-level `__init__.py`
re-exports the public surface).

## Why does `standart.go` / `standart.c` exist but `standart.py` doesn't yet?

The misspelled filename is parity with `/c`'s 2004 spelling and will
appear in Phase 2 (arithmetic helpers: GCD, mod_pow, mul_mod,
continued_fraction). Phase 0+1 has no arithmetic to factor out, so
the file isn't created yet.

## Design state

Brainstormed sections §1-§5 with the user (architecture, constraints,
file layout, data model, gate model). Sections §6-§12 (module-by-module
API surface, testing details, build/CI integration, scope statements,
approval log) have not yet been written into a formal spec doc. The
next session should either finish brainstorming those sections before
Phase 2, or proceed with the architectural agreements already in place.

## Sibling implementations

- [/c](../c) -- MPI / OpenMPI 5.x. 26-qubit registers tested at NP=1..8.
- [/go](../go) -- goroutines / per-call WaitGroup. 70 tests, race-clean.
