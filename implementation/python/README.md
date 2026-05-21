# implementation/python -- PyTorch state-vector quantum simulator

A pure-Python state-vector quantum-circuit simulator using PyTorch for
tensor ops and device-agnostic GPU acceleration. Sibling of
`implementation/c` (MPI) and `implementation/go` (goroutines), covering
the same thesis claims via a third parallelism model: tensor ops on
whatever device PyTorch can reach (NVIDIA CUDA, AMD ROCm, Apple Metal /
MPS), with a CPU fallback.

## Status

**Feature-complete.** The implementation covers the same thesis claims
as `/c` and `/go`:

- `qubit.Qreg` — state-vector register with construction, accessors,
  the qubit-axis helper, memory preflight, and a CPU measurement RNG.
- `qubit.standart` — `gcd_u64`, `mod_pow`, `continued_fraction`,
  `is_power_of_two`, `ilog2_u32`. Python builtins (`math.gcd`, `pow`,
  `fractions.Fraction.limit_denominator`) wrapped with
  `qubit:`-prefixed validation.
- Single-qubit gates — `apply_u` (the `tensordot` + `movedim`
  workhorse) plus the named gates `apply_h`, `apply_x`, `apply_y`,
  `apply_z`, `apply_s`, `apply_t`, `apply_phase`, `apply_rx`,
  `apply_ry`, `apply_rz`.
- Controlled gates — `apply_cu` (4x4 block-diagonal workhorse via
  permute + matmul + inverse-permute) plus `apply_cnot`, `apply_cz`,
  `apply_controlled_phase`, `apply_swap` (the 3-CNOT identity).
- Multi-controlled gates — `apply_multi_controlled_z` for phase-flip
  diffusion and `apply_multi_controlled_x` (generalised Toffoli),
  both vectorized mask/gather/scatter; no per-amplitude loops.
- Measurement — `measure_qubit` (single-qubit projective with collapse
  and renormalise), `measure_all` (sample full basis from `|amp|^2`),
  `sample_distribution` (snapshot + restore so the original isn't
  mutated), `clone` (independent amp + RNG state), `dump` (structured
  list of non-zero amplitudes).
- QFT — `apply_qft` (forward, includes final bit-reversal swaps so
  output is in natural binary order) and `apply_qft_inverse` (true
  inverse via reversed loop order and negated phase angles, not three
  forward applications). Sub-register support via `start=` / `n=`.
- Grover — `apply_grover(q, n_qubits, oracle, user=None,
  iterations=None)`. The oracle is a `Callable[[Qreg, Any], None]`
  that phase-flips marked basis states; the default iteration count
  is the optimum for one marked item (`floor(pi/4 * sqrt(N))`).
- Shor — `apply_modular_exp` (vectorized permutation via
  `torch.gather` with a CPU-built index tensor), `apply_shor_period`
  (period-finding circuit), `shor_factor(N, max_attempts=20,
  seed=None)` (end-to-end factoring, bit-for-bit reproducible when
  seeded). v1 covers Shor-15 (12-qubit register) and Shor-21
  (16-qubit register, gated by `RUN_SHOR_21=1` to keep the default
  test loop fast).
- `qubit-demo` CLI — `--algo {bell,qft,grover,shor}` for quick
  algorithm-level smoke tests. Installed via `[project.scripts]`.

Function-style (`apply_h(q, 0)`) and method-style (`q.apply_h(0)`)
call shapes are both first-class for every gate, measurement op, QFT,
Grover, and Shor primitive.

**Tests:** 410 default + 1 gated Shor-21 = **411 total**. All green
under `ruff check`, `mypy --strict`, and on every available device
(CPU + MPS locally; CPU-only on Linux CI).

## Quickstart

```bash
cd implementation/python
make sync                      # creates .venv, installs torch + dev tools
make test                      # 410 tests, ~0.5s on Apple Silicon
make check                     # lint + typecheck + test (the full PR gate)

make demo                      # default: bell
make demo ALGO=qft             # QFT|0> = uniform
make demo ALGO=grover          # Grover marks |1111>
make demo ALGO=shor            # factor 15

RUN_SHOR_21=1 uv run pytest tests/test_shor.py -q   # 16-qubit gated test
```

The CLI also runs directly: `uv run qubit-demo --algo shor`.

## Key design choices

- **Device-agnostic via `torch.device`.** `Qreg(n_qubits)` auto-detects
  `cuda` > `mps` > `cpu`. Override with `Qreg(n_qubits, device='cpu')`.
- **Tensor-native gates.** State is a flat `(2**n,)` complex tensor;
  gate code reshapes into the `(2,) * n` view and `einsum`/`tensordot`
  the unitary against the target axis. No per-amplitude loops.
- **Dtype policy.** `complex128` on CPU / CUDA / ROCm; `complex64` on
  MPS (the MPS backend lacks `float64`, so `complex128` is impossible
  there). Explicit `Qreg(..., device='mps', dtype=torch.complex128)`
  raises `ValueError` with a guidance message; the simulator never
  silently downgrades.
- **Qubit 0 is LSB.** Matches `/c` and `/go`. The tensor axis for
  qubit `q` is `n_qubits - 1 - q`, centralised in `qubit/_axis.py`.
- **CPU measurement RNG.** Even when `_amp` lives on GPU, the
  measurement RNG is a CPU `torch.Generator` — MPS-generator quirks
  are real and measurement is a readout boundary anyway. Seeded tests
  are bit-identical across CPU / CUDA / MPS.
- **No `destroy()` / `close()` / context manager.** Python's GC
  reclaims PyTorch tensors when the `Qreg` goes out of scope. No
  `with` block, no manual cleanup.

## API at a glance

```python
import math, torch
from qubit import Qreg, apply_h, apply_x, qubit_axis, shor_factor

# Auto-detect device + dtype.
q = Qreg(n_qubits=4)
print(q.device, q.dtype, q.n_qubits)

# Or explicit:
q = Qreg(4, device='cpu', seed=42, dtype=torch.complex128)

# Seeded measurement-RNG is reproducible across same-seed Qregs.
q.init_basis(5)  # |0101>
print(q.prob_of(5))      # 1.0
print(q.norm())          # 1.0

# Gates: both function-style and method-style work.
q.apply_h(0)             # method-style
apply_x(q, 1)            # function-style
print(q.norm())          # still 1.0 (unitary)

# Custom unitary via apply_u (the workhorse every named gate dispatches to).
inv2 = 1.0 / math.sqrt(2.0)
h = torch.tensor(
    [[inv2 + 0j, inv2 + 0j], [inv2 + 0j, -inv2 + 0j]],
    dtype=q.dtype, device=q.device,
)
q.apply_u(2, h)

# Defensive CPU clone for inspection / cross-implementation comparison:
amps = q.amplitudes_copy()   # 1-D CPU tensor, len 16

# Qubit-to-axis helper (used internally by every gate):
assert qubit_axis(0, 4) == 3   # qubit 0 is the LSB / rightmost axis
assert qubit_axis(3, 4) == 0   # qubit 3 is the MSB / leftmost axis

# End-to-end Shor.
result = shor_factor(15, seed=42)
print(result)  # ShorFactorResult(p=3, q=5, attempts=1)
```

## Scope and known limitations

- **`shor_factor` does not detect prime powers** (`N = p**k` for prime
  `p`, `k >= 2`). Such inputs make every quantum attempt fail the
  factor-derivation check; the function exhausts `max_attempts` and
  reports failure. The classical pre-check (`round(N**(1/k))**k ==
  N`) belongs in the caller. Even `N` is handled here (short-
  circuits without a quantum step). Matches `/c` and `/go` scope.
- **Shor-25 and beyond are not a v1 performance target.** At 25
  qubits the modexp permutation tensor is 256 MiB (int64) on CPU
  before transfer, and the gather output adds another 512 MiB. The
  arithmetic is correct, but the memory cost is significant.
  Shor-15 (12-qubit register) and Shor-21 (16-qubit register) are
  the supported algorithm targets; both run in well under a second
  on CPU.
- **MPS dtype is `complex64`, not `complex128`.** The MPS backend
  lacks `float64`. `apply_modular_exp`'s permutation tensor and the
  resulting gather output are unaffected (both int64 / complex64),
  but precision-sensitive computations (e.g., very deep gate
  sequences) accumulate rounding faster than on CPU/CUDA.
  `amp_tol_for` and `prob_tol_for` return looser tolerances when
  the register's dtype is `complex64`.
- **No CUDA testing in CI.** GitHub-hosted runners don't have GPUs.
  The device-parametrised tests run CPU-only on CI; MPS coverage
  comes from local runs on Apple Silicon.

## Layout

```
qubit/
  __init__.py            # public API re-exports
  _axis.py               # qubit_axis (LSB-first convention lives here)
  _device.py             # default_device, default_dtype, validate_dtype_device
  _memory.py             # estimate_state_bytes, estimate_peak_bytes, preflight
  _assert.py             # qubit:-prefixed ValueError/TypeError helpers
  _view.py               # state_view + validate_matrix (gate helpers)
  qreg.py                # Qreg class + method wrappers for every gate
  standart.py            # arithmetic helpers (gcd, mod_pow, continued_fraction, ...)
  gates_single.py        # apply_u + apply_h/x/y/z/s/t/phase/rx/ry/rz
  gates_controlled.py    # apply_cu + cnot/cz/controlled_phase/swap
  gates_multi.py         # apply_multi_controlled_z + apply_multi_controlled_x
  measure.py             # measure_qubit / measure_all / sample_distribution / clone / dump
  qft.py                 # apply_qft + apply_qft_inverse (with bit-reversal swaps)
  grover.py              # apply_grover (uniform prep + oracle/diffusion iterations)
  shor.py                # apply_modular_exp + apply_shor_period + shor_factor
  cli.py                 # qubit-demo CLI (argparse, exit codes, top-level catch)
tests/
  conftest.py            # device fixture (parametrises over available devices)
  test_assert.py
  test_axis.py
  test_device.py
  test_memory.py
  test_qreg.py
  test_standart.py
  test_gates_single.py
  test_gates_controlled.py
  test_gates_multi.py
  test_measure.py
  test_qft.py
  test_grover.py
  test_shor.py
  test_cli.py
  test_import.py
pyproject.toml
Makefile
README.md
assessment.md           # thesis-claim coverage map
```

The underscore-prefixed modules are package-private; external callers
should import only from `qubit` (the top-level `__init__.py`
re-exports the public surface).

The misspelled `standart.py` filename is deliberate parity with `/c`'s
2004 spelling and the `/go` sibling. The function names inside use
modern snake_case (`gcd_u64`, `mod_pow`, `continued_fraction`,
`is_power_of_two`, `ilog2_u32`); the `_u64` / `_u32` suffixes are
preserved as visual cues even though Python ints have no width limit.

## Cross-implementation parity

| Claim | This implementation | Sibling locations |
|---|---|---|
| LSB-first basis indexing | `qubit/_axis.py::qubit_axis` | `/go gates_single.go`, `/c parallel.c` |
| Tensor-native gates | `qubit/_view.py::state_view`, `gates_single.apply_u` | `/go parallelOverPairs`, `/c apply_u` |
| ModularExp via permutation | `qubit/shor.py::_build_modexp_perm` | `/go shor.go::_build_modexp_permutation`, `/c shor.c::apply_modular_exp` |
| QFT with bit-reversal swaps | `qubit/qft.py::apply_qft` | `/go qft.go::ApplyQFT`, `/c qft.c::apply_qft` |
| Continued-fraction recovery | `qubit/standart.py::continued_fraction` | `/go standart.go::ContinuedFraction`, `/c standart.c::continued_fraction` |
| Factor N=15 reliably | `tests/test_shor.py::test_shor_factor_15_*` | `/go shor_test.go`, `/c tests/test_shor.c` |
| Shor-21 gated by env var | `tests/test_shor.py::test_shor_period_a2_mod21_gated` | `/go shor_test.go::TestShorPeriodA2Mod21`, `/c tests/test_shor.c` |

See [`assessment.md`](./assessment.md) for the detailed file:line map.

## Sibling implementations

- [/c](../c) — MPI / OpenMPI 5.x. 26-qubit registers tested at NP=1..8.
- [/go](../go) — goroutines / per-call WaitGroup. 70 tests, race-clean.
