# implementation/python — coverage of thesis claims

Updated at the end of Phase 9. File:line references point to the
canonical implementation site for each claim. Sibling implementations:
[`/c`](../c/assessment.md) (MPI) and [`/go`](../go/assessment.md)
(goroutines) cover the same claims via different parallelism models.

## §8 (sparse-gate strategy, 2026 thesis)

| Claim | Status | Location |
|---|---|---|
| In-place single-qubit gate, O(2^n) | ✓ | `qubit/gates_single.py::apply_u` |
| State vector as a flat `(2^n,)` PyTorch tensor | ✓ | `qubit/qreg.py::Qreg._amp` |
| Tensor-native dispatch via `tensordot` + `movedim` | ✓ | `qubit/gates_single.py::apply_u` |
| Controlled gates via 4x4 block-diagonal CU | ✓ | `qubit/gates_controlled.py::apply_cu` |
| ModularExp as in-place permutation via `torch.gather` | ✓ | `qubit/shor.py::apply_modular_exp` |
| qreg API per §12 | ✓ | `qubit/qreg.py::Qreg` |

**Parallelism model.** Instead of MPI ranks (`/c`) or goroutines
(`/go`), this implementation routes work through PyTorch's tensor ops
on whatever device is available (NVIDIA CUDA, AMD ROCm, Apple Metal /
MPS, or CPU fallback). The "parallel dispatch" is PyTorch's kernel
scheduler; the simulator never has a `parallelOverPairs`-style
abstraction of its own.

## §9 (2026 QFT)

| Claim | Status | Location |
|---|---|---|
| QFT forward + inverse | ✓ | `qubit/qft.py::apply_qft` / `apply_qft_inverse` |
| Includes final bit-reversal swaps (natural binary order) | ✓ | `qubit/qft.py::apply_qft` final swap loop |
| Inverse is a TRUE inverse, not three forward applications | ✓ | `qubit/qft.py::apply_qft_inverse` (reverses loop order, negates phases) |
| Period detection on known periodic input | ✓ (tested) | `tests/test_qft.py::test_qft_on_basis_state_matches_analytic` |
| Round-trip exhaustively across basis states for n=1..4 | ✓ (tested) | `tests/test_qft.py::test_qft_then_qft_inverse_round_trip` |

## §10 (2026 Grover)

| Claim | Status | Location |
|---|---|---|
| Phase-oracle callback API | ✓ | `qubit/grover.py::apply_grover` (oracle: `Callable[[Qreg, Any], None]`) |
| H^n → oracle / diffusion loop | ✓ | `qubit/grover.py::apply_grover` |
| Diffusion via H X MCZ X H sandwich | ✓ | `qubit/grover.py::apply_grover` |
| Default iteration count = floor(π/4·√N) | ✓ | `qubit/grover.py::apply_grover` |
| Over-iteration drops probability (rotation reverses past π/2) | ✓ (tested) | `tests/test_grover.py::test_grover_over_iteration_drops_probability` |
| Multiple marked items (4 of 16 in 1 iter → P=1.0) | ✓ (tested) | `tests/test_grover.py::test_grover_four_marked_in_16_one_iteration` |
| n_qubits=1 degenerate case supported | ✓ (tested) | `tests/test_grover.py::test_grover_n_qubits_one_does_not_amplify_but_runs` |

## §11 (2026 Shor)

| Claim | Status | Location |
|---|---|---|
| `apply_modular_exp` with y >= N pass-through | ✓ | `qubit/shor.py::apply_modular_exp` |
| `apply_shor_period` (period-finding subroutine) | ✓ | `qubit/shor.py::apply_shor_period` |
| `shor_factor` (end-to-end with retry loop) | ✓ | `qubit/shor.py::shor_factor` |
| Continued-fraction post-processing | ✓ | `qubit/standart.py::continued_fraction` |
| Factor N=15 reliably with seed | ✓ (tested) | `tests/test_shor.py::test_shor_factor_15_seeded` |
| Even N short-circuit (no quantum step) | ✓ (tested) | `tests/test_shor.py::test_shor_factor_even_N_short_circuits` |
| Shor-21 period of 2 mod 21 (gated) | ✓ (gated) | `tests/test_shor.py::test_shor_period_a2_mod21_gated` |
| Reproducibility under seed | ✓ (tested) | `tests/test_shor.py::test_shor_factor_seeded_is_deterministic` |

**Scope caveat.** `shor_factor` does NOT detect prime powers
(`N = p**k` for prime `p`, `k >= 2`). Such inputs make every quantum
attempt fail the factor-derivation check; the function exhausts
`max_attempts` and reports failure. The classical pre-check belongs
in the caller. Matches `/c` and `/go` scope. Documented inline at
`qubit/shor.py::shor_factor` docstring.

## §12 (qreg API)

Every entry in spec §6 is implemented in `qubit/qreg.py` + the
adjacent gate / measurement / algorithm files. Python-specific
deviations from the abstract API:

- **No `Destroy()` / `close()` / context manager.** Python's GC
  reclaims PyTorch tensors when the `Qreg` goes out of scope.
- **Method wrappers for every state-mutating function.** `q.apply_h(0)`
  reads more naturally for method-chaining; `apply_h(q, 0)` matches
  the lower-level function signature. Both are public.
- **Amplitude slice is unexported (`_amp`).** Accessors are
  `amplitude(i)`, `amplitudes_copy()`, `prob_of(basis)`, `norm()`.
- **Functional options at construction.** `Qreg(n, *, device=None,
  seed=None, dtype=None, check_memory=True)`. `None` triggers
  auto-detection per the device/dtype policy.
- **Programmer-error panics use `ValueError` / `TypeError` with the
  `qubit:` prefix.** Same uniform-exception policy across the package;
  no `panic`-vs-`error` split as in `/go`. Library code never calls
  `sys.exit`; the CLI (`qubit/cli.py`) catches at the top level and
  exits non-zero.

## Out of scope for v1

- Distributed execution across machines (would re-introduce MPI's
  problem space; outside the GPU-tensor-ops premise).
- Density matrices / mixed states.
- Noise models.
- Prime-power detection in `shor_factor` (classical pre-check; matches
  `/c` and `/go`).
- Shor-25 and beyond as a routine target (the modexp permutation
  tensor at 25 qubits is 256 MiB on CPU before transfer; the gather
  output is another 512 MiB. The arithmetic is correct, but the
  memory cost is significant. Shor-15 and Shor-21 are the supported
  algorithm targets.)

## Test matrix

`uv run pytest -q` covers **414 active tests** in well under a second on
Apple Silicon (`make check` adds `ruff` and `mypy --strict`). The
gated Shor-21 test brings the total to **415** via `RUN_SHOR_21=1`
and runs the 16-qubit
period-finding circuit with fixed `a=2`; the only stochasticity is
the QFT-readout measurement, so the test asserts the recovered
period divides 6 (the true order of 2 mod 21).

Device coverage on this Apple Silicon host: CPU + MPS via the
parametrised `device` fixture in `tests/conftest.py`. CI runs Linux
CPU-only.

## CLI

The `qubit-demo` console script is exposed via
`[project.scripts] qubit-demo = "qubit.cli:main"` in `pyproject.toml`.
It mirrors the demos in `/c`'s `cmd/qubit` and `/go`'s
`cmd/qubit/main.go`:

```bash
uv run qubit-demo --algo bell
uv run qubit-demo --algo qft
uv run qubit-demo --algo grover
uv run qubit-demo --algo shor
```

All four demos are seeded where applicable so successive runs produce
identical output. Demo errors print to stderr with a `qubit-demo:`
prefix and the process exits non-zero.
