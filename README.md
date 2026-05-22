# Quantum Computation: Implementation of Quantum Gates on Linux Clusters

Dissertation source and accompanying quantum-circuit simulators.

[![CI](https://github.com/c0ze/dissertation-thesis/actions/workflows/ci.yml/badge.svg)](https://github.com/c0ze/dissertation-thesis/actions/workflows/ci.yml)

Bachelor's thesis, Istanbul Bilgi University. Originally submitted 2004
under the supervision of Bülent Özel; revised 2026 to modernise the
build, rewrite the algorithm chapters (QFT, Grover, Shor), and add
three new reference implementations of the library API described in
the thesis.

* `qucomp.pdf` — the revised dissertation (2026).
* `qucomp-original.pdf` — the original submission (2004), preserved
  verbatim.
* `source code/` — LaTeX sources for the dissertation.
* `implementation/` — four sibling implementations of the simulator
  API, each parallelised through a different model. See the comparison
  table below.

The thesis is a textbook-style tour: §1–§7 cover the physics, the
mathematical preliminaries, models of computation, and the quantum
algorithms (Deutsch–Jozsa, QFT, Grover, Shor). §8 motivates a
distribution strategy for state-vector simulators. §9–§11 develop the
three landmark algorithms in detail. §12 specifies a small library API
(`qreg`, gates, measurement, QFT, Grover, Shor). The
`implementation/` directory turns §8 and §12 into runnable code.

## Implementations

| Directory | Language | Parallelism | Status | Tests | Max qubits |
|---|---|---|---|---|---|
| [`implementation/original/`](implementation/original) | C99 + LAM/OpenMPI | MPI ranks, dense $2^n\times 2^n$ operator | 2004 artifact, preserved | `make check` (NP=4) | 3 (hard-coded demo) |
| [`implementation/c/`](implementation/c) | C11 + OpenMPI 5.x | MPI ranks, in-place sparse gates | feature-complete | 75 cases × NP=1,2,4 (+NP=8 for `test-large`) | 60 |
| [`implementation/go/`](implementation/go) | Go 1.21 | goroutines + `sync.WaitGroup` | feature-complete | 70 cases, race-clean | 26 |
| [`implementation/python/`](implementation/python) | Python 3.12+ + PyTorch | tensor ops on CUDA / ROCm / MPS / CPU | feature-complete | 414 default + 1 gated Shor-21 | memory-bound; Shor-15/21 tested |

All three new implementations share the same public surface — a
`qreg` register type, the named gates (`H`, `X`, `Y`, `Z`, `S`, `T`,
phase, `Rx/Ry/Rz`), controlled and multi-controlled variants, the
QFT and its inverse, Grover with a user-supplied phase oracle, and
end-to-end Shor factoring with continued-fraction post-processing.
The differences are in *how* the work is parallelised, not what the
library does.

### `implementation/original/` — 2004 dense-matrix MPI

The original code that accompanied the 2004 dissertation, modernised
only at the build wrapper (LAM-MPI is end-of-life; the source now
compiles under OpenMPI 5.x). The bundled `main()` is a single
hard-coded demo: apply $H\otimes H\otimes H$ to $|010\rangle$ and
print the resulting amplitudes from each worker rank.

This implementation is a **proof of concept for the chosen
distribution strategy**, not a working library. It builds the
operator as a dense $2^n\times 2^n$ matrix on rank 0, broadcasts the
input state to every rank, and partitions operator rows across
workers. There is no result gather, no chaining of gates, no
measurement, and no algorithm. The master rank never computes; the
first $8/N$ output rows are silently dropped. Non-power-of-two rank
counts deadlock.

It is preserved exactly because §8 of the revised thesis critiques
this strategy on its own merits — $2^{2n+4}$ bytes of operator
memory makes it unreachable at any cluster size beyond a handful of
qubits — and the sparse-gate alternative in `/c`, `/go`, and
`/python` is the response.

### `implementation/c/` — sparse-gate MPI

The reference implementation of §8's chosen strategy. State is a
flat amplitude array distributed across MPI ranks by the top
$\log_2(\text{NP})$ qubits; gates that touch a global qubit perform
a pairwise `MPI_Sendrecv` against the partner rank; modular
exponentiation uses `MPI_Alltoallv` to bucket the permutation.

Same binary scales from `mpirun -n 1` to a multi-node cluster
without source changes. Tested at NP = 1, 2, 4 on every push and
NP = 1, 2, 4, 8 in the `test-large` job. Shor-15 factors reliably
within 8 attempts; the gated Shor-21 test exercises the 16-qubit
period-finding circuit at NP up to 8.

### `implementation/go/` — goroutine parallelism

The same library API expressed in idiomatic Go. State is shared
memory; the parallel dispatcher (`parallelOverPairs`) spawns
per-call goroutines and joins them with a `sync.WaitGroup`. No
persistent worker pool and no `Destroy()` — the GC reclaims the
register when it goes out of scope.

Programmer-error panics use Go panic with `qubit: ...` prefix;
construction errors return `error`. The package is race-clean
under `go test -race`. Cap at 26 qubits (1 GiB amplitude slice,
2 GiB modular-exp peak); the difference vs `/c`'s 60 is an
allocation bound, not a shift-overflow bound.

### `implementation/python/` — PyTorch tensor ops

The same library API, written as pure-Python on top of PyTorch.
State is a flat `(2**n,)` complex tensor; gates reshape into the
`(2,)*n` view and `tensordot` / `einsum` the unitary against the
target axis — no per-amplitude loops. The "parallel dispatch" is
PyTorch's kernel scheduler on whatever device is available:
NVIDIA CUDA, AMD ROCm, Apple Metal (MPS), or CPU fallback.

Both function-style (`apply_h(q, 0)`) and method-style
(`q.apply_h(0)`) call shapes are first-class. The CLI is exposed
as `qubit-demo` via `[project.scripts]`. Tested under
`ruff check`, `mypy --strict`, and pytest on CPU + MPS locally;
CPU-only on Linux CI.

## Quickstart

Each implementation has its own quickstart in its README. The
shortest paths:

```bash
# Sparse-gate MPI (requires OpenMPI 5.x)
cd implementation/c
make && make test
mpirun -n 4 build/bin/qubit --algo qft

# Go (requires Go 1.21)
cd implementation/go
make test
make demo ALGO=grover

# Python (requires uv)
cd implementation/python
make sync && make check
uv run qubit-demo --algo shor

# 2004 artifact (requires OpenMPI; the bundled binary is 2004 i386 ELF)
cd implementation/original
make && make run NP=4
```

## Cross-implementation parity

Each implementation has an `assessment.md` that maps the thesis
claims (§8 strategy, §9 QFT, §10 Grover, §11 Shor, §12 API) to its
own source files. The same algorithm-level guarantees hold across
all three new implementations:

| Claim | `/c` | `/go` | `/python` |
|---|---|---|---|
| In-place single-qubit gate, $O(2^n)$ | `matrix.c::apply_u_local` | `qubit/gates_single.go::ApplyU` | `qubit/gates_single.py::apply_u` |
| Controlled gates via 4×4 block-diagonal | `matrix.c::apply_cu` | `qubit/gates_controlled.go::ApplyCU` | `qubit/gates_controlled.py::apply_cu` |
| Modular exp as in-place permutation | `shor.c::apply_modular_exp` (+ `MPI_Alltoallv`) | `qubit/shor.go::ApplyModularExp` | `qubit/shor.py::apply_modular_exp` (`torch.gather`) |
| QFT with bit-reversal swaps | `qft.c::apply_qft` | `qubit/qft.go::ApplyQFT` | `qubit/qft.py::apply_qft` |
| QFT inverse (true inverse, not 3× forward) | `qft.c::apply_qft_inverse` | `qubit/qft.go::ApplyQFTInverse` | `qubit/qft.py::apply_qft_inverse` |
| Continued-fraction recovery | `standart.c::continued_fraction` | `qubit/standart.go::ContinuedFraction` | `qubit/standart.py::continued_fraction` |
| Factor $N=15$ reliably | `tests/test_shor.c::test_shor_factor_15` | `qubit/shor_test.go::TestShorFactor15` | `tests/test_shor.py::test_shor_factor_15_seeded` |
| Shor-21 period of 2 mod 21 (gated) | `tests/test_shor.c` (gated by `RUN_SHOR_21=1`) | `qubit/shor_test.go::TestShorPeriodA2Mod21` | `tests/test_shor.py::test_shor_period_a2_mod21_gated` |

The Python `standart.py` and Go `standart.go` deliberately mirror
`/c`'s misspelling of "standard" — visual cue that the three files
are siblings.

## Why three new implementations?

Each parallelism model exposes a different trade-off:

* **MPI (`/c`)** is the natural fit when you actually have a cluster
  and want a single source path that scales from a laptop to a
  multi-node job. Communication is explicit; the partition is
  visible in the code. Cost: every gate that touches a global
  qubit incurs a round trip.
* **Goroutines (`/go`)** is the natural fit on a single shared-memory
  machine — no marshalling, no rank coordination, no `Destroy()`.
  Cost: no path to multiple machines without re-introducing MPI's
  problem space.
* **PyTorch (`/python`)** is the natural fit when you want GPU
  offload without writing CUDA. The simulator never sees the device;
  PyTorch dispatches. Cost: dtype constraints on MPS, and the
  per-tensor allocation pattern is wasteful at very large $n$.

The library API is the same in all three. Reading any one of the
`assessment.md` files alongside `qucomp.pdf` should make the
mapping obvious.

## Out of scope

Across all three new implementations:

* Density matrices and mixed states.
* Noise models.
* Tensor-network or stabilizer-formalism shortcuts.
* Distributed execution across machines for `/go` and `/python`
  (would re-introduce MPI's problem space; `/c` covers it).
* Prime-power detection in `shor_factor` (classical pre-check that
  belongs in the caller).

## Layout

```
implementation/
  original/    # 2004 dense-matrix MPI artifact
  c/           # sparse-gate MPI, modern C11
  go/          # goroutines, Go 1.21
  python/      # PyTorch tensor ops, Python 3.12+
source code/   # LaTeX sources for qucomp.pdf
docs/
  superpowers/
    specs/     # per-implementation design documents
    plans/     # phase-by-phase implementation plans
.github/
  workflows/
    ci.yml     # one workflow covers all four implementations
qucomp.pdf            # revised dissertation (2026)
qucomp-original.pdf   # 2004 submission, preserved
```

## Licence and citation

The dissertation text and the 2004 implementation are reproduced
with permission of the author. The 2026 implementations
(`implementation/c`, `implementation/go`, `implementation/python`)
are released alongside the revised thesis. Cite as:

> Karaduman, A. *Quantum Computation: Implementation of Quantum
> Gates on Linux Clusters*. Bachelor's dissertation, Istanbul Bilgi
> University. Originally submitted 2004; revised 2026.
