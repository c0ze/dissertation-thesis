# implementation/c

[![CI](https://github.com/c0ze/dissertation-thesis/actions/workflows/ci.yml/badge.svg)](https://github.com/c0ze/dissertation-thesis/actions/workflows/ci.yml)

A sparse-gate, MPI-distributed quantum-circuit simulator written in C.
Distributed from line one; the same binary runs single-process or across
a multi-node cluster.

The pristine 2004 dense-matrix implementation lives next door in
`implementation/original/` and is not built or run by this directory.

See:
* `../../source code/parallel_simulation.tex` (§8 of the dissertation)
  for the strategy this library implements.
* `../../source code/ref_guide.tex` (§12) for the public API.
* `../../docs/superpowers/specs/2026-05-21-implementation-c-design.md`
  for the design document this code follows.
* `assessment.md` for a living matrix of which thesis claims are covered
  and where.

## Requirements

* `mpicc` and `mpirun` from a modern MPI distribution
  (tested with OpenMPI 5.x: `brew install open-mpi` on macOS,
  `sudo apt-get install libopenmpi-dev openmpi-bin` on Debian/Ubuntu).
* A C11 compiler. The makefile defaults to `mpicc -std=c11 -O2`.

## Build and test

```sh
make            # builds libqubit objects + bin/qubit demo
make test       # runs every tests/test_*.c at NP = 1, 2, 4
make test-large # NP = 1, 2, 4, 8; sets RUN_SHOR_21=1 which unlocks the
                # 16-qubit test_shor_period_a2_mod21 period-finding test
                # (skipped in the default `make test` loop for iteration
                # speed and test-suite layering)
make demo ALGO=qft NP=4
make clean
```

## Quick examples

```c
#include "matrix.h"
#include "qft.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_qft(q, 0, 3);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /* prob_of is collective (MPI_Allreduce internally) — every rank
     * must call it. Printing then gated to rank 0 to avoid duplicate
     * output. The earlier draft of this example had the call inside
     * the rank-0 guard, which would deadlock at NP > 1.              */
    for (size_t i = 0; i < 8; i++) {
        double p = prob_of(q, i);
        if (rank == 0) printf("prob(|%zu>) = %.4f\n", i, p);
    }
    qreg_destroy(q);
    MPI_Finalize();
}
```

## File layout

| File | Purpose |
|---|---|
| `matrix.h` / `matrix.c` | qreg struct, lifecycle, all gate primitives, measurement |
| `parallel.h` / `parallel.c` | MPI exchange (pairwise Sendrecv, Alltoallv bucketing, locality helpers) |
| `standart.h` / `standart.c` | gcd, modular exponentiation, continued fractions, power-of-two helpers |
| `qft.h` / `qft.c` | Quantum Fourier Transform and its inverse, with final bit-reversal swaps |
| `grover.h` / `grover.c` | Grover's algorithm with a user-supplied phase oracle |
| `shor.h` / `shor.c` | Shor's modular exponentiation, period finding, and end-to-end factoring |
| `qubit.c` | Demo program with `--algo {bell,qft,grover,shor}` |
| `tests/` | Per-module test binaries plus the MPI-aware Unity wrapper |
| `assessment.md` | Living coverage matrix vs the thesis |

## Constraints

* `NP` (MPI process count) must be a power of two.
* `n_qubits` capped at `QREG_MAX_QUBITS = 60` (defined in `matrix.h`).
* All gate-argument errors are programmer bugs and abort via `QREG_ASSERT`
  in every build configuration (including release builds with `-DNDEBUG`),
  using `MPI_Abort` so collective hangs don't happen on partial failure.

## Status

See `assessment.md`. The v1 library covers every public function listed
in spec §6 (which is thesis §12 plus disclosed extensions); end-to-end
Shor reliably factors `N = 15` within 8 attempts at NP = 1, 2, 4.
