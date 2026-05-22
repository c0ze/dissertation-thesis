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
* A C11 compiler with GCC/Clang `__uint128_t` support. `standart.c`'s
  `mod_pow` uses 128-bit intermediates to dodge overflow during
  `(a*b) mod N` on 64-bit moduli; both GCC and Clang on every
  platform we test on (Linux x86_64, macOS arm64) provide
  `__uint128_t` as a non-standard extension. The makefile defaults to
  `mpicc -std=gnu11 -O2` to leave that extension exposed.

## Build and test

```sh
make            # compiles library .o files under build/ + bin/qubit demo
                # (no libqubit.a archive is produced; tests + the demo
                #  link the object files directly)
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
* **Collective discipline.** Every operation on a distributed `qreg`
  should be treated as collective over `q->comm`. Every rank that
  owns the register must call the same gate / measurement / reduction
  routine in the same order. This is enforced today by the
  measurement and reduction code paths (which call
  `MPI_Allreduce` / `MPI_Bcast` internally); local-only gates do not
  communicate today but should still be called on every rank, so that
  the same source path works under a later `NP` where the same gate
  would touch a global qubit.

## v1 limitations

These are the deviations from a fully scalable / fully featured
implementation that you should know about before pushing the library
past its current envelope:

* **MPI counts are `int`.** `exchange_amplitudes` and the
  `Alltoallv` calls in `redistribute_pairs` cast `local_size` and
  the per-rank send / receive counts to `int`. That caps a single
  MPI message at `INT_MAX ≈ 2^31 - 1` complex amplitudes, i.e. a
  per-rank slice of up to 31 local qubits before per-message
  chunking becomes necessary. Beyond that, either chunk the
  exchanges around `INT_MAX` or move to MPI-4 large-count APIs
  (`MPI_Sendrecv_c`, `MPI_Alltoallv_c`) keyed on `MPI_Count`.
* **`apply_multi_controlled_x` is local-only.** Every control and
  the target must be a local qubit (index `< n_qubits - p`). The
  distributed version would itself decompose into a Toffoli +
  ancilla ladder and is left for a follow-up. Hitting this from a
  global qubit aborts via `QREG_ASSERT`.
* **`apply_grover` precondition.** The routine applies `H` to each
  of the first `n_qubits` qubits and then iterates oracle +
  diffusion; if the register is not in `|0...0>` on entry, the same
  gate sequence still runs but the meaning of the result is no
  longer standard Grover. Call `qreg_init_basis(q, 0)` first if in
  doubt. The bundled tests do this explicitly.
* **`apply_shor_period` returns a candidate.** The denominator from
  continued-fraction recovery is not guaranteed to equal the true
  order of `a mod N`. Verify with `a^r mod N == 1` (or fail back to
  the retry loop) before treating it as the order. `shor_factor`
  does this internally; callers that go through `apply_shor_period`
  directly must handle it themselves.
* **Measurement RNG is `rand()`.** `measure_qubit`, `measure_all`,
  and `sample_distribution` use the C standard library RNG on rank
  0 (the outcome is broadcast to the other ranks for collective
  correctness). Without an explicit seed the sequence is
  deterministic across runs. Call `qreg_seed(q, seed)` to seed
  before measurement; `shor_factor` seeds itself on first use.

## Status

See `assessment.md`. The v1 library covers every public function listed
in spec §6 (which is thesis §12 plus disclosed extensions); end-to-end
Shor reliably factors `N = 15` within 8 attempts at NP = 1, 2, 4.
