# Original 2004 MPI quantum simulator

The C/MPI source under this directory is the original 2004 implementation
that accompanied the dissertation, preserved untouched. The source files
were written for Linux + LAM-MPI and have not been edited beyond fixing
the build wrapper so they compile and run on modern macOS / Linux.

This README:

1. Documents what the code does, and how it relates to what the thesis
   says it does (the two do not entirely agree).
2. Explains how to build and run it on modern macOS in 2026.
3. Records the expected output for the bundled 3-qubit Hadamard demo.

## Contents

| File | Purpose |
|---|---|
| `qubit.c` | Main: hardcoded 3-qubit demo applying $H \otimes H \otimes H$ to $\lvert 010 \rangle$ |
| `matrix.c`, `matrix.h` | Dense `matrix` struct, `create_matrix`, `init_*`, `tensor_product`, `dot_product` |
| `parallel.c`, `parallel.h` | MPI wrappers: `broadcast_matrix`, `send_matrix`, `get_matrix`, `send_row` |
| `standart.c`, `standart.h` | Helpers: `error`, `power`, `Complex_Multiplication`, `power_of_two`, `get_biggest_2s_power` |
| `qubit.doc` | Plain-text API documentation from 2004 |
| `makefile` | Modernised build wrapper (the 2004 wrapper was a single `mpicc` line) |
| `lamhosts` | LAM-MPI host file (single line: `iridium`, the author's 2004 box) |
| `spread` | `scp`-based helper to copy the binary to every host in `lamhosts` |
| `qubit` | The 2004 Linux 32-bit ELF binary, preserved verbatim |
| `build/qubit` | Locally rebuilt macOS binary (gitignored) |

## What the code actually does

The bundled `main()` is a single, hardcoded demonstration --- not a
general-purpose simulator front-end. It does the following:

1. Allocates an 8-amplitude state vector representing the basis state
   $\lvert 010 \rangle$ (rank 0 only).
2. Builds the operator $H \otimes H \otimes H$ as a dense $8 \times 8$
   matrix by repeated tensor product of the $2 \times 2$ Hadamard
   (rank 0 only).
3. `broadcast_matrix` distributes the input state vector to every MPI
   rank.
4. `send_row` distributes rows of the dense operator from rank 0 to the
   higher ranks (rank 0 keeps the first `row_distributed` rows for
   itself but never computes them; see "Discrepancies" below).
5. Each higher rank receives its assigned rows and prints the result of
   `dot_product(row, input_vector)`, i.e., one scalar per row.

The output is therefore the per-row amplitudes of the post-Hadamard
state, split across the worker ranks and printed without any gather.

## Discrepancies between the code and the thesis

The thesis (Section 8 of `qucomp.pdf`) describes a strategy in which the
output vector is gathered after computation so that subsequent gates can
use it as an input. The bundled code implements steps 1--4 of that
strategy but not the gather step. Specifically:

1. **The master never computes.** Rank 0 sends rows starting at
   `1 * row_distributed`; the first `row_distributed` rows are
   nominally "kept on the master" but the master never multiplies them
   by the state vector. With $N$ ranks, only $N-1$ of them do work and
   the first `8 / N` rows of the output are silently dropped.
2. **The remainder of an uneven split is dropped.** When the operator
   row count is not divisible by the largest power of two $\leq N$,
   the `left_over` variable is computed but no rank receives the
   trailing rows.
3. **There is no result gather.** Each worker prints its own slice; the
   master never sees the assembled output. This means the bundled code
   cannot be chained into a multi-gate program even though Section 8
   of the thesis explicitly motivates the chosen partition scheme by
   the need to chain gates.
4. **Single-process mode produces no output.** With `mpirun -n 1` the
   `if (rank != 0)` block on the worker side never executes; the
   program completes silently.
5. **Non-power-of-two rank counts deadlock.** `size_2P` is computed as
   the largest power of two $\leq N$, but every worker rank up to
   $N-1$ still enters the `get_matrix` path and posts `MPI_Recv`. With
   $N=3$, the master only sends to rank 1; rank 2's `MPI_Recv` blocks
   forever. The makefile guards against this with a fail-fast check on
   `NP` (see below), so users get an explanatory error rather than a
   hung `mpirun`.
6. **The demo is hardcoded.** `qureg_size = 3` and the operator
   (Hadamard via tensor product) are baked into `qubit.c`; there is no
   CLI or input-file front-end. To run a different demo you must edit
   the source and rebuild.

These are limitations of the 2004 artifact, not bugs introduced during
this revival. They are recorded here for honest accounting. The
companion thesis chapter 8 critiques the dense-matrix strategy as the
fundamentally wrong primitive regardless of these specific
implementation slips; the sparse-gate API documented in chapter 12 is
what the simulator should look like instead.

## Requirements (macOS, 2026)

This was originally built under LAM-MPI on Linux. LAM-MPI is end-of-life;
the modern equivalent is OpenMPI (or MPICH). On macOS:

```sh
brew install open-mpi
```

That gives you `mpicc` (a Clang wrapper) and `mpirun`. The repository's
top-level toolchain (Apple Clang 17, OpenMPI 5.x) was used to verify the
build for this revision.

On Debian / Ubuntu:

```sh
sudo apt-get install libopenmpi-dev openmpi-bin
```

On Fedora:

```sh
sudo dnf install openmpi openmpi-devel
# then activate the openmpi environment, e.g.:
# module load mpi/openmpi-x86_64
```

## Build

```sh
make                     # produces ./build/qubit (macOS / Linux native)
make clean               # removes ./build
make run                 # runs with 4 MPI ranks (the default)
make run NP=1            # supported alternatives:  NP = 1, 2, 4, 8
make run NP=2
make run NP=8
make check               # builds, runs with NP=4, asserts per-rank amplitudes
```

`NP` must be a power of two no larger than $2^{\text{qureg\_size}} = 8$.
The makefile fails fast with a clear error message for any other value
because the unmodified 2004 source deadlocks on non-power-of-two rank
counts (see Discrepancy 5 above).

The wrapper uses two leniency flags
(`-Wno-implicit-function-declaration`, `-Wno-implicit-int`) so the
unmodified 2004 source compiles under modern C standards. The two
declarations involved are `malloc` (missing `#include <stdlib.h>` in
`matrix.c`) and `get_biggest_2s_power` (defined in `standart.c` but never
declared in `standart.h`). Pre-C99 these were warnings; ISO C99 made
them errors.

In addition, the build emits **25 warnings** that we deliberately leave
in place rather than touch the 2004 source to silence:

| Count | Warning | Origin | Why it is harmless |
|---:|---|---|---|
| 22 | `plain '_Complex' requires a type specifier; assuming '_Complex double'` | every cast or sizeof of bare `__complex__` in `matrix.c`, `parallel.c`, and the forward declarations in `standart.h` | both GCC and Clang default `__complex__` (no base type) to `__complex__ double`, so the implicit and declared types agree |
| 2 | `operator '>>' has lower precedence than '-'` (`-Wshift-op-parentheses`) | `matrix.c:47--48` in `create_qubit` (the recursive base-case split: `value-(value>>size-1)*power(2,size-1)`) | the parentheses the compiler is asking for would change nothing for the well-formed inputs the demo passes; the existing precedence happens to match the author's intent |
| 1 | `non-void function does not return a value in all control paths` (`-Wreturn-type`) | `standart.c` in `get_biggest_2s_power` (no `return` after the loop) | the loop terminates only when it finds a power of two and returns from inside; the trailing path is unreachable for any positive `size` the demo can produce |

The point of listing them is honesty: a fresh `make` will print 25
warning lines, not three. None of them are wrong in a way that affects
the verified output of the 3-qubit demo.

## Expected output (3-qubit Hadamard demo)

The bundled `main()` applies $H \otimes H \otimes H$ to $\lvert 010 \rangle$,
whose exact state-vector output is

$$
H^{\otimes 3} \lvert 010 \rangle
\;=\; \tfrac{1}{\sqrt{8}}\,(+,\,+,\,-,\,-,\,+,\,+,\,-,\,-)^\top
\;\approx\; \begin{pmatrix}
+0.354 \\ +0.354 \\ -0.354 \\ -0.354 \\
+0.354 \\ +0.354 \\ -0.354 \\ -0.354
\end{pmatrix}.
$$

With 4 MPI ranks (`row_distributed = 2`):

| Rank | Output rows | Expected amplitudes |
|---|---|---|
| 0 (master) | --- | (rows 0--1 never computed) |
| 1 | 2, 3 | $-0.354$, $-0.354$ |
| 2 | 4, 5 | $+0.354$, $+0.354$ |
| 3 | 6, 7 | $-0.354$, $-0.354$ |

`make check` asserts the full structure of the NP=4 run:

* exactly 4 occurrences of `(-0.354 + 0.000 i)` in stdout;
* exactly 2 occurrences of `(0.354 + 0.000 i)`;
* exactly one `my rank : N my result :` header for each of ranks 1, 2, 3
  and *zero* headers for rank 0 (verifying that the master never enters
  the worker path);
* and, per-rank, the sign pattern of the printed amplitudes matches the
  expected `[--]`, `[++]`, `[--]` for ranks 1, 2, 3 (verifying that the
  rank-to-rows assignment is what the partition formula says it should
  be). The makefile uses `awk` to associate each amplitude line with
  the rank header that immediately preceded it.

With 2 MPI ranks (`row_distributed = 4`):

| Rank | Output rows | Expected amplitudes |
|---|---|---|
| 0 (master) | --- | (rows 0--3 never computed) |
| 1 | 4, 5, 6, 7 | $+0.354$, $+0.354$, $-0.354$, $-0.354$ |

With 8 MPI ranks (`row_distributed = 1`, requires `--oversubscribe` on a
laptop): ranks 1--7 each compute one row; row 0 is again dropped.

## Notes on the bundled binaries and helper script

* `./qubit` is the original 2004 Linux ELF, 32-bit i386, dynamically linked
  against `/lib/ld-linux.so.2`. It will not run on macOS or on a modern
  64-bit Linux without extensive multilib gymnastics. It is preserved
  verbatim as a historical artifact.
* `./spread` was a shell helper to `scp` the binary to every host listed
  in `./lamhosts` for the 2004 LAM-MPI cluster. It is not useful today
  (modern MPI runtimes have their own provisioning paths) but is left
  in place for the same archival reason.
* `./lamhosts` contains the single hostname `iridium`, the author's
  2004 cluster head. Also left in place verbatim.

## Relation to the rest of the repository

The dense-matrix approach implemented here is critiqued at length in
Section 8 of the dissertation (`source code/parallel_simulation.tex`),
which then proposes the in-place sparse gate application that scalable
state-vector simulators have converged on since. The corresponding API
is documented in Section 12 (`source code/ref_guide.tex`) and is, at the
time of this revision, not yet implemented. A future
`implementation/sparse-gate/` (or similarly named) directory would be
where that lives.
