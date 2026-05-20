# Assessment: how much of the 2004 paper does this implementation cover?

Roughly **40 %**, with the gap being mostly things the thesis itself defends
as the chosen approach but then does not actually do.

## Section 8 (parallel simulation strategy) --- 5 of 11 claims implemented

The 2004 Section 8 walks through three candidate distribution strategies,
picks one, and lays out an eleven-step argument for how that one works.
The code covers roughly the first half of that argument.

| # | What §8 claims | Implemented? | Where |
|---|---|---|---|
| 1 | SIMD: every node runs the same code on different data | yes | inherent in `mpirun` |
| 2 | One node is a master that distributes data; others compute | yes | `qubit.c` rank-0 vs `rank != 0` paths |
| 3 | Strategy A (one qubit per processor) --- surveyed and rejected | n/a | discussion only |
| 4 | Strategy B (one input-vector row per node) --- surveyed and rejected | n/a | discussion only |
| 5 | Chosen strategy: broadcast the input state vector, partition operator rows | yes | `broadcast_matrix`, `send_row` |
| 6 | Workers compute their row × state-vector dot products | yes | `dot_product` per row in `qubit.c:113` |
| 7 | "After the computation is finished, the nodes begin broadcasting their results" | **no** | workers `printf`; never broadcast |
| 8 | "If each node sent its result to the master, the master would have the output vector" | **no** | no result gather; master never receives |
| 9 | "If the quantum programs are applied sequentially [...] this approach needs to send the output vector to the worker nodes as the input vector of the next quantum program" | **no** | no chaining; the demo applies exactly one gate |
| 10 | "It is wise to keep a copy of the output vector on all nodes" | **no** | never assembled, let alone shared |
| 11 | "Depending on network architecture, use hypercube template" | **no** | star/hypercube discussion is not in code |

The entire downstream half of the chosen strategy --- the gather/redistribute
pattern that motivates the partition choice in the first place --- is
unimplemented. The code can compute one gate's output, scattered across
stdout streams from worker ranks, and that is it.

## Section 9 (library documentation, the C API) --- most primitives present

| Promised in §9 | Implemented? | Where |
|---|---|---|
| `matrix` struct with `__complex__ double**` | yes | `matrix.h` |
| `create_matrix(r, c)` | yes | `matrix.c:5` |
| `init_matrix`, `init_matrix_c`, `init_matrix_n` | yes | `matrix.c:57,73,65` |
| `init_hadamard`, `init_CNOT` | yes | `matrix.c:103,91` |
| `create_qubit(n, value)` | yes | `matrix.c:17` |
| `print_matrix` | yes | `matrix.c:81` |
| `tensor_product`, `dot_product` | yes | `matrix.c:115,134` |
| `send_matrix`, `get_matrix`, `broadcast_matrix` | yes | `parallel.c:88,56,14` |
| `~matrix()` deconstructor (named "Deconstructor" in §9) | **no** | no `free_matrix`; everything leaks |
| QFT | **no** | conclusion admits it is unfinished |
| Grover | **no** | not present |
| Measurement / normalisation checks | **no** | not present |
| Contiguous `__complex__ double*` storage (cache-friendly) | **no** | uses array-of-pointers, as documented |

Roughly **90 % of the documented API exists as code**. The 2004 author was
explicit about this in the conclusion: *"many functions to provide basic
Quantum Computer operations (ie Quantum Fourier Transform) was unfinished."*

## As an actual quantum simulator --- 30 %

Even within the implemented subset, the bundled `main()` ships:

- One hardcoded test: 3 qubits, input $\lvert 010\rangle$, operator $H \otimes H \otimes H$.
- No CLI, no input file, no circuit description format.
- No way to run a second gate without recompiling and rerunning.
- Master rank does no compute work; the first `8/N` output rows of every run are silently dropped.
- Remainder rows on uneven splits are also dropped.
- Non-power-of-two rank counts deadlock (workers post `MPI_Recv` against sends that never come).
- `mpirun -n 1` produces no output at all.

The maths is correct on the rows the workers do compute --- the simulator's
output for $H^{\otimes 3}\lvert 010\rangle$ matches the analytical
$\tfrac{1}{\sqrt{8}}(+,+,-,-,+,+,-,-)^\top$ at `-n 2`, `-n 4`, and `-n 8`.
So the dense-matrix dot-product logic is sound. It is the surrounding
scaffolding that is partial.

## Net read

What the 2004 work actually delivered is a **proof of concept for the
chosen distribution strategy**, not a working quantum-simulator library.
The C reference shows that you can in fact broadcast a state vector,
partition rows of a $2^n \times 2^n$ operator across MPI ranks, and have
each rank compute its slice. That is genuine and verifiable. Everything
beyond that --- gathering, chaining, ergonomics, any algorithm the thesis
name-drops --- was promised in the prose and not written.

The thesis was honest about this in 2004 ("many functions [...] was
unfinished"); the implementation's coverage gap matches the conclusion's
admission.

This assessment is what motivates the structure of the revised
dissertation's Section 8 (`source code/parallel_simulation.tex`): rather
than papering over the gap, the chapter explicitly retains the
dense-matrix strategy as the bundled historical artifact, critiques it on
its own merits ($2^{2n+4}$ bytes of operator memory --- unreachable at
any cluster size), and points forward to the in-place sparse-gate
strategy that the revised library API (Section 12) describes and that a
future `implementation/<name>/` would implement.
