# Design: implementation/c --- sparse-gate MPI quantum simulator

**Date:** 2026-05-21
**Status:** Approved through §5 in brainstorming; awaiting written spec review.
**Owner:** Arda Karaduman
**Author of this doc:** Claude

## 1. Goal and scope

Build a complete C/MPI quantum-circuit simulator at `implementation/c/`
that covers the union of claims made by:

- The 2004 thesis (now §9 in the revised document) --- the original
  library API.
- The 2026 revision (§8 sparse-gate strategy; §9 QFT; §10 Grover; §11
  Shor; §12 corrected `qreg` library API).

The 2004 dense-matrix implementation at `implementation/original/` is
preserved untouched as a historical artifact. The new implementation is
greenfield in design (sparse-gate primitives, in-place state-vector
updates, distributed across MPI ranks) but reuses 2004 file names for
continuity: `matrix.c/h`, `parallel.c/h`, `standart.c/h`, `qubit.c`,
`makefile`, `README.md`, `assessment.md`. The 2004 LAM-MPI artefacts
(`qubit` ELF binary, `spread`, `lamhosts`, `qubit.doc`) are removed
from `/c` --- they remain in `/original/`.

## 2. Constraints and conventions

| Constraint | Value | Source |
|---|---|---|
| Distributed from line one | Every primitive is MPI-aware; same code runs at NP=1, 2, 4, 8, ... | User decision (brainstorm Q1) |
| API starts from thesis §12 and co-updates the thesis when extended | Existing thesis §12 signatures are preserved **unless** explicitly listed as a v1 refinement in §6 (e.g. `apply_grover` takes a `void *user` callback context; `apply_shor_period` returns a `shor_period_result` struct rather than void). Every refinement, whether a new function or a signature change, is committed back to the thesis source in the same change set so the two stay in sync. See §6.1 for the additions disclosed up front and §6.5--§6.6 for the explicit refinements. | User decision (brainstorm Q2) |
| Existing 2004 file names preserved | Contents fully rewritten; git shows continuity | User decision (brainstorm Q3) |
| Test framework | Unity (vendored at `tests/unity/`) | User decision (brainstorm Q4) |
| Qubit indexing | 0-indexed from LSB; matches thesis §5 | Thesis §5.2 |
| Tensor convention | Big-endian: qubit $n-1$ is the most-significant bit | Thesis §5.2 |
| MPI rank count | Must be a power of two; validated at `qreg_create` | Required by partition scheme |
| Complex type | `complex double` (C99) | Modern standard |
| Numeric tolerance | $10^{-10}$ per amplitude component, $10^{-9}$ for probabilities | Spec choice |

## 3. File layout

```
implementation/c/
├── matrix.h, matrix.c       # qreg struct + single/two-qubit gate primitives
├── parallel.h, parallel.c   # MPI exchange layer (Sendrecv, Alltoallv, broadcast)
├── standart.h, standart.c   # gcd, mod_pow, continued fractions, complex helpers
├── qft.h, qft.c             # QFT + inverse QFT
├── grover.h, grover.c       # Grover diffusion + iteration
├── shor.h, shor.c           # modular_exp shortcut + period finding + factoring
├── qubit.c                  # demo main with --algo flag (bell, qft, grover, shor)
├── makefile                 # build wrapper
├── README.md                # public docs for the C library
├── assessment.md            # living coverage matrix vs the thesis
└── tests/
    ├── unity/                    # vendored Unity (unity.h, unity_internals.h, unity.c)
    ├── test_runner.h             # shared MPI-aware main() macro (see §7.2)
    ├── test_assert.h             # ASSERT_NEAR_AMP, ASSERT_NORM_ONE, etc.
    ├── test_matrix.c             # → builds to build/tests/test_matrix
    ├── test_parallel.c           # → build/tests/test_parallel
    ├── test_distributed_gates.c  # → build/tests/test_distributed_gates  (P1-driven)
    ├── test_standart.c           # → build/tests/test_standart
    ├── test_qft.c                # → build/tests/test_qft
    ├── test_grover.c             # → build/tests/test_grover
    └── test_shor.c               # → build/tests/test_shor
```

## 4. Data model

The state vector is the only authoritative quantum data. No operator
matrix is ever materialised.

```c
typedef struct {
    complex double *amp;     /* local slice, length local_size                */
    int      n_qubits;       /* global qubit count                            */
    int      rank, n_procs;  /* MPI rank and total processes (size = 2^p)     */
    size_t   local_size;     /* = 2^(n_qubits - p), guaranteed power of 2     */
    MPI_Comm comm;
} qreg;
```

### 4.1 Distribution

With $P = 2^p$ ranks and $n$ qubits:

* Each rank holds $2^{n-p}$ contiguous amplitudes.
* Rank $r$ owns global indices $[\,r \cdot 2^{n-p},\; (r+1) \cdot 2^{n-p}\,)$.
* For any global index $g$: $\text{rank} = g \gg (n-p)$, $\text{offset} = g \;\&\; (2^{n-p} - 1)$.

### 4.2 Local vs global qubits

Set once at `qreg_create`:

* Qubits $0 \ldots n-p-1$ are **local** --- fully resident on every rank.
* Qubits $n-p \ldots n-1$ are **global** --- touching them requires inter-rank communication.

### 4.3 Validation and bounds

#### Lifecycle: returns NULL on failure

`qreg_create` returns `NULL` if any of:

* `n_procs` is not a power of two.
* `n_procs > 2^n_qubits`.
* `n_qubits < 1`.
* `n_qubits > QREG_MAX_QUBITS` (see bounds below).

#### Size and shift bounds

The simulator uses `size_t` for global state-vector indices and `1ULL << k`
for bit-mask construction. To keep all index arithmetic and shifts
well-defined on 64-bit systems, we set

```c
#define QREG_MAX_QUBITS 60      /* in matrix.h */
```

which leaves a comfortable 4-bit headroom over the largest plausible
state vector ($2^{60}$ amplitudes $= 16{,}384$ exabytes --- far past any
real cluster). `qreg_create` rejects `n_qubits` above the cap. This
guarantees every `(1ULL << k)` shift in the code base is well-defined
($k < 64$), and that no $2^n$ computation overflows `size_t`.

#### Gate-argument validation

Gate functions are programmer-facing primitives, not user-input-facing.
Invalid arguments are programmer bugs and must abort loudly in every
build configuration, including release builds compiled with `-DNDEBUG`.

Standard `<assert.h>` `assert()` is **not** sufficient for this because
`NDEBUG` disables it (and downstream consumers compiling against
`libqubit.a` may define `NDEBUG` themselves). We therefore define a
project-local macro in `matrix.h`:

```c
/* Always-on assert. Survives -DNDEBUG. */
#define QREG_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "QREG_ASSERT failed at %s:%d: " msg "\n" \
                            "  condition: %s\n", \
                            __FILE__, __LINE__, #cond); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)
```

`MPI_Abort` rather than `abort()` because under `mpirun` a plain
`abort()` on one rank can leave the others hanging in collective calls.
Every check listed in the table below is enforced with `QREG_ASSERT`.

The required checks:

| Argument class | Required check (must hold or `QREG_ASSERT` fires) |
|---|---|
| `qreg_init_basis(q, basis)` | $0 \le \text{basis} < 2^{q\to n\_qubits}$ |
| Single-qubit gate target $t$ | $0 \le t < q\to n\_qubits$ |
| Two-qubit gate $(c, t)$ | $0 \le c, t < q\to n\_qubits$ and $c \ne t$ |
| Controlled-U $(c, t, U)$ | as above; $U$ pointer non-`NULL` |
| Multi-controlled gates | every control index in range; controls pairwise distinct; controls disjoint from target |
| QFT `(start, n)` | $n \ge 1$, $0 \le start$, $start + n \le q\to n\_qubits$ |
| Shor `apply_modular_exp(...)` | counting register and target register both fit; ranges disjoint; $N \ge 2$; $\gcd(a, N) = 1$ (asserted **at this primitive itself**, not only at `shor_factor`, because `apply_modular_exp` is publicly exposed and unit-tested in isolation); $N \le 2^{\text{target\_width}}$ so $(a^x y) \bmod N$ never falls outside the target register's representable range |
| Measurement | target in range; `prob_of` basis $< 2^{n\_qubits}$ |

The `make DEBUG=1` build additionally enables sanitiser instrumentation
(`-fsanitize=address -fsanitize=undefined`) so off-by-one slips surface
even when an assert condition is mis-written.

The rationale for `QREG_ASSERT` over error-return codes is simplicity:
none of the documented entry points have a legitimate way to be called
with a bad qubit index in a working program, and threading status codes
through every primitive would clutter every test and demo without
catching real bugs.

#### Why single-process still goes through the same paths

For $P = 1$ (single process) every qubit is local; no MPI calls fire
even though the code path is MPI-aware. The same binary covers
single-node and cluster runs.

## 5. Gate model

### 5.1 Single-qubit gate on qubit $k$

**$k$ local** ($k < n-p$):

```c
size_t stride = 1ULL << k;
size_t step   = stride << 1;
for (size_t base = 0; base < local_size; base += step)
    for (size_t off = 0; off < stride; off++) {
        size_t i0 = base + off;
        size_t i1 = i0 + stride;
        complex double a0 = amp[i0], a1 = amp[i1];
        amp[i0] = u[0][0]*a0 + u[0][1]*a1;
        amp[i1] = u[1][0]*a0 + u[1][1]*a1;
    }
```

$O(2^{n-p})$ work per rank, zero communication.

**$k$ global** ($k \ge n-p$):

* Partner rank: `partner = rank XOR (1 << (k - (n-p)))`.
* Bit value of qubit $k$ on this rank: `mybit = (rank >> (k - (n-p))) & 1`.
* Each rank `MPI_Sendrecv`s its entire local slice with the partner.
* If `mybit == 0`, apply the top row of $U$ keeping `amp[i] = u[0][0]*amp[i] + u[0][1]*buf[i]`.
* If `mybit == 1`, apply the bottom row: `amp[i] = u[1][0]*buf[i] + u[1][1]*amp[i]`.

$O(2^{n-p})$ data exchanged per global-qubit gate per rank, one round-trip.

### 5.2 Controlled gate (control $c$, target $t$)

A controlled gate combines amplitudes whose indices differ in the
**target** bit, gated by the control being set. The control is a
predicate; the partner-finding logic is always driven by the target.
Four cases by locality of $c$ and $t$ (with `tbit = t - (n-p)` and
`cbit = c - (n-p)` for the global indices into the rank-bit space):

1. **Both local:** iterate amplitudes where bit $c$ is set, apply target gate to bit $t$. No communication.

2. **$c$ local, $t$ global:** partner is `rank XOR (1 << tbit)`. Every rank exchanges its full slice via `MPI_Sendrecv`. On the received buffer, only amplitudes whose **local** bit $c$ is set participate in the $2\times2$ multiply; the rest are passed through unchanged.

3. **$c$ global, $t$ local:** the control bit is constant within a rank. If `(rank >> cbit) & 1` is 0 the gate is a no-op on this rank; if 1 the gate reduces to a local single-qubit gate on bit $t$. No communication.

4. **Both global:** partner is still `rank XOR (1 << tbit)` (driven by the **target** bit, not the control). Because the partner differs from us only in bit $t$, the partner's control bit value equals ours, so the rank-pair is uniformly "control set" or "control clear":
    * If `(rank >> cbit) & 1 == 0`: no exchange, no work --- the gate is a no-op on this rank-pair.
    * If `(rank >> cbit) & 1 == 1`: pairwise `MPI_Sendrecv` exchange, then apply the $2\times2$ to every amplitude pair (which is identical in shape to the case-2 application without the per-element control predicate).

   The earlier draft of this section said partner was by control's bit, which would have exchanged the wrong amplitudes. The fix is to always XOR by `tbit` and let the control bit decide whether to act.

### 5.3 Multi-controlled-Z

Special case used by Grover's diffusion. Phase-flips the single amplitude
$\lvert 1 \ldots 1\rangle$. In the simulator this is one amplitude on
exactly one rank --- $O(1)$.

### 5.4 Measurement

* `prob_of(q, basis)`: O(1) on the owning rank, `MPI_Bcast` to all.
* `measure_qubit(q, k)`:
    1. Each rank sums $\sum |a_i|^2$ for amplitudes with bit $k = 0$.
    2. `MPI_Allreduce` sums across ranks → $p_0$ globally.
    3. Rank 0 draws a sample, `MPI_Bcast`s the outcome.
    4. Every rank zeros amplitudes inconsistent with the outcome and rescales by $1 / \sqrt{p}$.
* `measure_all`: cumulative-prob sample via reduction; on success, collapse to the chosen basis state.

### 5.5 Modular exponentiation (Shor)

`apply_modular_exp(q, counting_start, t, target_start, n, a, N)` with
`uint64_t a, N` (matches thesis Listing 7 after the type widening):

1. Each rank walks its local slice. For each non-zero amplitude, decompose the global index into $(x, y, \text{rest})$ where $x$ is the counting register's value and $y$ is the target register's. Compute the new $y$:
    * if $0 \le y < N$: $y_{\text{new}} = (a^x \cdot y) \bmod N$.
    * if $y \ge N$ (the target register is wider than $N$ requires, so $2^n - N$ values lie outside the modular ring): **$y_{\text{new}} = y$**. Leaving these values fixed makes the map a permutation of $\{0, \ldots, 2^n-1\}$ --- and therefore a valid unitary --- rather than the many-to-one collapse that $\lvert y\rangle \to \lvert a^x y \bmod N\rangle$ would be when $y \ge N$. This is the standard simulator convention and matches the structure of the honest circuit (which only modifies values in the modular ring).
2. Reassemble the destination global index from $(x, y_{\text{new}}, \text{rest})$.
3. Bucket `(new_global, amplitude)` pairs by destination rank.
4. Single `MPI_Alltoallv` exchanges the pairs.
5. Each rank accumulates received pairs into a fresh local slice.
6. Replace `amp` with the new slice; free buckets.

## 6. Module-by-module surface

### 6.1 matrix.h --- qreg lifecycle + primitive gates

Functions from thesis §12 reproduced exactly, plus a small set of additions
(marked **ext**) that fall out naturally during implementation. Per the
brainstorming agreement, each addition triggers a parallel update to
thesis §12 so the API and documentation stay in sync.

API additions over thesis §12: `apply_y, apply_s, apply_t,
apply_rx, apply_ry, apply_rz` (rounding out the single-qubit set);
`apply_cz` (constant-overhead Pauli-Z control); `apply_multi_controlled_z,
apply_multi_controlled_x` (needed by Grover and reusable); `qreg_clone,
qreg_norm, qreg_dump` (utilities); `sample_distribution` (shots-style
sampling); `shor_factor` (high-level end-to-end factoring wrapper). Each
of these is a strict superset of what the thesis documents and does not
contradict any signature it already specifies.

```c
qreg *qreg_create   (int n_qubits, MPI_Comm comm);
void  qreg_destroy  (qreg *q);
qreg *qreg_clone    (const qreg *q);
void  qreg_init_basis(qreg *q, size_t basis_state);

/* single-qubit */
void apply_h    (qreg *q, int target);
void apply_x    (qreg *q, int target);
void apply_y    (qreg *q, int target);
void apply_z    (qreg *q, int target);
void apply_s    (qreg *q, int target);
void apply_t    (qreg *q, int target);
void apply_phase(qreg *q, int target, double theta);
void apply_rx   (qreg *q, int target, double theta);
void apply_ry   (qreg *q, int target, double theta);
void apply_rz   (qreg *q, int target, double theta);
void apply_u    (qreg *q, int target, complex double u[2][2]);

/* two-qubit */
void apply_cnot            (qreg *q, int control, int target);
void apply_cz              (qreg *q, int control, int target);
void apply_controlled_phase(qreg *q, int control, int target, double theta);
void apply_cu              (qreg *q, int control, int target,
                            complex double u[2][2]);
void apply_swap            (qreg *q, int a, int b);

/* multi-controlled */
void apply_multi_controlled_z(qreg *q, int n);   /* phase flip on |1...1>  */
void apply_multi_controlled_x(qreg *q,
                              const int *controls, int n_controls,
                              int target);       /* generalised Toffoli   */

/* measurement */
size_t measure_all   (qreg *q);
int    measure_qubit (qreg *q, int target);
double prob_of       (const qreg *q, size_t basis);
void   sample_distribution(const qreg *q, size_t *out, int shots);

/* utilities */
double qreg_norm     (const qreg *q);
void   qreg_dump     (const qreg *q, FILE *f);   /* rank 0 prints global state */
```

### 6.2 parallel.h --- MPI exchange primitives

Internal-but-exposed:

```c
void exchange_amplitudes(qreg *q, int partner_rank);
                                /* full-slice Sendrecv, used by global gates */
void redistribute_pairs(qreg *q, size_t *new_indices, size_t n_pairs,
                        const complex double *vals);
                                /* Alltoallv bucket + accumulate, used by    */
                                /* modular_exp                               */
int  is_local_qubit (const qreg *q, int k);
int  partner_for    (const qreg *q, int k);   /* XOR-by-bit partner          */
int  rank_owns      (const qreg *q, size_t global_index);
size_t global_to_local(const qreg *q, size_t global_index);
size_t local_to_global(const qreg *q, size_t local_index);
```

### 6.3 standart.h --- numerical helpers

```c
uint64_t gcd_u64    (uint64_t a, uint64_t b);
uint64_t mod_pow    (uint64_t base, uint64_t exp, uint64_t mod);
void continued_fraction(double x, uint64_t max_denominator,
                        uint64_t *num, uint64_t *den);
                       /* returns the best convergent p/q of x with q <= max */
int      is_power_of_two(int x);
int      ilog2_u32  (uint32_t x);   /* assumes x is power of two            */
```

### 6.4 qft.h

```c
void apply_qft        (qreg *q, int start, int n_qubits);
void apply_qft_inverse(qreg *q, int start, int n_qubits);
```

`start` is the LSB-end qubit index of the contiguous range that holds the
QFT input/output. `n_qubits` is the width of that range.

**Bit-order convention: `apply_qft` includes the final bit-reversal swaps**,
matching the listing in thesis §9.3. This means that if the input register
encodes integer $x$ in the natural little-endian-by-qubit layout
($\text{bit } k \to 2^k$), the output amplitude at index $y$ in the same
layout equals
$\tfrac{1}{\sqrt{N}}\sum_x \alpha_x\,e^{2\pi i\, xy / N}$. In particular,
when used inside Shor's algorithm, the integer obtained by measuring the
counting register after `apply_qft_inverse` can be fed directly to the
continued-fraction step without any caller-side bit reversal.
`apply_qft_inverse` likewise performs swaps so the inverse round-trip
returns the original state.

### 6.5 grover.h

```c
typedef void (*oracle_fn)(qreg *q, void *user);
void apply_grover(qreg *q, int n_qubits, oracle_fn oracle, void *user,
                  int iterations);
```

`oracle` is a user callback that applies the phase oracle
$O_f \lvert x\rangle = (-1)^{f(x)} \lvert x\rangle$ in place. `user` is
a context pointer the oracle can use to know what to mark.

### 6.6 shor.h

```c
typedef struct {
    uint64_t r;          /* recovered period, 0 if failed         */
    uint64_t measured_c; /* the integer the QFT measurement gave  */
} shor_period_result;

shor_period_result apply_shor_period(qreg *q,
                                     int counting_start, int t,
                                     int target_start, int n,
                                     uint64_t a, uint64_t N);

typedef struct {
    uint64_t p, q;       /* non-trivial factors of N, 0 if failed */
    int      attempts;
} shor_factor_result;

shor_factor_result shor_factor(uint64_t N, int max_attempts);
```

`shor_factor` is the high-level entry that loops: pick random $a$, build
a fresh `qreg` of the right size, call `apply_shor_period`, post-process
via continued fractions and gcd, retry until factors found or attempts
exhausted.

## 7. Testing

### 7.1 Framework

Unity vendored at `tests/unity/`. Adds three files (`unity.h`,
`unity_internals.h`, `unity.c`). MIT licensed.

**One binary per test_<module>.c file.** Each module's tests link into
their own standalone executable under `build/tests/test_<module>`. This
keeps registration simple (each file defines its own `register_tests()`
and inherits `main()` via the shared header), avoids name collisions
across modules, and lets us run, debug, or rerun a single module in
isolation under `mpirun`.

Test files include `"unity.h"` and `"test_runner.h"`, define module-local
`TEST_*` functions, and a `register_tests()` function that calls
`RUN_TEST(...)` for each one.

### 7.2 MPI-aware runner

The shared `main()` lives in `tests/test_runner.h` as a macro the module
file expands once. Pseudocode:

```c
/* tests/test_runner.h */
#define TEST_RUNNER_MAIN()                                                  \
    void register_tests(void);                                              \
    int main(int argc, char **argv) {                                       \
        MPI_Init(&argc, &argv);                                             \
        int rank;  MPI_Comm_rank(MPI_COMM_WORLD, &rank);                    \
        /* Redirect non-rank-0 chatter BEFORE UnityBegin so the report      \
         * from rank 0 is the only thing the user sees. Both stdout and    \
         * stderr because Unity writes some failure detail to stderr.       */\
        if (rank != 0) {                                                    \
            freopen("/dev/null", "w", stdout);                              \
            freopen("/dev/null", "w", stderr);                              \
        }                                                                   \
        UnityBegin(__FILE__);                                               \
        register_tests();                                                   \
        /* Every rank calls UnityEnd: it is what finalises Unity's          \
         * failure count and returns it. Calling UnityEnd only on rank 0    \
         * left Unity.TestFailures in an undefined state on other ranks     \
         * and could mask non-rank-0 failures.                              */\
        int unity_fail = UnityEnd();                                        \
        int local_fail  = (unity_fail != 0) ? 1 : 0;                        \
        int global_fail = 0;                                                \
        MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_LOR,       \
                      MPI_COMM_WORLD);                                      \
        MPI_Finalize();                                                     \
        return global_fail;                                                 \
    }
```

Each `test_<module>.c` ends with `TEST_RUNNER_MAIN()`. Non-rank-0
processes redirect both stdout and stderr to `/dev/null` before
`UnityBegin` so only rank 0's report reaches the user. Every rank still
calls `UnityEnd` --- it is what finalises Unity's failure count --- and
the executable's exit code is the `MPI_LOR` across ranks, so failures
on any rank surface as a non-zero exit. The test runner in the makefile
asserts that exit code at NP=1, 2, 4 (and 8 under
`test-large`).

### 7.3 Coverage matrix

| Module | Test cases |
|---|---|
| `test_matrix` | qreg_create rejects non-power-of-two NP; init_basis correct for many values of `basis`; norm invariant; single-qubit gates against analytic reference $(H, X, Y, Z, S, T, R_\theta)$; CNOT entanglement-creation from $\lvert+0\rangle \to \lvert\Phi^+\rangle$; CZ; SWAP swaps; multi-controlled-Z flips only $\lvert 1...1\rangle$; `apply_u` for a random unitary preserves norm |
| `test_parallel` | exchange_amplitudes round-trip restores state; partner_for arithmetic for both single-qubit and target-bit cases; rank_owns and global-to-local round-trip; broadcast_init has the same state on every rank afterward |
| `test_distributed_gates` (**new, P1-driven**) | At NP $\geq 2$, exercise every locality combination of controlled gates against the **analytic global state vector**, gathered to rank 0 via `qreg_dump` (or equivalent allgather) and compared component-wise. The cases: (a) single-qubit $H$ on a global qubit applied to a chosen basis input must yield the known $\tfrac{1}{\sqrt{2}}(\lvert\cdot 0\rangle + \lvert\cdot 1\rangle)$ pattern; (b) CNOT with (control local, target global) applied to $\lvert+0\rangle$ on the relevant pair must yield $\lvert\Phi^+\rangle$; (c) CNOT with (control global, target local) likewise; (d) CNOT with (control global, target global) likewise --- the case where the bad partner-by-control routing of the earlier spec draft would silently exchange the wrong amplitudes. Each case asserts against the analytic state component-wise to `AMP_TOL`, and runs at NP=2 and NP=4, with the qubit indices chosen so the local/global boundary falls exactly between the control and target |
| `test_standart` | gcd vs reference; mod_pow vs ground truth; continued_fraction finds 22/7 from $\pi$, 355/113 from $\pi$, $s/r$ for known periods |
| `test_qft` | QFT on 1 qubit equals $H$; QFT $\cdot$ QFT$^{-1}$ = I on 4 qubits; QFT of $\lvert 0...0\rangle$ is uniform $\lvert+\ldots+\rangle$; QFT of a known periodic input has all mass on multiples of $N/r$ |
| `test_grover` | Single marked item in $N=16$, $\geq 0.99$ success probability after $\lfloor\pi/4\sqrt{16}\rfloor = 3$ iterations; 4 marked items in $N=16$ needs $\lfloor\pi/4\cdot 2\rfloor = 1$ iteration; over-iterating reduces success (proves the optimum) |
| `test_shor` | mod_pow consistency with gcd; period of $a^x \bmod 15$ for $a=7$ is 4; **modular_exp leaves $y \ge N$ unchanged (the reversibility-preserving pass-through from §5.5)**; factoring $N=15 \to \{3, 5\}$. (An earlier draft of this row also listed factoring $N=21 \to \{3, 7\}$ under `make test-large`; that case was aspirational and was not implemented in v1 --- it would need a ~16-qubit register and is left as future work, consistent with `implementation/c/assessment.md`.) |

### 7.4 Test runs

`make test` runs each test executable at NP=1, 2, 4. Failure at any NP
fails the suite. `make test-large` reruns the existing suite at NP=8.
(An earlier draft of this section advertised a "Shor-21" case under
test-large; that was aspirational and was not implemented in v1. See
`implementation/c/assessment.md` for the canonical statement of
coverage.)

`make check` is preserved as a fast smoke test (Bell state preparation at
NP=4) for parity with `/original`.

### 7.5 Tolerance constants

In `tests/test_assert.h`:

```c
#define AMP_TOL  1e-10   /* per amplitude component */
#define PROB_TOL 1e-9    /* probabilities and sums */
```

## 8. Build system

```
make                  # libqubit.a + bin/qubit (single demo binary)
make test             # builds tests, runs at NP=1, 2, 4
make test-large       # reruns existing suite at NP=8
make demo ALGO=qft NP=4
make check            # quick smoke-test (Bell state at NP=4)
make clean
make distclean
```

Compiler defaults: `mpicc -std=c11 -O2 -Wall -Wextra`. Override via
`CFLAGS+=`. Debug build: `make DEBUG=1` → `-O0 -g -fsanitize=address`
where Clang supports it.

The Unity .c file is compiled separately with `-Wno-format -Wno-unused`
because of its variadic printf conventions.

Build artefacts go to `build/`; the makefile, like `/original/`,
preserves the gitignored out-of-tree layout.

## 9. Algorithm scope

### 9.1 In v1

All of:

* Lifecycle: create, destroy, clone, init.
* All single-qubit gates listed in §6.1.
* All two-qubit gates listed in §6.1.
* Multi-controlled-Z and multi-controlled-X.
* Measurement: full, single qubit, probability, sample.
* QFT and inverse.
* Grover with phase-oracle callback.
* Shor period finding (`apply_shor_period`) and high-level factoring (`shor_factor`).
* Demo program covering Bell state, QFT round-trip, Grover with marked
  oracle, Shor factoring 15.

### 9.2 Explicitly out

* Density matrices / mixed states (state vector only)
* Noise models
* Tensor networks / stabilizer formalism
* GPU offload
* Python bindings

### 9.3 Performance targets (single laptop, Apple Silicon)

* 25 qubits comfortably (single process, 512 MB amplitude vector + working memory)
* 30 qubits on 8 ranks distributed across nodes with 4 GB each
* Shor of $N=15$ (~20 qubits) under 5 s wall-clock, single process
* Grover on $N = 2^{20}$, 1 marked item, under 1 s
* QFT on 20 qubits: correctness only, no speed target

These are guidance, not pass/fail gates for the v1 release.

## 10. Coverage tracking

`implementation/c/assessment.md` (rewritten from the `/original/` version)
maintains a living matrix of which thesis claims this implementation
covers, with file:line references for every cell. Updated at the end of
each implementation milestone. The goal is full ✓ coverage of:

* The 2004 library API (now §9 of the revised thesis) --- coverage is
  **functional equivalence via the qreg API**, not literal preservation
  of the 2004 function names. `tensor_product`, `dot_product`,
  `create_matrix`, `send_matrix`, etc. are not exposed by the new
  library; their use cases are served by the qreg primitives. The
  assessment matrix documents the mapping explicitly.
* All §8 (2026) sparse-gate strategy claims
* All §9 (2026) QFT claims
* All §10 (2026) Grover claims
* All §11 (2026) Shor claims
* All §12 (2026) qreg API entries (with the additions listed in §6.1
  of this design appended to thesis §12 as they land)

## 11. Out of scope for this design

* The actual implementation plan (file order, milestone breakdown, what
  gets built first). That goes into the writing-plans skill output once
  this spec is approved.
* Thesis updates. If the implementation reveals an improvement to the
  API or strategy, the thesis source is patched in a separate commit;
  the spec is updated to track.

## 12. Approval

Sections §1--§5 of this spec were each presented in brainstorming and
approved. This document is the consolidated written form.
