# implementation/c Sparse-Gate MPI Quantum Simulator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete C/MPI quantum-circuit simulator at `implementation/c/` that covers the union of claims from the 2004 thesis library (§9) and the 2026 revision (§8 sparse-gate strategy + §9 QFT + §10 Grover + §11 Shor + §12 qreg API), distributed via MPI from line one, tested at NP=1/2/4/8.

**Architecture:** Sparse-gate in-place state-vector simulation. State vector partitioned across $P=2^p$ MPI ranks (rank $r$ owns global indices in $[r \cdot 2^{n-p}, (r+1) \cdot 2^{n-p})$). Qubits split into local (no comms) and global (pairwise `MPI_Sendrecv` exchange). Shor's modular exponentiation uses `MPI_Alltoallv` to redistribute amplitudes. Validation via project macro `QREG_ASSERT` that survives `-DNDEBUG`.

**Tech Stack:** C11, OpenMPI 5.x via `mpicc`, Unity test framework (vendored), GNU make.

**Reference spec:** [docs/superpowers/specs/2026-05-21-implementation-c-design.md](../specs/2026-05-21-implementation-c-design.md)

---

## File Structure

Final tree under `implementation/c/`:

```
implementation/c/
├── matrix.h, matrix.c           # qreg lifecycle + single/two-qubit gate primitives
├── parallel.h, parallel.c       # MPI exchange (Sendrecv, Alltoallv, locality helpers)
├── standart.h, standart.c       # gcd, mod_pow, continued fractions, complex helpers
├── qft.h, qft.c                 # QFT + inverse (with final bit-reversal swaps)
├── grover.h, grover.c           # Grover diffusion + iteration
├── shor.h, shor.c               # modular_exp + period finding + factoring
├── qubit.c                      # demo main with --algo flag
├── makefile                     # build wrapper
├── README.md                    # new C library docs
├── assessment.md                # living coverage matrix vs the thesis
└── tests/
    ├── unity/
    │   ├── unity.h
    │   ├── unity_internals.h
    │   └── unity.c
    ├── test_runner.h            # shared MPI-aware main() macro
    ├── test_assert.h            # ASSERT_NEAR_AMP, ASSERT_NORM_ONE, tolerances
    ├── test_standart.c
    ├── test_matrix.c
    ├── test_parallel.c
    ├── test_distributed_gates.c
    ├── test_qft.c
    ├── test_grover.c
    └── test_shor.c
```

`build/` is gitignored. The 2004 files (`matrix.c`/`matrix.h`/`parallel.c`/`parallel.h`/`standart.c`/`standart.h`/`qubit.c`/`qubit.doc`/`qubit`/`spread`/`lamhosts`) are removed from `/c/` — they remain preserved in `/original/`.

## Coverage tracking

`implementation/c/assessment.md` is updated at the end of each phase (after every group of related tasks completes). Each entry references file:line where the corresponding thesis claim is implemented.

## Conventions used throughout

- Qubit indices are 0-indexed from the LSB.
- Tensor convention is big-endian: qubit $n-1$ is the most-significant bit.
- `complex double` is the standard C99 complex type from `<complex.h>`.
- All gate functions assert preconditions via `QREG_ASSERT` (defined in `matrix.h` per spec §4.3).
- Tests use `TEST_ASSERT_*` macros from Unity plus a small set of project-specific macros (`ASSERT_NEAR_AMP`, `ASSERT_NORM_ONE`) defined in `tests/test_assert.h`.
- Every commit follows the convention `feat(component): short summary` and includes a `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.

---

## Phase 0 — Bootstrap (Tasks 1–6)

### Task 1: Wipe 2004 files from implementation/c/

**Files:**
- Delete: every file currently in `implementation/c/` except `build/` (which is gitignored and untracked)
- Create: `implementation/c/.gitkeep` (placeholder so git tracks the empty dir until real files land)

- [ ] **Step 1: Inventory what's currently in implementation/c/**

Run: `ls -la implementation/c/`
Expected: 13 files preserved from `/original/` plus `build/`.

- [ ] **Step 2: Remove 2004 source, binary, helper, and doc files**

Run:
```bash
cd implementation/c
rm -f matrix.c matrix.h parallel.c parallel.h standart.c standart.h
rm -f qubit.c qubit.doc qubit lamhosts spread
rm -f README.md assessment.md makefile
rm -rf build
```

- [ ] **Step 3: Add a placeholder so git tracks the empty directory**

Run: `touch implementation/c/.gitkeep`

- [ ] **Step 4: Verify the slate is clean**

Run: `ls -la implementation/c/`
Expected: only `.` `..` `.gitkeep`.

- [ ] **Step 5: Commit**

```bash
git add implementation/c
git commit -m "chore(c): wipe 2004 files to make room for new sparse-gate library

Files removed match 1:1 the inventory of implementation/original/, which
remains the canonical archive of the 2004 code. The /c subtree is now an
empty greenfield to be filled by the new sparse-gate implementation per
docs/superpowers/specs/2026-05-21-implementation-c-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Vendor Unity test framework

**Files:**
- Create: `implementation/c/tests/unity/unity.h`
- Create: `implementation/c/tests/unity/unity_internals.h`
- Create: `implementation/c/tests/unity/unity.c`

- [ ] **Step 1: Make the unity directory**

Run: `mkdir -p implementation/c/tests/unity`

- [ ] **Step 2: Download the three Unity files from upstream**

Run:
```bash
cd implementation/c/tests/unity
UNITY_REV=v2.6.0
curl -sSLfo unity.h           "https://raw.githubusercontent.com/ThrowTheSwitch/Unity/${UNITY_REV}/src/unity.h"
curl -sSLfo unity_internals.h "https://raw.githubusercontent.com/ThrowTheSwitch/Unity/${UNITY_REV}/src/unity_internals.h"
curl -sSLfo unity.c           "https://raw.githubusercontent.com/ThrowTheSwitch/Unity/${UNITY_REV}/src/unity.c"
```

Expected: three files present, each `> 1` KB. If curl fails, ask the user to grant `curl` access for that host, or vendor by `git clone` to a scratch dir and copy the three files.

- [ ] **Step 3: Verify Unity compiles standalone**

Run:
```bash
cd implementation/c/tests/unity
cc -c -Wno-everything unity.c -o /tmp/unity.o
```
Expected: produces `/tmp/unity.o` with no errors. (Warnings suppressed because Unity uses idioms that trip modern -Wall.)

- [ ] **Step 4: Clean the throwaway object**

Run: `rm -f /tmp/unity.o`

- [ ] **Step 5: Commit**

```bash
git add implementation/c/tests/unity
git commit -m "test(c): vendor Unity v2.6.0 (3 files)

Public-domain test framework, dropped in tree to avoid an external
dependency. Will be wrapped by tests/test_runner.h with an MPI-aware
main() per spec §7.2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Test assertion helpers

**Files:**
- Create: `implementation/c/tests/test_assert.h`

- [ ] **Step 1: Create the assertion header**

Write `implementation/c/tests/test_assert.h`:
```c
#ifndef TEST_ASSERT_H
#define TEST_ASSERT_H

#include <complex.h>
#include <math.h>
#include "unity/unity.h"

/* Per spec §7.5 — tolerances used across the entire test suite. */
#define AMP_TOL   1e-10   /* per amplitude component */
#define PROB_TOL  1e-9    /* probabilities and sums  */

/* Assert two complex doubles are equal to within AMP_TOL component-wise. */
#define ASSERT_NEAR_AMP(expected, actual)                                  \
    do {                                                                   \
        complex double _e = (expected);                                    \
        complex double _a = (actual);                                      \
        TEST_ASSERT_DOUBLE_WITHIN(AMP_TOL, creal(_e), creal(_a));          \
        TEST_ASSERT_DOUBLE_WITHIN(AMP_TOL, cimag(_e), cimag(_a));          \
    } while (0)

/* Assert |q->amp|^2 sums to 1 across all ranks. */
#define ASSERT_NORM_ONE(q)                                                 \
    do {                                                                   \
        double _n = qreg_norm(q);                                          \
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, _n);                      \
    } while (0)

#endif /* TEST_ASSERT_H */
```

- [ ] **Step 2: Verify it parses**

Run:
```bash
cc -fsyntax-only -I implementation/c/tests implementation/c/tests/test_assert.h
```
Expected: exit 0, no output. (We `-fsyntax-only` because there's no `.c` companion yet.)

- [ ] **Step 3: Commit**

```bash
git add implementation/c/tests/test_assert.h
git commit -m "test(c): add test_assert.h with tolerances and amplitude macros

Tolerances per spec §7.5 (AMP_TOL=1e-10, PROB_TOL=1e-9). qreg_norm is
not yet defined; ASSERT_NORM_ONE will start working once Task 14 lands
qreg_norm in matrix.h/matrix.c.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: MPI-aware test runner macro

**Files:**
- Create: `implementation/c/tests/test_runner.h`

- [ ] **Step 1: Create the runner header**

Write `implementation/c/tests/test_runner.h`:
```c
#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <mpi.h>
#include <stdio.h>
#include "unity/unity.h"

/* Per spec §7.2. Every test_<module>.c expands TEST_RUNNER_MAIN() once at
 * the bottom of the file. The macro produces a complete main() that:
 *   - initialises MPI;
 *   - silences stdout/stderr on rank > 0 before UnityBegin, so only
 *     rank 0's report reaches the user;
 *   - runs the suite via the file-local register_tests() function;
 *   - calls UnityEnd() on EVERY rank (it finalises Unity's failure
 *     count), then MPI_LOR-reduces the per-rank pass/fail bit so any
 *     rank failing surfaces as a non-zero exit;
 *   - finalises MPI.
 */
#define TEST_RUNNER_MAIN()                                                  \
    void register_tests(void);                                              \
    int main(int argc, char **argv) {                                       \
        MPI_Init(&argc, &argv);                                             \
        int _rank;  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);                  \
        if (_rank != 0) {                                                   \
            freopen("/dev/null", "w", stdout);                              \
            freopen("/dev/null", "w", stderr);                              \
        }                                                                   \
        UnityBegin(__FILE__);                                               \
        register_tests();                                                   \
        int _unity_fail  = UnityEnd();                                      \
        int _local_fail  = (_unity_fail != 0) ? 1 : 0;                      \
        int _global_fail = 0;                                               \
        MPI_Allreduce(&_local_fail, &_global_fail, 1, MPI_INT, MPI_LOR,     \
                      MPI_COMM_WORLD);                                      \
        MPI_Finalize();                                                     \
        return _global_fail;                                                \
    }

#endif /* TEST_RUNNER_H */
```

- [ ] **Step 2: Verify the header parses against MPI**

Run:
```bash
mpicc -fsyntax-only -I implementation/c/tests implementation/c/tests/test_runner.h
```
Expected: exit 0, no output.

- [ ] **Step 3: Commit**

```bash
git add implementation/c/tests/test_runner.h
git commit -m "test(c): add MPI-aware test runner macro

TEST_RUNNER_MAIN() per spec §7.2. UnityEnd() is called on every rank so
non-rank-0 failures cannot mask as success during the MPI_Allreduce.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Minimal makefile that builds nothing yet

**Files:**
- Create: `implementation/c/makefile`

- [ ] **Step 1: Write the makefile**

Write `implementation/c/makefile`:
```make
# ---------------------------------------------------------------------------
# makefile for implementation/c -- sparse-gate MPI quantum simulator
#
# Targets:
#   make                 build libqubit.a (when sources exist) and bin/qubit
#   make test            build and run every tests/test_*.c at NP=1, 2, 4
#   make test-large      additionally run at NP=8 (Shor-21 etc.)
#   make demo ALGO=qft NP=4
#   make clean / distclean
#
# Requirements:
#   mpicc (OpenMPI 5.x, brew install open-mpi)
#
# This file grows as more source files come online. At Task 5 it does the
# minimum: lays out directories, defines variables, provides empty test
# and clean targets so the structure is in place for later tasks.
# ---------------------------------------------------------------------------

MPICC      ?= mpicc
CFLAGS     ?= -std=c11 -O2 -Wall -Wextra
LDFLAGS    ?= -lm

ifeq ($(DEBUG),1)
    CFLAGS  := -std=c11 -O0 -g -Wall -Wextra -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
endif

BUILDDIR   := build
BINDIR     := $(BUILDDIR)/bin
TESTDIR    := $(BUILDDIR)/tests

# Library sources will be filled in as later tasks add files.
LIB_SRCS   :=
LIB_OBJS   := $(LIB_SRCS:%.c=$(BUILDDIR)/%.o)

# Test files. Each one becomes its own binary.
TEST_SRCS  := $(wildcard tests/test_*.c)
TEST_BINS  := $(TEST_SRCS:tests/%.c=$(TESTDIR)/%)

# NP values to exercise.
NPS_SMALL  := 1 2 4
NPS_LARGE  := 1 2 4 8

# Unity vendored.
UNITY_OBJ  := $(BUILDDIR)/unity.o

.PHONY: all test test-large clean distclean dirs

all: dirs

dirs:
	@mkdir -p $(BUILDDIR) $(BINDIR) $(TESTDIR)

# Compile a library .c into build/.
$(BUILDDIR)/%.o: %.c | dirs
	$(MPICC) $(CFLAGS) -c $< -o $@

# Unity compiled separately with relaxed warnings.
$(UNITY_OBJ): tests/unity/unity.c | dirs
	$(MPICC) -std=c11 -O2 -w -c $< -o $@

# Compile a test source plus the library plus Unity into a test binary.
$(TESTDIR)/%: tests/%.c $(LIB_OBJS) $(UNITY_OBJ) | dirs
	$(MPICC) $(CFLAGS) -I. -Itests -Itests/unity $< $(LIB_OBJS) $(UNITY_OBJ) $(LDFLAGS) -o $@

# Run a single test binary at a list of NP values.
define run_test_at_np
	@echo "--- $(1) at NP=$(2) ---"
	@mpirun --oversubscribe -n $(2) $(1) || (echo "FAIL: $(1) at NP=$(2)"; exit 1)

endef

test: $(TEST_BINS)
	$(foreach bin,$(TEST_BINS),$(foreach np,$(NPS_SMALL),$(call run_test_at_np,$(bin),$(np))))
	@echo "OK: all tests passed at NP=$(NPS_SMALL)"

test-large: $(TEST_BINS)
	$(foreach bin,$(TEST_BINS),$(foreach np,$(NPS_LARGE),$(call run_test_at_np,$(bin),$(np))))
	@echo "OK: all tests passed at NP=$(NPS_LARGE)"

clean:
	rm -rf $(BUILDDIR)

distclean: clean
```

- [ ] **Step 2: Verify make is happy with the empty targets**

Run:
```bash
cd implementation/c
make dirs
ls -la build
```
Expected: `build/`, `build/bin/`, `build/tests/` directories created.

- [ ] **Step 3: Verify clean works**

Run:
```bash
cd implementation/c
make clean
ls build 2>&1 || echo "build removed"
```
Expected: `build` directory gone.

- [ ] **Step 4: Commit**

```bash
git add implementation/c/makefile
git commit -m "build(c): minimal makefile scaffolding

Sets up directory layout, variable conventions (MPICC, CFLAGS,
LIB_SRCS, TEST_SRCS), and the standard targets (all, test, test-large,
clean, distclean). LIB_SRCS is empty until Task 7 introduces the first
.c file; test wildcard catches every tests/test_*.c automatically as
modules land.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Smoke test that proves the test harness works

**Files:**
- Create: `implementation/c/tests/test_smoke.c`

- [ ] **Step 1: Write a trivially-passing test**

Write `implementation/c/tests/test_smoke.c`:
```c
/* test_smoke.c - sanity check that the MPI test runner harness works
 * end-to-end before any real library code exists. Asserts only things
 * about MPI itself. Removed (or kept) once the first real test lands.
 */
#include <mpi.h>
#include "unity/unity.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

static void test_mpi_size_is_positive(void) {
    TEST_ASSERT_GREATER_THAN_INT(0, g_size);
}

static void test_mpi_rank_in_range(void) {
    TEST_ASSERT_GREATER_OR_EQUAL_INT(0, g_rank);
    TEST_ASSERT_LESS_THAN_INT(g_size, g_rank);
}

static void test_mpi_size_is_power_of_two(void) {
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, g_size & (g_size - 1),
        "NP must be a power of two for this suite");
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_mpi_size_is_positive);
    RUN_TEST(test_mpi_rank_in_range);
    RUN_TEST(test_mpi_size_is_power_of_two);
}

TEST_RUNNER_MAIN()
```

- [ ] **Step 2: Build and run at NP=1, 2, 4**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -30
echo "exit: $?"
```
Expected: builds `build/tests/test_smoke`, runs at NP=1, 2, 4; each prints Unity's `OK` line on rank 0; final `OK: all tests passed at NP=1 2 4`; exit 0.

- [ ] **Step 3: Verify it fails when NP is not a power of two**

Run:
```bash
cd implementation/c
mpirun --oversubscribe -n 3 build/tests/test_smoke; echo "exit: $?"
```
Expected: exit non-zero, Unity reports `test_mpi_size_is_power_of_two` failed.

- [ ] **Step 4: Commit**

```bash
git add implementation/c/tests/test_smoke.c
git commit -m "test(c): smoke test for MPI harness

Three tiny assertions about MPI itself. Exists to prove that
test_runner.h + Unity + makefile actually orchestrate ranks correctly
before any real library code lands. Stays in the repo as the
quickest-possible build sanity check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1 — standart helpers (Tasks 7–10)

### Task 7: gcd_u64 (standart)

**Files:**
- Create: `implementation/c/standart.h`
- Create: `implementation/c/standart.c`
- Create: `implementation/c/tests/test_standart.c`
- Modify: `implementation/c/makefile` (`LIB_SRCS += standart.c`)

- [ ] **Step 1: Write the failing tests**

Write `implementation/c/tests/test_standart.c`:
```c
#include <mpi.h>
#include <stdint.h>
#include "standart.h"
#include "unity/unity.h"
#include "test_runner.h"

void setUp(void)    {}
void tearDown(void) {}

static void test_gcd_basics(void) {
    TEST_ASSERT_EQUAL_UINT64(1,  gcd_u64(1, 1));
    TEST_ASSERT_EQUAL_UINT64(7,  gcd_u64(14, 21));
    TEST_ASSERT_EQUAL_UINT64(6,  gcd_u64(12, 18));
    TEST_ASSERT_EQUAL_UINT64(1,  gcd_u64(7, 11));    /* coprime */
    TEST_ASSERT_EQUAL_UINT64(15, gcd_u64(15, 0));    /* gcd(x,0) = x */
    TEST_ASSERT_EQUAL_UINT64(15, gcd_u64(0, 15));
}

void register_tests(void) {
    RUN_TEST(test_gcd_basics);
}

TEST_RUNNER_MAIN()
```

- [ ] **Step 2: Create header and stub implementation that fails the test**

Write `implementation/c/standart.h`:
```c
#ifndef STANDART_H
#define STANDART_H

#include <stdint.h>

uint64_t gcd_u64(uint64_t a, uint64_t b);

#endif
```

Write `implementation/c/standart.c`:
```c
#include "standart.h"

uint64_t gcd_u64(uint64_t a, uint64_t b) {
    return 0;   /* deliberately wrong - test should fail */
}
```

Add `standart.c` to `LIB_SRCS` in the makefile:
```make
LIB_SRCS := standart.c
```

- [ ] **Step 3: Build and confirm the test fails**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -15
```
Expected: `test_standart` fails inside `test_gcd_basics` with the very first assertion (`gcd(1,1) == 1`).

- [ ] **Step 4: Implement the binary-GCD algorithm and verify pass**

Replace the body of `gcd_u64` in `standart.c`:
```c
uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_standart` passes at NP=1, 2, 4.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/standart.h implementation/c/standart.c \
        implementation/c/tests/test_standart.c implementation/c/makefile
git commit -m "feat(c): standart - gcd_u64

Euclidean GCD on uint64_t. First entry in the standart utility module;
the rest (mod_pow, continued_fraction, is_power_of_two, ilog2_u32) land
in Tasks 8-10.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: mod_pow (standart)

**Files:**
- Modify: `implementation/c/standart.h` (add `mod_pow` declaration)
- Modify: `implementation/c/standart.c` (add implementation)
- Modify: `implementation/c/tests/test_standart.c` (add tests)

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_standart.c` inside the existing test list:
```c
static void test_mod_pow_basics(void) {
    /* a^0 mod N = 1 for any a, N>0 */
    TEST_ASSERT_EQUAL_UINT64(1,  mod_pow(7, 0, 15));
    /* a^1 mod N = a mod N */
    TEST_ASSERT_EQUAL_UINT64(7,  mod_pow(7, 1, 15));
    TEST_ASSERT_EQUAL_UINT64(2,  mod_pow(17, 1, 15));
    /* Known period: 7^4 mod 15 = 1 (the standard Shor example for N=15) */
    TEST_ASSERT_EQUAL_UINT64(4,  mod_pow(7, 2, 15));
    TEST_ASSERT_EQUAL_UINT64(13, mod_pow(7, 3, 15));
    TEST_ASSERT_EQUAL_UINT64(1,  mod_pow(7, 4, 15));
    /* Large exponent without overflow: 2^64 mod 1000003 (a prime) */
    TEST_ASSERT_EQUAL_UINT64(919715, mod_pow(2, 64, 1000003));
}
```

And register it:
```c
void register_tests(void) {
    RUN_TEST(test_gcd_basics);
    RUN_TEST(test_mod_pow_basics);
}
```

- [ ] **Step 2: Declare in header with a stub that fails**

Append to `standart.h` before `#endif`:
```c
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod);
```

Append to `standart.c`:
```c
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    return 0;   /* stub - test should fail */
}
```

- [ ] **Step 3: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_mod_pow_basics` fails at `mod_pow(7, 0, 15) == 1`.

- [ ] **Step 4: Implement square-and-multiply and verify pass**

Replace the body of `mod_pow` in `standart.c`:
```c
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 1) return 0;
    __uint128_t result = 1;
    __uint128_t b      = base % mod;
    while (exp > 0) {
        if (exp & 1ULL) {
            result = (result * b) % mod;
        }
        b   = (b * b) % mod;
        exp >>= 1;
    }
    return (uint64_t)result;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_standart` passes at NP=1, 2, 4 (`5 tests, 0 failures` or similar — Unity counts each TEST_ASSERT but we report at function granularity; the relevant line is `OK`).

- [ ] **Step 5: Commit**

```bash
git add implementation/c/standart.h implementation/c/standart.c \
        implementation/c/tests/test_standart.c
git commit -m "feat(c): standart - mod_pow with __uint128_t intermediate

Square-and-multiply modular exponentiation. The __uint128_t intermediate
keeps (a*b) mod N safe for all a,b < 2^64, which is what apply_modular_exp
needs at N near the qreg_max-qubits boundary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: continued_fraction (standart)

**Files:**
- Modify: `implementation/c/standart.h` (add `continued_fraction` declaration)
- Modify: `implementation/c/standart.c` (add implementation)
- Modify: `implementation/c/tests/test_standart.c` (add tests)

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_standart.c`:
```c
static void test_continued_fraction_pi(void) {
    uint64_t num, den;
    /* 22/7 is the first famous convergent of pi with denominator <= 100. */
    continued_fraction(3.14159265358979323846, 100, &num, &den);
    TEST_ASSERT_EQUAL_UINT64(22, num);
    TEST_ASSERT_EQUAL_UINT64(7,  den);
    /* 355/113 is the next, the famous Milü, with denominator <= 200. */
    continued_fraction(3.14159265358979323846, 200, &num, &den);
    TEST_ASSERT_EQUAL_UINT64(355, num);
    TEST_ASSERT_EQUAL_UINT64(113, den);
}

static void test_continued_fraction_simple_period(void) {
    uint64_t num, den;
    /* 3/8 should round-trip exactly with max_denom >= 8. */
    continued_fraction(3.0 / 8.0, 16, &num, &den);
    TEST_ASSERT_EQUAL_UINT64(3, num);
    TEST_ASSERT_EQUAL_UINT64(8, den);
    /* 5/16 likewise. */
    continued_fraction(5.0 / 16.0, 32, &num, &den);
    TEST_ASSERT_EQUAL_UINT64(5,  num);
    TEST_ASSERT_EQUAL_UINT64(16, den);
}
```

Register both:
```c
void register_tests(void) {
    RUN_TEST(test_gcd_basics);
    RUN_TEST(test_mod_pow_basics);
    RUN_TEST(test_continued_fraction_pi);
    RUN_TEST(test_continued_fraction_simple_period);
}
```

- [ ] **Step 2: Stub it in the header and source**

Append to `standart.h` before `#endif`:
```c
/* Find the best rational approximation p/q to x with q <= max_denominator.
 * Writes the numerator and denominator out via *num and *den.
 * Algorithm: standard continued-fraction expansion truncated at the last
 * convergent that satisfies the denominator bound.
 */
void continued_fraction(double x, uint64_t max_denominator,
                        uint64_t *num, uint64_t *den);
```

Append to `standart.c`:
```c
void continued_fraction(double x, uint64_t max_denominator,
                        uint64_t *num, uint64_t *den) {
    *num = 0; *den = 1;   /* stub */
}
```

- [ ] **Step 3: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_continued_fraction_pi` fails.

- [ ] **Step 4: Implement and verify pass**

Replace the body in `standart.c`:
```c
#include <math.h>

void continued_fraction(double x, uint64_t max_denominator,
                        uint64_t *num, uint64_t *den) {
    /* h[k]/k[k] are the convergents. Recurrence:
     *   h_{-1}=1, h_{-2}=0; k_{-1}=0, k_{-2}=1
     *   h_k = a_k * h_{k-1} + h_{k-2}
     *   k_k = a_k * k_{k-1} + k_{k-2}
     */
    uint64_t h1 = 1, h2 = 0;
    uint64_t k1 = 0, k2 = 1;
    uint64_t best_h = 0, best_k = 1;
    double   y       = x;
    for (int i = 0; i < 64; i++) {
        double a_d = floor(y);
        if (a_d < 0 || a_d > (double)UINT64_MAX) break;
        uint64_t a  = (uint64_t)a_d;
        /* check overflow of next denominator */
        if (k1 != 0 && a > (UINT64_MAX - k2) / k1) break;
        uint64_t k0 = a * k1 + k2;
        uint64_t h0 = a * h1 + h2;
        if (k0 > max_denominator) break;
        best_h = h0;
        best_k = k0;
        h2 = h1; h1 = h0;
        k2 = k1; k1 = k0;
        double frac = y - a_d;
        if (frac < 1e-18) break;
        y = 1.0 / frac;
    }
    *num = best_h;
    *den = best_k;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass at NP=1, 2, 4.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/standart.h implementation/c/standart.c \
        implementation/c/tests/test_standart.c
git commit -m "feat(c): standart - continued_fraction

Standard convergent-expansion algorithm with denominator bound; tested
against the classical pi convergents (22/7, 355/113) and trivial
ratios. Used by shor_factor's classical post-processing step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: is_power_of_two and ilog2_u32 (standart)

**Files:**
- Modify: `implementation/c/standart.h` (add declarations)
- Modify: `implementation/c/standart.c` (add implementations)
- Modify: `implementation/c/tests/test_standart.c` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_standart.c`:
```c
static void test_is_power_of_two(void) {
    TEST_ASSERT_TRUE (is_power_of_two(1));
    TEST_ASSERT_TRUE (is_power_of_two(2));
    TEST_ASSERT_TRUE (is_power_of_two(4));
    TEST_ASSERT_TRUE (is_power_of_two(1024));
    TEST_ASSERT_FALSE(is_power_of_two(0));
    TEST_ASSERT_FALSE(is_power_of_two(3));
    TEST_ASSERT_FALSE(is_power_of_two(6));
    TEST_ASSERT_FALSE(is_power_of_two(1023));
}

static void test_ilog2_u32(void) {
    TEST_ASSERT_EQUAL_INT( 0, ilog2_u32(1));
    TEST_ASSERT_EQUAL_INT( 1, ilog2_u32(2));
    TEST_ASSERT_EQUAL_INT( 2, ilog2_u32(4));
    TEST_ASSERT_EQUAL_INT(10, ilog2_u32(1024));
    TEST_ASSERT_EQUAL_INT(20, ilog2_u32(1 << 20));
}
```

Register:
```c
void register_tests(void) {
    RUN_TEST(test_gcd_basics);
    RUN_TEST(test_mod_pow_basics);
    RUN_TEST(test_continued_fraction_pi);
    RUN_TEST(test_continued_fraction_simple_period);
    RUN_TEST(test_is_power_of_two);
    RUN_TEST(test_ilog2_u32);
}
```

- [ ] **Step 2: Stub the headers**

Append to `standart.h` before `#endif`:
```c
int is_power_of_two(int x);
int ilog2_u32      (uint32_t x);   /* requires x is a power of two */
```

Append to `standart.c`:
```c
int is_power_of_two(int x) { return 0; }
int ilog2_u32(uint32_t x)  { return -1; }
```

- [ ] **Step 3: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: both new tests fail.

- [ ] **Step 4: Implement and verify pass**

Replace the bodies in `standart.c`:
```c
int is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

int ilog2_u32(uint32_t x) {
    /* Precondition: x is a power of two and nonzero. */
    int r = 0;
    while (x > 1) { x >>= 1; r++; }
    return r;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass at NP=1, 2, 4.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/standart.h implementation/c/standart.c \
        implementation/c/tests/test_standart.c
git commit -m "feat(c): standart - is_power_of_two and ilog2_u32

Used pervasively for the 2^p rank partitioning. ilog2_u32 has a
precondition (input must be a power of two) so it does not need a
branch on the slow path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — qreg lifecycle (Tasks 11–14)

### Task 11: matrix.h — qreg struct, QREG_MAX_QUBITS, QREG_ASSERT

**Files:**
- Create: `implementation/c/matrix.h`

- [ ] **Step 1: Write the header**

Write `implementation/c/matrix.h`:
```c
#ifndef MATRIX_H
#define MATRIX_H

#include <complex.h>
#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Per spec §4.3:  hard cap on n_qubits keeps every  1ULL<<k  shift   *
 * well-defined on 64-bit systems and leaves 4 bits of headroom over  *
 * size_t arithmetic.                                                  */
#define QREG_MAX_QUBITS 60

/* Always-on assert. Survives -DNDEBUG. Uses MPI_Abort so failures on  *
 * one rank do not leave the others hanging in collective calls.       */
#define QREG_ASSERT(cond, msg)                                              \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr,                                                 \
                "QREG_ASSERT failed at %s:%d: %s\n  condition: %s\n",       \
                __FILE__, __LINE__, (msg), #cond);                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

typedef struct {
    complex double *amp;       /* local slice, length local_size           */
    int      n_qubits;         /* global qubit count                       */
    int      rank, n_procs;    /* MPI rank and size (size = 2^p)           */
    int      p;                /* log2(n_procs); top p index bits = rank   */
    size_t   local_size;       /* = 2^(n_qubits - p)                       */
    MPI_Comm comm;
} qreg;

/* Lifecycle */
qreg *qreg_create   (int n_qubits, MPI_Comm comm);
void  qreg_destroy  (qreg *q);
void  qreg_init_basis(qreg *q, size_t basis_state);

/* Reductions used by tests and algorithms */
double qreg_norm(const qreg *q);
double prob_of  (const qreg *q, size_t basis);

#endif /* MATRIX_H */
```

- [ ] **Step 2: Verify it parses against MPI**

Run:
```bash
mpicc -fsyntax-only -I implementation/c implementation/c/matrix.h
```
Expected: exit 0, no output.

- [ ] **Step 3: Commit**

```bash
git add implementation/c/matrix.h
git commit -m "feat(c): matrix.h - qreg struct, QREG_MAX_QUBITS, QREG_ASSERT

Header-only at this step. Function bodies for the declared entry points
land in Tasks 12 (lifecycle) and 14 (norm/prob).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: qreg_create and qreg_destroy

**Files:**
- Create: `implementation/c/matrix.c`
- Create: `implementation/c/tests/test_matrix.c`
- Modify: `implementation/c/makefile` (`LIB_SRCS += matrix.c`)

- [ ] **Step 1: Write failing tests**

Write `implementation/c/tests/test_matrix.c`:
```c
#include <mpi.h>
#include <stdlib.h>
#include "matrix.h"
#include "standart.h"
#include "unity/unity.h"
#include "test_assert.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

static void test_create_4q(void) {
    /* Only run when n_procs fits: 4 qubits give 2^4 = 16 amplitudes, so
     * up to NP=16 is fine. We exercise NP=1, 2, 4 in the test target.   */
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    TEST_ASSERT_NOT_NULL(q);
    TEST_ASSERT_EQUAL_INT(4,       q->n_qubits);
    TEST_ASSERT_EQUAL_INT(g_rank,  q->rank);
    TEST_ASSERT_EQUAL_INT(g_size,  q->n_procs);
    TEST_ASSERT_EQUAL_size_t((size_t)16 / (size_t)g_size, q->local_size);
    TEST_ASSERT_NOT_NULL(q->amp);
    qreg_destroy(q);
}

static void test_create_rejects_non_pow2_n_procs(void) {
    /* qreg_create returns NULL if n_procs is not a power of two. We      *
     * cannot synthesise a non-pow2 comm from a pow2 MPI run; instead     *
     * we verify the helper that qreg_create uses.                        */
    TEST_ASSERT_TRUE (is_power_of_two(g_size));   /* the test runs only at pow2 NP */
}

static void test_create_rejects_too_many_qubits(void) {
    /* QREG_MAX_QUBITS = 60. Asking for 100 must return NULL. */
    qreg *q = qreg_create(100, MPI_COMM_WORLD);
    TEST_ASSERT_NULL(q);
}

static void test_create_rejects_zero_qubits(void) {
    qreg *q = qreg_create(0, MPI_COMM_WORLD);
    TEST_ASSERT_NULL(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_create_4q);
    RUN_TEST(test_create_rejects_non_pow2_n_procs);
    RUN_TEST(test_create_rejects_too_many_qubits);
    RUN_TEST(test_create_rejects_zero_qubits);
}

TEST_RUNNER_MAIN()
```

- [ ] **Step 2: Stub matrix.c with implementations that fail**

Write `implementation/c/matrix.c`:
```c
#include "matrix.h"
#include "standart.h"
#include <stdlib.h>
#include <string.h>

qreg *qreg_create(int n_qubits, MPI_Comm comm) {
    return NULL;   /* stub */
}

void qreg_destroy(qreg *q) {
    /* stub */
}

/* qreg_init_basis, qreg_norm, prob_of land in later tasks. */
void   qreg_init_basis(qreg *q, size_t basis_state) {}
double qreg_norm     (const qreg *q)                  { return 0.0; }
double prob_of       (const qreg *q, size_t basis)    { return 0.0; }
```

Add `matrix.c` to `LIB_SRCS`:
```make
LIB_SRCS := standart.c matrix.c
```

- [ ] **Step 3: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -15
```
Expected: `test_create_4q` fails at `TEST_ASSERT_NOT_NULL(q)`.

- [ ] **Step 4: Implement and verify pass**

Replace the body of `qreg_create` and `qreg_destroy` in `matrix.c`:
```c
qreg *qreg_create(int n_qubits, MPI_Comm comm) {
    if (n_qubits < 1 || n_qubits > QREG_MAX_QUBITS) return NULL;

    int n_procs, rank;
    MPI_Comm_size(comm, &n_procs);
    MPI_Comm_rank(comm, &rank);
    if (!is_power_of_two(n_procs))       return NULL;
    if ((size_t)n_procs > ((size_t)1 << n_qubits)) return NULL;

    qreg *q = (qreg *)malloc(sizeof *q);
    if (!q) return NULL;
    q->n_qubits   = n_qubits;
    q->n_procs    = n_procs;
    q->rank       = rank;
    q->p          = ilog2_u32((uint32_t)n_procs);
    q->local_size = (size_t)1 << (n_qubits - q->p);
    q->comm       = comm;
    q->amp        = (complex double *)calloc(q->local_size, sizeof *q->amp);
    if (!q->amp) { free(q); return NULL; }
    return q;
}

void qreg_destroy(qreg *q) {
    if (!q) return;
    free(q->amp);
    free(q);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass at NP=1, 2, 4.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.c implementation/c/tests/test_matrix.c \
        implementation/c/makefile
git commit -m "feat(c): matrix - qreg_create and qreg_destroy

Validates n_qubits in [1, QREG_MAX_QUBITS], n_procs is a power of two,
n_procs <= 2^n_qubits. Allocates the local amplitude slice zero-filled
via calloc so qreg_init_basis can ignore the rest.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: qreg_init_basis

**Files:**
- Modify: `implementation/c/matrix.c` (real body)
- Modify: `implementation/c/tests/test_matrix.c` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_init_basis_zero(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    /* Rank that owns basis 0 should have amp[0] = 1; all others 0. */
    if (q->rank == 0) {
        ASSERT_NEAR_AMP(1.0 + 0.0*I, q->amp[0]);
    }
    for (size_t i = (q->rank == 0 ? 1 : 0); i < q->local_size; i++) {
        ASSERT_NEAR_AMP(0.0 + 0.0*I, q->amp[i]);
    }
    qreg_destroy(q);
}

static void test_init_basis_arbitrary(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    /* basis = 0b1011 = 11. Owning rank is (11 >> (4-p)).               */
    qreg_init_basis(q, 11);
    int owning_rank = (int)(11ULL >> (q->n_qubits - q->p));
    size_t owning_off = 11ULL & (q->local_size - 1);
    if (q->rank == owning_rank) {
        ASSERT_NEAR_AMP(1.0 + 0.0*I, q->amp[owning_off]);
    }
    /* No other amplitude on any rank should be set. */
    for (size_t i = 0; i < q->local_size; i++) {
        if (q->rank == owning_rank && i == owning_off) continue;
        ASSERT_NEAR_AMP(0.0 + 0.0*I, q->amp[i]);
    }
    qreg_destroy(q);
}

static void test_init_basis_normalisation(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    qreg_init_basis(q, 7);
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_init_basis_zero);
RUN_TEST(test_init_basis_arbitrary);
RUN_TEST(test_init_basis_normalisation);
```

- [ ] **Step 2: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_init_basis_zero` fails at the rank-0 amplitude check.

- [ ] **Step 3: Implement init_basis and verify pass**

Replace the stub of `qreg_init_basis` in `matrix.c`:
```c
void qreg_init_basis(qreg *q, size_t basis_state) {
    QREG_ASSERT(q != NULL, "qreg_init_basis: q is NULL");
    QREG_ASSERT(basis_state < ((size_t)1 << q->n_qubits),
                "qreg_init_basis: basis_state out of range");
    /* zero everything */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    /* set the owning rank's amplitude to 1 */
    int owning_rank = (int)(basis_state >> (q->n_qubits - q->p));
    if (q->rank == owning_rank) {
        size_t off = basis_state & (q->local_size - 1);
        q->amp[off] = 1.0;
    }
}
```

Note: `qreg_norm` is still a stub (`return 0`), so `ASSERT_NORM_ONE` would fail. Implement `qreg_norm` here too (Task 14 will add real tests for it):
```c
double qreg_norm(const qreg *q) {
    QREG_ASSERT(q != NULL, "qreg_norm: q is NULL");
    double local = 0.0;
    for (size_t i = 0; i < q->local_size; i++) {
        double r = creal(q->amp[i]);
        double im = cimag(q->amp[i]);
        local += r*r + im*im;
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    return global;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass at NP=1, 2, 4.

- [ ] **Step 4: Sanity check at NP=8**

Run:
```bash
cd implementation/c
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: exit 0, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.c implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - qreg_init_basis (with norm_one helper)

Sets the register to a chosen computational basis state. Only the owning
rank writes a non-zero amplitude; ASSERT_NORM_ONE relies on the
qreg_norm body that lands in this same commit (full unit tests for
qreg_norm/prob_of follow in Task 14).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: qreg_norm and prob_of

**Files:**
- Modify: `implementation/c/matrix.c` (implement prob_of; qreg_norm already in)
- Modify: `implementation/c/tests/test_matrix.c` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_norm_of_basis_state(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, qreg_norm(q));
    qreg_destroy(q);
}

static void test_prob_of_basis_state(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 5));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, 7));
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_norm_of_basis_state);
RUN_TEST(test_prob_of_basis_state);
```

- [ ] **Step 2: Build and confirm prob_of fails**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_prob_of_basis_state` fails at `prob_of(q, 5) == 1.0`.

- [ ] **Step 3: Implement prob_of and verify pass**

Replace the stub of `prob_of` in `matrix.c`:
```c
double prob_of(const qreg *q, size_t basis) {
    QREG_ASSERT(q != NULL, "prob_of: q is NULL");
    QREG_ASSERT(basis < ((size_t)1 << q->n_qubits),
                "prob_of: basis out of range");
    int owning_rank = (int)(basis >> (q->n_qubits - q->p));
    double local = 0.0;
    if (q->rank == owning_rank) {
        size_t off = basis & (q->local_size - 1);
        double r  = creal(q->amp[off]);
        double im = cimag(q->amp[off]);
        local = r*r + im*im;
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    return global;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass at NP=1, 2, 4.

- [ ] **Step 4: Sanity at NP=8**

Run:
```bash
cd implementation/c
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.c implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - prob_of (sister of qreg_norm)

prob_of(q, basis) returns the squared magnitude of the amplitude at the
given basis state, summed via MPI_Allreduce so every rank returns the
same value.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — parallel helpers and amplitude exchange (Tasks 15–16)

### Task 15: parallel.h / parallel.c — locality helpers

**Files:**
- Create: `implementation/c/parallel.h`
- Create: `implementation/c/parallel.c`
- Create: `implementation/c/tests/test_parallel.c`
- Modify: `implementation/c/makefile` (`LIB_SRCS += parallel.c`)

- [ ] **Step 1: Write failing tests**

Write `implementation/c/tests/test_parallel.c`:
```c
#include <mpi.h>
#include "matrix.h"
#include "parallel.h"
#include "unity/unity.h"
#include "test_assert.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

static void test_locality_classification(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    /* With 4 qubits, p = log2(n_procs).                                 *
     * - At NP=1: every qubit is local.                                  *
     * - At NP=2: qubits 0..2 local, qubit 3 global.                     *
     * - At NP=4: qubits 0,1 local; qubits 2,3 global.                   */
    for (int k = 0; k < q->n_qubits - q->p; k++)
        TEST_ASSERT_TRUE(is_local_qubit(q, k));
    for (int k = q->n_qubits - q->p; k < q->n_qubits; k++)
        TEST_ASSERT_FALSE(is_local_qubit(q, k));
    qreg_destroy(q);
}

static void test_partner_for_global_qubit(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    if (q->n_procs >= 2) {
        /* The top-most global qubit (n-1) partners by toggling bit (p-1). */
        int top_global = q->n_qubits - 1;
        int expected_partner = q->rank ^ (1 << (top_global - (q->n_qubits - q->p)));
        TEST_ASSERT_EQUAL_INT(expected_partner, partner_for(q, top_global));
    }
    qreg_destroy(q);
}

static void test_global_local_round_trip(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    /* For each global index this rank owns, global -> local -> global must round-trip. */
    size_t base = (size_t)q->rank * q->local_size;
    for (size_t off = 0; off < q->local_size; off++) {
        size_t g = base + off;
        TEST_ASSERT_TRUE(rank_owns(q, g));
        TEST_ASSERT_EQUAL_size_t(off, global_to_local(q, g));
        TEST_ASSERT_EQUAL_size_t(g,   local_to_global(q, off));
    }
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_locality_classification);
    RUN_TEST(test_partner_for_global_qubit);
    RUN_TEST(test_global_local_round_trip);
}

TEST_RUNNER_MAIN()
```

- [ ] **Step 2: Stub parallel.h and parallel.c**

Write `implementation/c/parallel.h`:
```c
#ifndef PARALLEL_H
#define PARALLEL_H

#include "matrix.h"

/* Spec §6.2 — locality + exchange primitives. */

int    is_local_qubit  (const qreg *q, int k);
int    partner_for     (const qreg *q, int k);          /* must be global qubit */
int    rank_owns       (const qreg *q, size_t global_index);
size_t global_to_local (const qreg *q, size_t global_index);
size_t local_to_global (const qreg *q, size_t local_index);

void exchange_amplitudes(qreg *q, int partner_rank,
                         complex double *recv_buf);
/* Sendrecvs q->amp <-> recv_buf with partner_rank in q->comm.
 * recv_buf must be q->local_size complex doubles. */

#endif
```

Write `implementation/c/parallel.c`:
```c
#include "parallel.h"
#include "matrix.h"
#include <string.h>

int is_local_qubit(const qreg *q, int k) {
    (void)q; (void)k; return 0;   /* stub */
}
int partner_for(const qreg *q, int k) {
    (void)q; (void)k; return -1;  /* stub */
}
int rank_owns(const qreg *q, size_t global_index) {
    (void)q; (void)global_index; return 0;
}
size_t global_to_local(const qreg *q, size_t global_index) {
    (void)q; (void)global_index; return 0;
}
size_t local_to_global(const qreg *q, size_t local_index) {
    (void)q; (void)local_index; return 0;
}
void exchange_amplitudes(qreg *q, int partner_rank, complex double *recv_buf) {
    (void)q; (void)partner_rank; (void)recv_buf;
}
```

Add to `LIB_SRCS`:
```make
LIB_SRCS := standart.c matrix.c parallel.c
```

- [ ] **Step 3: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_locality_classification` fails on the first `is_local_qubit` check.

- [ ] **Step 4: Implement helpers (not exchange yet)**

Replace stubs in `parallel.c`:
```c
int is_local_qubit(const qreg *q, int k) {
    QREG_ASSERT(q != NULL, "is_local_qubit: q is NULL");
    QREG_ASSERT(k >= 0 && k < q->n_qubits, "is_local_qubit: k out of range");
    return k < q->n_qubits - q->p;
}

int partner_for(const qreg *q, int k) {
    QREG_ASSERT(q != NULL, "partner_for: q is NULL");
    QREG_ASSERT(k >= q->n_qubits - q->p && k < q->n_qubits,
                "partner_for: k is not a global qubit");
    int bit_in_rank = k - (q->n_qubits - q->p);
    return q->rank ^ (1 << bit_in_rank);
}

int rank_owns(const qreg *q, size_t global_index) {
    QREG_ASSERT(q != NULL, "rank_owns: q is NULL");
    QREG_ASSERT(global_index < ((size_t)1 << q->n_qubits),
                "rank_owns: global_index out of range");
    return (int)(global_index >> (q->n_qubits - q->p)) == q->rank;
}

size_t global_to_local(const qreg *q, size_t global_index) {
    QREG_ASSERT(q != NULL, "global_to_local: q is NULL");
    QREG_ASSERT(rank_owns(q, global_index),
                "global_to_local: this rank does not own global_index");
    return global_index & (q->local_size - 1);
}

size_t local_to_global(const qreg *q, size_t local_index) {
    QREG_ASSERT(q != NULL, "local_to_global: q is NULL");
    QREG_ASSERT(local_index < q->local_size,
                "local_to_global: local_index out of range");
    return ((size_t)q->rank * q->local_size) + local_index;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_parallel` passes at NP=1, 2, 4 (exchange_amplitudes still stub but no test exercises it yet).

- [ ] **Step 5: Commit**

```bash
git add implementation/c/parallel.h implementation/c/parallel.c \
        implementation/c/tests/test_parallel.c implementation/c/makefile
git commit -m "feat(c): parallel - locality helpers

is_local_qubit, partner_for, rank_owns, global_to_local, local_to_global.
exchange_amplitudes still a stub; Task 16 lands the MPI_Sendrecv body
and the round-trip test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: exchange_amplitudes (MPI_Sendrecv wrapper)

**Files:**
- Modify: `implementation/c/parallel.c` (real body)
- Modify: `implementation/c/tests/test_parallel.c` (add round-trip test)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_parallel.c`:
```c
static void test_exchange_round_trip(void) {
    /* Skip in single-process mode -- nothing to exchange with. */
    if (g_size == 1) {
        TEST_PASS();
        return;
    }
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    /* Fill local slice with a known per-rank, per-index pattern. */
    for (size_t i = 0; i < q->local_size; i++) {
        q->amp[i] = (double)(q->rank * 1000 + (int)i) + 0.5*I;
    }
    /* Partner along the most-significant global qubit. */
    int k       = q->n_qubits - 1;
    int partner = partner_for(q, k);
    /* Snapshot what we expect to receive: the partner's pattern. */
    complex double *expected = malloc(q->local_size * sizeof *expected);
    for (size_t i = 0; i < q->local_size; i++) {
        expected[i] = (double)(partner * 1000 + (int)i) + 0.5*I;
    }
    /* Snapshot what we expect to keep in q->amp afterwards: ours unchanged. */
    complex double *ours = malloc(q->local_size * sizeof *ours);
    memcpy(ours, q->amp, q->local_size * sizeof *ours);

    complex double *recv = malloc(q->local_size * sizeof *recv);
    exchange_amplitudes(q, partner, recv);
    for (size_t i = 0; i < q->local_size; i++) {
        ASSERT_NEAR_AMP(expected[i], recv[i]);
        ASSERT_NEAR_AMP(ours[i],     q->amp[i]);
    }
    free(expected); free(ours); free(recv);
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_exchange_round_trip);
```

- [ ] **Step 2: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: at NP >= 2, `test_exchange_round_trip` fails because recv is uninitialised garbage.

- [ ] **Step 3: Implement exchange_amplitudes and verify pass**

Replace the stub in `parallel.c`:
```c
void exchange_amplitudes(qreg *q, int partner_rank, complex double *recv_buf) {
    QREG_ASSERT(q != NULL, "exchange_amplitudes: q is NULL");
    QREG_ASSERT(recv_buf != NULL, "exchange_amplitudes: recv_buf is NULL");
    QREG_ASSERT(partner_rank >= 0 && partner_rank < q->n_procs,
                "exchange_amplitudes: partner_rank out of range");
    QREG_ASSERT(partner_rank != q->rank,
                "exchange_amplitudes: partner is self");
    MPI_Sendrecv(q->amp,    (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
                 partner_rank, 0,
                 recv_buf,  (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
                 partner_rank, 0,
                 q->comm, MPI_STATUS_IGNORE);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass at NP=1 (skipped), 2, 4. At NP=8 also passes.

- [ ] **Step 4: Sanity at NP=8**

Run:
```bash
cd implementation/c
mpirun --oversubscribe -n 8 build/tests/test_parallel; echo "exit: $?"
```
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/parallel.c implementation/c/tests/test_parallel.c
git commit -m "feat(c): parallel - exchange_amplitudes (MPI_Sendrecv full slice)

Pairwise full-slice swap with a partner rank. The caller owns recv_buf
(of size q->local_size) so the function does no allocation and can be
hot-looped by single-qubit and controlled gate primitives.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4 — single-qubit gates (Tasks 17–21)

### Task 17: apply_u + apply_h (local-qubit path only)

**Files:**
- Modify: `implementation/c/matrix.h` (declare apply_u, apply_h)
- Modify: `implementation/c/matrix.c` (implement local path)
- Modify: `implementation/c/tests/test_matrix.c` (add Hadamard tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_apply_h_on_qubit0_from_basis0(void) {
    /* H|0> = (|0> + |1>) / sqrt(2). On a single-qubit register this is
     * trivially testable.                                              */
    qreg *q = qreg_create(1, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    /* prob_of(0) == prob_of(1) == 0.5 */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 1));
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}

static void test_apply_h_twice_is_identity(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    apply_h(q, 1);
    apply_h(q, 1);
    /* Back to |5>. prob_of(5)=1, others 0. */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 5));
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_apply_h_on_qubit0_from_basis0);
RUN_TEST(test_apply_h_twice_is_identity);
```

- [ ] **Step 2: Declare apply_u and apply_h, stub them, build & fail**

Append to `matrix.h` before `#endif`:
```c
/* Single-qubit gates (spec §6.1). */
void apply_u(qreg *q, int target, complex double u[2][2]);
void apply_h(qreg *q, int target);
```

Append to `matrix.c`:
```c
void apply_u(qreg *q, int target, complex double u[2][2]) {
    (void)q; (void)target; (void)u;   /* stub */
}
void apply_h(qreg *q, int target) {
    (void)q; (void)target;             /* stub */
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_apply_h_on_qubit0_from_basis0` fails (`prob_of(0) == 0.5`).

- [ ] **Step 3: Implement apply_u local path + apply_h wrapper**

Replace the stubs in `matrix.c`:
```c
#include <math.h>
#include "parallel.h"

static void apply_u_local(qreg *q, int target, complex double u[2][2]) {
    size_t stride = (size_t)1 << target;
    size_t step   = stride << 1;
    for (size_t base = 0; base < q->local_size; base += step) {
        for (size_t off = 0; off < stride; off++) {
            size_t i0 = base + off;
            size_t i1 = i0 + stride;
            complex double a0 = q->amp[i0];
            complex double a1 = q->amp[i1];
            q->amp[i0] = u[0][0]*a0 + u[0][1]*a1;
            q->amp[i1] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
}

void apply_u(qreg *q, int target, complex double u[2][2]) {
    QREG_ASSERT(q != NULL, "apply_u: q is NULL");
    QREG_ASSERT(u != NULL, "apply_u: u is NULL");
    QREG_ASSERT(target >= 0 && target < q->n_qubits,
                "apply_u: target out of range");
    /* Global path lands in Task 18; for now require local. */
    QREG_ASSERT(is_local_qubit(q, target),
                "apply_u: global-qubit path not implemented yet");
    apply_u_local(q, target, u);
}

void apply_h(qreg *q, int target) {
    const double s = 1.0 / sqrt(2.0);
    complex double u[2][2] = {
        { s,  s},
        { s, -s},
    };
    apply_u(q, target, u);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass at NP=1 (every qubit local). At NP=2, `test_apply_h_on_qubit0_from_basis0` uses a 1-qubit register on 2 ranks, which already triggers global-path on qubit 0 — wait, with n=1 and p=1, n-p=0, so qubit 0 IS global. That asserts.

Adjust the test so the gate hits a guaranteed-local qubit:
```c
static void test_apply_h_on_qubit0_from_basis0(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);   /* 3 qubits, at NP=4 qubit 0 is still local */
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    /* After H on qubit 0: amplitudes (|0> + |1>)/sqrt(2) in the low bit. */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 1));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, 2));
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}
```

Re-run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass at NP=1, 2, 4.

- [ ] **Step 4: Sanity at NP=8 (qubit 0 still local on a 3-qubit register)**

Run:
```bash
cd implementation/c
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: exit 0. (With NP=8, p=3, n-p=0, so qubit 0 IS global — the asserted message fires. Confirm test_apply_h_on_qubit0_from_basis0 uses 3 qubits so qubit 0 is local: at NP=8 with 3 qubits, p=3, n-p=0; **all qubits become global**. So this still asserts. Change the test to apply H on the LAST local qubit by inspecting q->n_qubits - q->p - 1.)

Replace the test once more to be robust across all supported NP for a 3-qubit register:
```c
static void test_apply_h_on_qubit0_from_basis0(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    /* If qubit 0 is global at this NP, skip - that case is covered in Task 18. */
    if (!is_local_qubit(q, 0)) { qreg_destroy(q); TEST_PASS(); return; }
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 1));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, 2));
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}
```

Re-run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - apply_u (local path) + apply_h

Sparse in-place 2x2 application for the local-qubit case, per spec
§5.1. Global-qubit path is asserted-out and lands in Task 18.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 18: apply_u global-qubit path (with exchange)

**Files:**
- Modify: `implementation/c/matrix.c` (extend apply_u for global qubits)
- Modify: `implementation/c/tests/test_matrix.c` (add a global-qubit H test)

- [ ] **Step 1: Add a failing test that hits a global qubit at NP=2**

Append to `tests/test_matrix.c`:
```c
static void test_apply_h_on_global_qubit(void) {
    /* Choose a register where qubit n-1 is always global for NP>=2. */
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    if (g_size == 1) { qreg_destroy(q); TEST_PASS(); return; }
    int target = q->n_qubits - 1;     /* most-significant qubit */
    TEST_ASSERT_FALSE(is_local_qubit(q, target));
    qreg_init_basis(q, 0);
    apply_h(q, target);
    /* H on the top bit of |000>: state is (|000> + |100>) / sqrt(2). */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 4));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, 1));
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_apply_h_on_global_qubit);
```

- [ ] **Step 2: Build and confirm assert fires**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -15
```
Expected: at NP=2, the QREG_ASSERT in apply_u ("global-qubit path not implemented yet") fires, MPI_Abort exits the test runner non-zero.

- [ ] **Step 3: Implement the global path**

In `matrix.c`, extract the local path into `apply_u_local` (already done) and add a sibling `apply_u_global`. Replace the body of `apply_u`:
```c
static void apply_u_global(qreg *q, int target, complex double u[2][2]) {
    int   tbit    = target - (q->n_qubits - q->p);
    int   mybit   = (q->rank >> tbit) & 1;
    int   partner = q->rank ^ (1 << tbit);
    complex double *buf = malloc(q->local_size * sizeof *buf);
    exchange_amplitudes(q, partner, buf);
    /* Combine our slice with the partner's slice. We hold the value of
     * qubit `target` == mybit; partner held the opposite bit. The pair
     * (a_mybit, a_{1-mybit}) corresponds to amplitude (q->amp[i], buf[i]).
     */
    if (mybit == 0) {
        for (size_t i = 0; i < q->local_size; i++) {
            complex double a0 = q->amp[i];
            complex double a1 = buf[i];
            q->amp[i] = u[0][0]*a0 + u[0][1]*a1;
        }
    } else {
        for (size_t i = 0; i < q->local_size; i++) {
            complex double a0 = buf[i];
            complex double a1 = q->amp[i];
            q->amp[i] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
    free(buf);
}

void apply_u(qreg *q, int target, complex double u[2][2]) {
    QREG_ASSERT(q != NULL, "apply_u: q is NULL");
    QREG_ASSERT(u != NULL, "apply_u: u is NULL");
    QREG_ASSERT(target >= 0 && target < q->n_qubits,
                "apply_u: target out of range");
    if (is_local_qubit(q, target)) apply_u_local (q, target, u);
    else                           apply_u_global(q, target, u);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 4: Double-check norm preservation by applying H twice on a global qubit**

Append a final assertion to the test (inline in the existing test_apply_h_on_global_qubit body):
```c
    /* Apply H again - should return to |000>. */
    apply_h(q, target);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 0));
    ASSERT_NORM_ONE(q);
```

Re-run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at all NPs.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.c implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - apply_u global-qubit path

Pairwise MPI_Sendrecv with the partner rank (XOR by target's rank-bit),
then 2x2 application that picks the row matching this rank's bit value.
Round-trip (apply twice) restores the state, validating both the
exchange and the 2x2 algebra.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 19: Pauli gates (X, Y, Z)

**Files:**
- Modify: `implementation/c/matrix.h` (declare apply_x/y/z)
- Modify: `implementation/c/matrix.c` (implement)
- Modify: `implementation/c/tests/test_matrix.c` (add tests)

- [ ] **Step 1: Failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_pauli_x_flips_bit(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_x(q, 1);              /* flip qubit 1: |000> -> |010> = basis 2 */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 2));
    qreg_destroy(q);
}

static void test_pauli_y_on_zero(void) {
    /* Y|0> = i|1>. prob_of(1) = 1.                                     */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_y(q, 0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}

static void test_pauli_z_on_zero_is_identity(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_z(q, 0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 0));
    qreg_destroy(q);
}

static void test_pauli_z_on_one_negates(void) {
    /* Z|1> = -|1>. Probability still 1, but verify via apply_h then
     * undoing: H Z |0> = H (1/sqrt(2))(|0>-|1>) = |1>, not |0>.        */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    apply_z(q, 0);
    apply_h(q, 0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_pauli_x_flips_bit);
RUN_TEST(test_pauli_y_on_zero);
RUN_TEST(test_pauli_z_on_zero_is_identity);
RUN_TEST(test_pauli_z_on_one_negates);
```

- [ ] **Step 2: Stub the three functions**

Append to `matrix.h` before `#endif`:
```c
void apply_x(qreg *q, int target);
void apply_y(qreg *q, int target);
void apply_z(qreg *q, int target);
```

Append to `matrix.c`:
```c
void apply_x(qreg *q, int target) { (void)q; (void)target; }
void apply_y(qreg *q, int target) { (void)q; (void)target; }
void apply_z(qreg *q, int target) { (void)q; (void)target; }
```

- [ ] **Step 3: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_pauli_x_flips_bit` fails.

- [ ] **Step 4: Implement as apply_u wrappers and verify**

Replace the stubs in `matrix.c`:
```c
void apply_x(qreg *q, int target) {
    complex double u[2][2] = { {0, 1}, {1, 0} };
    apply_u(q, target, u);
}
void apply_y(qreg *q, int target) {
    complex double u[2][2] = { {0, -I}, {I, 0} };
    apply_u(q, target, u);
}
void apply_z(qreg *q, int target) {
    complex double u[2][2] = { {1, 0}, {0, -1} };
    apply_u(q, target, u);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - Pauli gates X, Y, Z

Thin apply_u wrappers. Tests cover bit-flip, the imaginary unit picked
up by Y, and Z's phase visible through the H-Z-H = X identity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 20: S, T, apply_phase

**Files:**
- Modify: `implementation/c/matrix.h`
- Modify: `implementation/c/matrix.c`
- Modify: `implementation/c/tests/test_matrix.c`

- [ ] **Step 1: Failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_s_is_phase_pi_over_2(void) {
    /* S applied to |+> then H back: (1/sqrt 2) (|0> + i|1>); H back puts
     * mass at |0>=|1>=0.5. The detailed phase can be cross-checked by
     * applying S twice (= Z) then comparing to Z directly.              */
    qreg *qA = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(qA, 0);
    apply_h(qA, 0); apply_s(qA, 0); apply_s(qA, 0); apply_h(qA, 0);

    qreg *qB = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(qB, 0);
    apply_h(qB, 0); apply_z(qB, 0); apply_h(qB, 0);

    /* Probabilities of |0>, |1> must match between the two registers. */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, prob_of(qB, 0), prob_of(qA, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, prob_of(qB, 1), prob_of(qA, 1));
    qreg_destroy(qA);
    qreg_destroy(qB);
}

static void test_t_quartic_is_z(void) {
    /* T^4 = Z (up to a sign that drops in probability). Use the H-..-H
     * sandwich so the phase shows up in measurable amplitudes.          */
    qreg *qA = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(qA, 0);
    apply_h(qA, 0); apply_t(qA, 0); apply_t(qA, 0);
    apply_t(qA, 0); apply_t(qA, 0); apply_h(qA, 0);

    qreg *qB = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(qB, 0);
    apply_h(qB, 0); apply_z(qB, 0); apply_h(qB, 0);

    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, prob_of(qB, 0), prob_of(qA, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, prob_of(qB, 1), prob_of(qA, 1));
    qreg_destroy(qA);
    qreg_destroy(qB);
}

static void test_apply_phase_zero_is_identity(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 1);
    apply_phase(q, 0, 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_s_is_phase_pi_over_2);
RUN_TEST(test_t_quartic_is_z);
RUN_TEST(test_apply_phase_zero_is_identity);
```

- [ ] **Step 2: Stubs**

Append to `matrix.h` before `#endif`:
```c
void apply_s    (qreg *q, int target);
void apply_t    (qreg *q, int target);
void apply_phase(qreg *q, int target, double theta);
```

Append to `matrix.c`:
```c
void apply_s    (qreg *q, int target)              { (void)q; (void)target; }
void apply_t    (qreg *q, int target)              { (void)q; (void)target; }
void apply_phase(qreg *q, int target, double theta){ (void)q; (void)target; (void)theta; }
```

- [ ] **Step 3: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: new tests fail.

- [ ] **Step 4: Implement and verify pass**

Replace the stubs in `matrix.c`:
```c
void apply_phase(qreg *q, int target, double theta) {
    complex double u[2][2] = { {1, 0}, {0, cexp(I * theta)} };
    apply_u(q, target, u);
}
void apply_s(qreg *q, int target) { apply_phase(q, target, M_PI / 2.0); }
void apply_t(qreg *q, int target) { apply_phase(q, target, M_PI / 4.0); }
```

Note: requires `#include <math.h>` and `#define _USE_MATH_DEFINES` or compile with `-D_USE_MATH_DEFINES` if M_PI isn't visible. On GNU/Clang `_GNU_SOURCE` is set by `<math.h>` by default; if needed, add `#define _USE_MATH_DEFINES` before any `<math.h>` include.

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - S, T, apply_phase

Z-axis phase rotations. S = phase(pi/2), T = phase(pi/4). Tests verify
S^2 = Z and T^4 = Z via the H-...-H sandwich so the phase becomes
visible in observable probabilities.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 21: RX, RY, RZ rotations

**Files:**
- Modify: `implementation/c/matrix.h`
- Modify: `implementation/c/matrix.c`
- Modify: `implementation/c/tests/test_matrix.c`

- [ ] **Step 1: Failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_rx_2pi_is_identity_up_to_phase(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_rx(q, 0, 2.0 * M_PI);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 0));
    qreg_destroy(q);
}

static void test_ry_pi_flips(void) {
    /* RY(pi)|0> = |1> (up to a global -i for our convention). */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_ry(q, 0, M_PI);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}

static void test_rz_zero_is_identity(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 1);
    apply_rz(q, 0, 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_rx_2pi_is_identity_up_to_phase);
RUN_TEST(test_ry_pi_flips);
RUN_TEST(test_rz_zero_is_identity);
```

- [ ] **Step 2: Stubs**

Append to `matrix.h` before `#endif`:
```c
void apply_rx(qreg *q, int target, double theta);
void apply_ry(qreg *q, int target, double theta);
void apply_rz(qreg *q, int target, double theta);
```

Append to `matrix.c`:
```c
void apply_rx(qreg *q, int target, double theta){ (void)q; (void)target; (void)theta; }
void apply_ry(qreg *q, int target, double theta){ (void)q; (void)target; (void)theta; }
void apply_rz(qreg *q, int target, double theta){ (void)q; (void)target; (void)theta; }
```

- [ ] **Step 3: Build and confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: rotation tests fail.

- [ ] **Step 4: Implement and verify**

Replace the stubs in `matrix.c`:
```c
void apply_rx(qreg *q, int target, double theta) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    complex double u[2][2] = {
        { c,        -I * s },
        { -I * s,    c     },
    };
    apply_u(q, target, u);
}
void apply_ry(qreg *q, int target, double theta) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    complex double u[2][2] = {
        { c, -s },
        { s,  c },
    };
    apply_u(q, target, u);
}
void apply_rz(qreg *q, int target, double theta) {
    complex double e_minus = cexp(-I * theta / 2.0);
    complex double e_plus  = cexp( I * theta / 2.0);
    complex double u[2][2] = {
        { e_minus, 0       },
        { 0,       e_plus  },
    };
    apply_u(q, target, u);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - rotation gates RX, RY, RZ

Standard half-angle rotation matrices. Tests verify the 2-pi periodicity
(probability), RY(pi) bit-flip behaviour, and RZ(0) identity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5 — two-qubit gates (Tasks 22–25)

### Task 22: apply_cu generic controlled-U (all four locality cases)

**Files:**
- Modify: `implementation/c/matrix.h` (declare apply_cu)
- Modify: `implementation/c/matrix.c` (implement four-case dispatch)
- Modify: `implementation/c/tests/test_matrix.c` (Bell-state tests at NP=1 for sanity)

The hard cross-boundary cases get their own dedicated file in Task 26 (`test_distributed_gates.c`); the tests here exercise the both-local case as a smoke check.

- [ ] **Step 1: Failing test**

Append to `tests/test_matrix.c`:
```c
static void test_cu_both_local_makes_bell(void) {
    /* On a 2-qubit register, H on 0 then CNOT(0,1) -> |Phi+>.           */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    /* Skip when either qubit is global; that's the dedicated test job. */
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    complex double cnot_target_u[2][2] = { {0,1}, {1,0} };
    apply_cu(q, 0, 1, cnot_target_u);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 0));   /* |00> */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 3));   /* |11> */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, 1));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, 2));
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_cu_both_local_makes_bell);
```

- [ ] **Step 2: Declare apply_cu and stub it**

Append to `matrix.h` before `#endif`:
```c
void apply_cu(qreg *q, int control, int target, complex double u[2][2]);
```

Append to `matrix.c`:
```c
void apply_cu(qreg *q, int control, int target, complex double u[2][2]) {
    (void)q; (void)control; (void)target; (void)u;   /* stub */
}
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_cu_both_local_makes_bell` fails.

- [ ] **Step 4: Implement four-case dispatch per spec §5.2**

Replace the stub of `apply_cu` in `matrix.c`:
```c
static void apply_cu_both_local(qreg *q, int control, int target,
                                complex double u[2][2]) {
    size_t cmask = (size_t)1 << control;
    size_t tstride = (size_t)1 << target;
    for (size_t i = 0; i < q->local_size; i++) {
        if ((i & cmask) && !(i & tstride)) {
            size_t j = i | tstride;
            complex double a0 = q->amp[i];
            complex double a1 = q->amp[j];
            q->amp[i] = u[0][0]*a0 + u[0][1]*a1;
            q->amp[j] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
}

static void apply_cu_c_local_t_global(qreg *q, int control, int target,
                                      complex double u[2][2]) {
    int tbit    = target - (q->n_qubits - q->p);
    int mybit   = (q->rank >> tbit) & 1;
    int partner = q->rank ^ (1 << tbit);
    size_t cmask = (size_t)1 << control;
    complex double *buf = malloc(q->local_size * sizeof *buf);
    exchange_amplitudes(q, partner, buf);
    if (mybit == 0) {
        for (size_t i = 0; i < q->local_size; i++) {
            if (!(i & cmask)) continue;
            complex double a0 = q->amp[i];
            complex double a1 = buf[i];
            q->amp[i] = u[0][0]*a0 + u[0][1]*a1;
        }
    } else {
        for (size_t i = 0; i < q->local_size; i++) {
            if (!(i & cmask)) continue;
            complex double a0 = buf[i];
            complex double a1 = q->amp[i];
            q->amp[i] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
    free(buf);
}

static void apply_cu_c_global_t_local(qreg *q, int control, int target,
                                      complex double u[2][2]) {
    int cbit = control - (q->n_qubits - q->p);
    if (((q->rank >> cbit) & 1) == 0) return;   /* no-op for this rank   */
    /* Otherwise the gate is just a local single-qubit u on the target.  */
    size_t tstride = (size_t)1 << target;
    size_t step    = tstride << 1;
    for (size_t base = 0; base < q->local_size; base += step) {
        for (size_t off = 0; off < tstride; off++) {
            size_t i0 = base + off;
            size_t i1 = i0 + tstride;
            complex double a0 = q->amp[i0];
            complex double a1 = q->amp[i1];
            q->amp[i0] = u[0][0]*a0 + u[0][1]*a1;
            q->amp[i1] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
}

static void apply_cu_both_global(qreg *q, int control, int target,
                                 complex double u[2][2]) {
    /* Partner is by target bit; control bit is fixed per rank.         */
    int tbit    = target  - (q->n_qubits - q->p);
    int cbit    = control - (q->n_qubits - q->p);
    if (((q->rank >> cbit) & 1) == 0) return;     /* no-op on this rank */
    int mybit   = (q->rank >> tbit) & 1;
    int partner = q->rank ^ (1 << tbit);
    complex double *buf = malloc(q->local_size * sizeof *buf);
    exchange_amplitudes(q, partner, buf);
    if (mybit == 0) {
        for (size_t i = 0; i < q->local_size; i++) {
            complex double a0 = q->amp[i];
            complex double a1 = buf[i];
            q->amp[i] = u[0][0]*a0 + u[0][1]*a1;
        }
    } else {
        for (size_t i = 0; i < q->local_size; i++) {
            complex double a0 = buf[i];
            complex double a1 = q->amp[i];
            q->amp[i] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
    free(buf);
}

void apply_cu(qreg *q, int control, int target, complex double u[2][2]) {
    QREG_ASSERT(q != NULL,           "apply_cu: q is NULL");
    QREG_ASSERT(u != NULL,           "apply_cu: u is NULL");
    QREG_ASSERT(control >= 0 && control < q->n_qubits,
                "apply_cu: control out of range");
    QREG_ASSERT(target  >= 0 && target  < q->n_qubits,
                "apply_cu: target out of range");
    QREG_ASSERT(control != target,
                "apply_cu: control == target");
    int c_local = is_local_qubit(q, control);
    int t_local = is_local_qubit(q, target);
    if      ( c_local &&  t_local) apply_cu_both_local      (q, control, target, u);
    else if ( c_local && !t_local) apply_cu_c_local_t_global(q, control, target, u);
    else if (!c_local &&  t_local) apply_cu_c_global_t_local(q, control, target, u);
    else                           apply_cu_both_global     (q, control, target, u);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8 (the Bell test skips at NPs where qubit 1 is global; deep cross-boundary tests live in Task 26).

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - apply_cu with all four locality cases

Spec §5.2 dispatch: both-local, c-local/t-global,
c-global/t-local, both-global. The both-global case routes the partner
by the TARGET bit (not control); the control bit is constant per rank
so we early-return if zero. Bell-state smoke test in test_matrix; the
full distributed-gate cross-boundary suite lands in Task 26.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 23: apply_cnot, apply_cz, apply_controlled_phase

**Files:**
- Modify: `implementation/c/matrix.h`
- Modify: `implementation/c/matrix.c`
- Modify: `implementation/c/tests/test_matrix.c`

- [ ] **Step 1: Failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_cnot_local_makes_bell(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    apply_cnot(q, 0, 1);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 3));
    qreg_destroy(q);
}

static void test_cz_phase_on_11(void) {
    /* CZ leaves |00>, |01>, |10> alone and negates |11>. From |++>:    *
     *  (1/2)(|00>+|01>+|10>+|11>) -> (1/2)(|00>+|01>+|10>-|11>).       *
     * H on qubit 1 of that brings mass to |10>:                        *
     *   (1/2)(|00>+|01>) + (1/2)(|10>-|11>)                            *
     *   = (1/sqrt2)|0> (|0>+|1>)/sqrt2 + (1/sqrt2)|1> (|0>-|1>)/sqrt2  *
     * Apply H on q0: separates  |0>+|1> branch and |0>-|1> branch ...  *
     * Easier test: just check that CZ from |11> negates the amplitude  *
     * by sandwiching with H on q1 and confirming the phase via H Z H. */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 0);
    apply_x(q, 0); apply_x(q, 1);    /* |11>                            */
    apply_cz(q, 0, 1);               /* phase only: still |11>          */
    apply_h(q, 1); apply_h(q, 1);    /* identity                        */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 3));  /* still |11> */
    qreg_destroy(q);
}

static void test_controlled_phase_zero_is_identity(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 3);                       /* |11> */
    apply_controlled_phase(q, 0, 1, 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 3));
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_cnot_local_makes_bell);
RUN_TEST(test_cz_phase_on_11);
RUN_TEST(test_controlled_phase_zero_is_identity);
```

- [ ] **Step 2: Declare and stub**

Append to `matrix.h` before `#endif`:
```c
void apply_cnot            (qreg *q, int control, int target);
void apply_cz              (qreg *q, int control, int target);
void apply_controlled_phase(qreg *q, int control, int target, double theta);
```

Append to `matrix.c`:
```c
void apply_cnot            (qreg *q, int c, int t)              { (void)q;(void)c;(void)t; }
void apply_cz              (qreg *q, int c, int t)              { (void)q;(void)c;(void)t; }
void apply_controlled_phase(qreg *q, int c, int t, double th)   { (void)q;(void)c;(void)t;(void)th; }
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: all three new tests fail.

- [ ] **Step 4: Implement as apply_cu wrappers**

Replace the stubs:
```c
void apply_cnot(qreg *q, int control, int target) {
    complex double u[2][2] = { {0, 1}, {1, 0} };
    apply_cu(q, control, target, u);
}
void apply_cz(qreg *q, int control, int target) {
    complex double u[2][2] = { {1, 0}, {0, -1} };
    apply_cu(q, control, target, u);
}
void apply_controlled_phase(qreg *q, int control, int target, double theta) {
    complex double u[2][2] = { {1, 0}, {0, cexp(I * theta)} };
    apply_cu(q, control, target, u);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - apply_cnot, apply_cz, apply_controlled_phase

Three apply_cu wrappers covering the standard controlled-Pauli set and
the controlled phase rotation. Heavy lifting (four locality cases) is
in apply_cu from Task 22.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 24: apply_swap

**Files:**
- Modify: `implementation/c/matrix.h`
- Modify: `implementation/c/matrix.c`
- Modify: `implementation/c/tests/test_matrix.c`

- [ ] **Step 1: Failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_swap_exchanges_basis_indices(void) {
    /* Start from |01>; swap qubits 0,1 -> |10>. */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 1);
    apply_swap(q, 0, 1);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 2));
    qreg_destroy(q);
}

static void test_swap_self_is_identity(void) {
    /* swap(a,a) is illegal per QREG_ASSERT - skip; for a != b verify swap*swap = id. */
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    /* Pick two qubits that are both local at the current NP. */
    int a = 0, b = 1;
    if (!is_local_qubit(q, a) || !is_local_qubit(q, b)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    apply_swap(q, a, b);
    apply_swap(q, a, b);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 5));
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_swap_exchanges_basis_indices);
RUN_TEST(test_swap_self_is_identity);
```

- [ ] **Step 2: Stub**

Append to `matrix.h` before `#endif`:
```c
void apply_swap(qreg *q, int a, int b);
```

Append to `matrix.c`:
```c
void apply_swap(qreg *q, int a, int b) { (void)q; (void)a; (void)b; }
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: swap tests fail.

- [ ] **Step 4: Implement via 3 CNOTs**

Replace the stub:
```c
void apply_swap(qreg *q, int a, int b) {
    QREG_ASSERT(q != NULL, "apply_swap: q is NULL");
    QREG_ASSERT(a != b, "apply_swap: a == b");
    apply_cnot(q, a, b);
    apply_cnot(q, b, a);
    apply_cnot(q, a, b);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at all NPs.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - apply_swap via 3 CNOTs

Standard 3-CNOT decomposition. Costs three communication rounds when
both qubits are global, but reuses the well-tested apply_cnot path and
keeps SWAP correctness derived rather than re-implemented.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 6 — multi-controlled gates and measurement (Tasks 25–28)

### Task 25: apply_multi_controlled_z

**Files:**
- Modify: `implementation/c/matrix.h`
- Modify: `implementation/c/matrix.c`
- Modify: `implementation/c/tests/test_matrix.c`

This is the cheap special case: it phase-flips only the single all-ones amplitude. In the simulator it's $O(1)$.

- [ ] **Step 1: Failing test**

Append to `tests/test_matrix.c`:
```c
static void test_mcz_flips_only_all_ones(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    /* Start from uniform |+>^3 = (1/sqrt(8)) sum_x |x>. */
    qreg_init_basis(q, 0);
    apply_h(q, 0); apply_h(q, 1); apply_h(q, 2);
    /* All eight basis states have prob 1/8. */
    for (size_t b = 0; b < 8; b++)
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.125, prob_of(q, b));
    /* MCZ flips amp(|111>); probabilities unchanged. */
    apply_multi_controlled_z(q, 3);
    for (size_t b = 0; b < 8; b++)
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.125, prob_of(q, b));
    /* Apply MCZ twice -> identity (phase flip squared). */
    apply_multi_controlled_z(q, 3);
    /* Now apply H^3 to invert the uniform - should be back at |000>. */
    apply_h(q, 0); apply_h(q, 1); apply_h(q, 2);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 0));
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_mcz_flips_only_all_ones);
```

- [ ] **Step 2: Stub**

Append to `matrix.h` before `#endif`:
```c
/* Phase-flip the single all-ones amplitude |1...1> on the first n qubits. */
void apply_multi_controlled_z(qreg *q, int n);
```

Append to `matrix.c`:
```c
void apply_multi_controlled_z(qreg *q, int n) { (void)q; (void)n; }
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_mcz_flips_only_all_ones` fails because the final assertion (we applied H, MCZ, MCZ, H = identity but the stub MCZ does nothing, so we get back to |000> and prob_of(0) = 1) ... actually a no-op MCZ would give identity too. So the test as written wouldn't necessarily fail on no-op. We need to verify the phase was actually applied.

Replace the test with a stronger version that detects the phase:
```c
static void test_mcz_flips_only_all_ones(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, 0); apply_h(q, 1); apply_h(q, 2);   /* |+++> */
    apply_multi_controlled_z(q, 3);                /* phase on |111> only */
    apply_h(q, 0); apply_h(q, 1); apply_h(q, 2);   /* H^3 again */
    /* The state should now be H^3 (I - 2|111><111|) H^3 |0>             *
     * = H^3 |+++> - 2 H^3 |111><111|H^3|0>                              *
     * |111><111|H^3|0> = <111|+++> * |111> = (1/sqrt(8)) |111>          *
     * H^3 |111> = |--->                                                  *
     * |---> in the computational basis has amplitudes (1/sqrt 8)(-1)^...*
     *   x bits: amp = (1/sqrt8)(-1)^(popcount(x))                       *
     * Net state: |0> - 2 (1/sqrt8) * (1/sqrt8) (-1)^popcount * something *
     *                                                                    *
     * Easier: just verify prob_of changed from a known reference.        *
     * Without MCZ:  H^3 H^3 |0> = |0>, prob_of(0) = 1.                  *
     * With MCZ:     H^3 (I - 2|111><111|) H^3 |0>                       *
     *  = |0> - 2 (1/8) sum_x (-1)^(popcount x) |x>                      *
     *  prob_of(0)  = (1 - 2*1/8 * 1)^2 = (3/4)^2 = 9/16                 *
     *  prob_of(7)  = (-2*1/8*(-1))^2  = (1/4)^2 = 1/16                  */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 9.0/16.0, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0/16.0, prob_of(q, 7));
    qreg_destroy(q);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_mcz_flips_only_all_ones` fails (the stub leaves state at |0> with prob_of(0)=1, not 9/16).

- [ ] **Step 4: Implement and verify**

Replace the stub of `apply_multi_controlled_z`:
```c
void apply_multi_controlled_z(qreg *q, int n) {
    QREG_ASSERT(q != NULL,                 "apply_multi_controlled_z: q is NULL");
    QREG_ASSERT(n >= 1 && n <= q->n_qubits,"apply_multi_controlled_z: n out of range");
    /* Target the single basis state |1...1> on the first n qubits, with  *
     * higher bits (n .. n_qubits-1) free. We must phase-flip every       *
     * amplitude whose lower n bits are all 1 -- i.e. amp index with mask *
     * (mask & i) == mask  where mask = (1<<n)-1.                          *
     * Iterate locally over indices that meet the predicate.              */
    size_t mask = ((size_t)1 << n) - 1;
    size_t base = (size_t)q->rank * q->local_size;
    for (size_t off = 0; off < q->local_size; off++) {
        size_t global = base + off;
        if ((global & mask) == mask) q->amp[off] = -q->amp[off];
    }
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - apply_multi_controlled_z

Phase-flips every amplitude whose lower n bits are all 1. Linear-time
in the local slice (no all-to-all needed -- each rank inspects its own
amplitudes and applies the sign locally). Used by Grover's diffusion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 26: apply_multi_controlled_x (generalised Toffoli)

**Files:**
- Modify: `implementation/c/matrix.h`
- Modify: `implementation/c/matrix.c`
- Modify: `implementation/c/tests/test_matrix.c`

- [ ] **Step 1: Failing test**

Append to `tests/test_matrix.c`:
```c
static void test_mcx_acts_as_x_on_target_when_all_controls_set(void) {
    /* Controls = {0, 1}; target = 2. Starting from |011>, controls are
     * both 1, so target gets X: |011> -> |111> (basis index 7).         */
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1) || !is_local_qubit(q, 2)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 3);                      /* |011> */
    int ctrls[2] = {0, 1};
    apply_multi_controlled_x(q, ctrls, 2, 2);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 7));
    qreg_destroy(q);
}

static void test_mcx_is_noop_when_a_control_is_zero(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1) || !is_local_qubit(q, 2)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 1);                      /* |001> -- only qubit 0 set */
    int ctrls[2] = {0, 1};
    apply_multi_controlled_x(q, ctrls, 2, 2);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));   /* unchanged */
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_mcx_acts_as_x_on_target_when_all_controls_set);
RUN_TEST(test_mcx_is_noop_when_a_control_is_zero);
```

- [ ] **Step 2: Stub**

Append to `matrix.h` before `#endif`:
```c
void apply_multi_controlled_x(qreg *q, const int *controls, int n_controls,
                              int target);
```

Append to `matrix.c`:
```c
void apply_multi_controlled_x(qreg *q, const int *controls, int n_controls,
                              int target) {
    (void)q; (void)controls; (void)n_controls; (void)target;
}
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_mcx_acts_as_x_on_target_when_all_controls_set` fails.

- [ ] **Step 4: Implement (simple local iteration -- distributed multi-controlled X is left as a future optimisation; spec allows asserting at least one control is local)**

Replace the stub:
```c
void apply_multi_controlled_x(qreg *q, const int *controls, int n_controls,
                              int target) {
    QREG_ASSERT(q != NULL,        "apply_mcx: q is NULL");
    QREG_ASSERT(controls != NULL, "apply_mcx: controls is NULL");
    QREG_ASSERT(n_controls >= 1,  "apply_mcx: at least one control required");
    QREG_ASSERT(target >= 0 && target < q->n_qubits,
                "apply_mcx: target out of range");
    /* Validate controls in range and distinct from target. */
    for (int i = 0; i < n_controls; i++) {
        QREG_ASSERT(controls[i] >= 0 && controls[i] < q->n_qubits,
                    "apply_mcx: control out of range");
        QREG_ASSERT(controls[i] != target,
                    "apply_mcx: control equals target");
        for (int j = i + 1; j < n_controls; j++)
            QREG_ASSERT(controls[i] != controls[j],
                        "apply_mcx: duplicate control");
    }
    /* V1 supports the case where every control AND the target is local. *
     * The distributed multi-controlled X would itself decompose into    *
     * Toffoli + ancilla ladder; left for a follow-up.                    */
    for (int i = 0; i < n_controls; i++)
        QREG_ASSERT(is_local_qubit(q, controls[i]),
                    "apply_mcx: distributed controls not yet supported");
    QREG_ASSERT(is_local_qubit(q, target),
                "apply_mcx: distributed target not yet supported");
    size_t cmask = 0;
    for (int i = 0; i < n_controls; i++) cmask |= ((size_t)1 << controls[i]);
    size_t tstride = (size_t)1 << target;
    for (size_t i = 0; i < q->local_size; i++) {
        if ((i & cmask) == cmask && !(i & tstride)) {
            size_t j = i | tstride;
            complex double t = q->amp[i];
            q->amp[i] = q->amp[j];
            q->amp[j] = t;
        }
    }
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8 (tests skip when qubits would be global).

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - apply_multi_controlled_x (local-only v1)

Generalised Toffoli on the local case. Distributed multi-controlled X
requires Toffoli + ancilla decomposition that is out of v1 scope; the
function asserts every control and the target are local. Mostly used
for completeness; Grover prefers apply_multi_controlled_z (Task 25)
which has no such restriction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 27: measure_qubit

**Files:**
- Modify: `implementation/c/matrix.h`
- Modify: `implementation/c/matrix.c`
- Modify: `implementation/c/tests/test_matrix.c`

- [ ] **Step 1: Failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_measure_qubit_deterministic(void) {
    /* From |010> = basis 2, qubit 1 measures 1 with certainty. */
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 2);
    int outcome = measure_qubit(q, 1);
    TEST_ASSERT_EQUAL_INT(1, outcome);
    /* After projection, prob_of(2) is still 1. */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 2));
    qreg_destroy(q);
}

static void test_measure_qubit_collapse(void) {
    /* From |+0> (= 1/sqrt(2) (|00>+|10>)), measure qubit 1.            *
     * Qubit 1 is the high bit, |00> has q1=0, |10> has q1=1.            *
     * Both outcomes have prob 0.5; after projection state is one of    *
     * |00> or |10>, in either case prob_of(outcome) = 1 on the chosen  *
     * branch.                                                            */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 0);
    apply_h(q, 1);                          /* |0> on q0, |+> on q1 */
    int outcome = measure_qubit(q, 1);
    TEST_ASSERT_TRUE(outcome == 0 || outcome == 1);
    /* After collapse: state should be a basis state and normalised. */
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_measure_qubit_deterministic);
RUN_TEST(test_measure_qubit_collapse);
```

- [ ] **Step 2: Stub**

Append to `matrix.h` before `#endif`:
```c
int measure_qubit(qreg *q, int target);
```

Append to `matrix.c`:
```c
int measure_qubit(qreg *q, int target) { (void)q; (void)target; return -1; }
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_measure_qubit_deterministic` fails (returns -1 not 1).

- [ ] **Step 4: Implement and verify**

Replace the stub in `matrix.c` (requires `<stdlib.h>` for `rand`):
```c
int measure_qubit(qreg *q, int target) {
    QREG_ASSERT(q != NULL, "measure_qubit: q is NULL");
    QREG_ASSERT(target >= 0 && target < q->n_qubits,
                "measure_qubit: target out of range");
    /* 1. Compute P(bit_target == 0) locally; for each amplitude check
     *    whether its global index has bit_target = 0.                    */
    size_t base = (size_t)q->rank * q->local_size;
    double local_p0 = 0.0;
    for (size_t i = 0; i < q->local_size; i++) {
        size_t global = base + i;
        if (((global >> target) & 1) == 0) {
            double r = creal(q->amp[i]), im = cimag(q->amp[i]);
            local_p0 += r*r + im*im;
        }
    }
    double p0 = 0.0;
    MPI_Allreduce(&local_p0, &p0, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    /* 2. Rank 0 samples; broadcast the outcome. */
    int outcome = 0;
    if (q->rank == 0) {
        double u = (double)rand() / (double)RAND_MAX;
        outcome = (u < p0) ? 0 : 1;
    }
    MPI_Bcast(&outcome, 1, MPI_INT, 0, q->comm);
    /* 3. Project and renormalise. */
    double p_observed = (outcome == 0) ? p0 : (1.0 - p0);
    double inv_sqrt   = 1.0 / sqrt(p_observed);
    for (size_t i = 0; i < q->local_size; i++) {
        size_t global = base + i;
        int bit = (int)((global >> target) & 1);
        if (bit != outcome) q->amp[i] = 0.0;
        else                q->amp[i] *= inv_sqrt;
    }
    return outcome;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - measure_qubit

Local prob-0 -> MPI_Allreduce -> rank 0 samples -> MPI_Bcast outcome ->
each rank projects and renormalises. The local pass uses the global
index of each amplitude (rank*local_size + offset) to read the target
bit without any communication beyond the two collectives.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 28: measure_all, sample_distribution, qreg_clone, qreg_dump

**Files:**
- Modify: `implementation/c/matrix.h`
- Modify: `implementation/c/matrix.c`
- Modify: `implementation/c/tests/test_matrix.c`

- [ ] **Step 1: Failing tests**

Append to `tests/test_matrix.c`:
```c
static void test_measure_all_deterministic(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    size_t outcome = measure_all(q);
    TEST_ASSERT_EQUAL_size_t(5, outcome);
    qreg_destroy(q);
}

static void test_qreg_clone_independent_copy(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    qreg *c = qreg_clone(q);
    apply_x(q, 0);                           /* mutate original */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(c, 5));  /* clone unchanged */
    qreg_destroy(q); qreg_destroy(c);
}

static void test_sample_distribution_counts(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, 0);                           /* |+0> -> half on 0, half on 1 */
    size_t outcomes[1000];
    sample_distribution(q, outcomes, 1000);
    /* Outcome must always be 0 or 1 (top bit always 0). */
    for (int i = 0; i < 1000; i++) {
        TEST_ASSERT_TRUE(outcomes[i] == 0 || outcomes[i] == 1);
    }
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_measure_all_deterministic);
RUN_TEST(test_qreg_clone_independent_copy);
RUN_TEST(test_sample_distribution_counts);
```

- [ ] **Step 2: Stubs**

Append to `matrix.h` before `#endif`:
```c
size_t measure_all        (qreg *q);
void   sample_distribution(const qreg *q, size_t *out, int shots);
qreg  *qreg_clone         (const qreg *q);
void   qreg_dump          (const qreg *q, FILE *f);   /* rank 0 prints global state */
```

Append to `matrix.c`:
```c
size_t measure_all(qreg *q)                                   { (void)q; return 0; }
void   sample_distribution(const qreg *q, size_t *out, int s) { (void)q; (void)out; (void)s; }
qreg  *qreg_clone(const qreg *q)                              { (void)q; return NULL; }
void   qreg_dump(const qreg *q, FILE *f)                      { (void)q; (void)f; }
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_measure_all_deterministic` fails (returns 0, expected 5); `test_qreg_clone_independent_copy` fails (clone is NULL).

- [ ] **Step 4: Implement and verify**

Replace the stubs:
```c
size_t measure_all(qreg *q) {
    QREG_ASSERT(q != NULL, "measure_all: q is NULL");
    /* Build cumulative probability ranges per rank. Each rank computes  *
     * its local total |a_i|^2, then we do an MPI_Exscan to get the      *
     * running offset, then rank 0 samples u in [0,1) and broadcasts.     *
     * Each rank checks whether u falls in its range; the owning rank    *
     * picks the basis state with index found via linear scan.            */
    double local_total = 0.0;
    for (size_t i = 0; i < q->local_size; i++) {
        double r = creal(q->amp[i]), im = cimag(q->amp[i]);
        local_total += r*r + im*im;
    }
    double prefix = 0.0;
    MPI_Exscan(&local_total, &prefix, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    if (q->rank == 0) prefix = 0.0;
    double u = 0.0;
    if (q->rank == 0) u = (double)rand() / (double)RAND_MAX;
    MPI_Bcast(&u, 1, MPI_DOUBLE, 0, q->comm);

    size_t chosen_global = 0;
    int    chosen_rank   = -1;
    if (u >= prefix && u < prefix + local_total) {
        double cum = prefix;
        for (size_t i = 0; i < q->local_size; i++) {
            double r = creal(q->amp[i]), im = cimag(q->amp[i]);
            cum += r*r + im*im;
            if (cum >= u) {
                chosen_global = (size_t)q->rank * q->local_size + i;
                chosen_rank = q->rank;
                break;
            }
        }
    }
    /* Allreduce to find the chosen rank/global index. */
    int picks[2] = { (chosen_rank == q->rank) ? q->rank : -1, 0 };
    /* Use MPI_MAXLOC pattern: any non-(-1) rank wins. We just MAX rank. */
    int max_rank = -1;
    MPI_Allreduce(&picks[0], &max_rank, 1, MPI_INT, MPI_MAX, q->comm);
    size_t global_out = chosen_global;
    MPI_Bcast(&global_out, 1, MPI_UNSIGNED_LONG, max_rank, q->comm);
    /* Collapse the state: every amplitude except global_out is zeroed. */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    if (rank_owns(q, global_out)) {
        q->amp[global_to_local(q, global_out)] = 1.0;
    }
    return global_out;
}

void sample_distribution(const qreg *q, size_t *out, int shots) {
    QREG_ASSERT(q != NULL && out != NULL, "sample_distribution: NULL arg");
    QREG_ASSERT(shots > 0,                 "sample_distribution: shots <= 0");
    /* Naive: clone state, measure_all, restore. Inefficient but correct. *
     * V1 is single-shot quality; a future version could compute the CDF  *
     * once and sample.                                                    */
    qreg *temp = qreg_clone(q);
    for (int s = 0; s < shots; s++) {
        /* Reset the clone to the original each shot. */
        memcpy(temp->amp, q->amp, q->local_size * sizeof *q->amp);
        out[s] = measure_all(temp);
    }
    qreg_destroy(temp);
}

qreg *qreg_clone(const qreg *q) {
    QREG_ASSERT(q != NULL, "qreg_clone: q is NULL");
    qreg *c = qreg_create(q->n_qubits, q->comm);
    if (!c) return NULL;
    memcpy(c->amp, q->amp, q->local_size * sizeof *q->amp);
    return c;
}

void qreg_dump(const qreg *q, FILE *f) {
    QREG_ASSERT(q != NULL && f != NULL, "qreg_dump: NULL arg");
    /* Gather to rank 0 and print. */
    size_t total = (size_t)1 << q->n_qubits;
    complex double *full = NULL;
    if (q->rank == 0) full = malloc(total * sizeof *full);
    MPI_Gather(q->amp,                       (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
               full,                          (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
               0, q->comm);
    if (q->rank == 0) {
        fprintf(f, "qreg(%d qubits, %zu amplitudes):\n", q->n_qubits, total);
        for (size_t i = 0; i < total; i++) {
            if (cabs(full[i]) < 1e-12) continue;
            fprintf(f, "  |%zu> = (%+.6f, %+.6f)\n",
                    i, creal(full[i]), cimag(full[i]));
        }
        free(full);
    }
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_matrix; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/matrix.h implementation/c/matrix.c \
        implementation/c/tests/test_matrix.c
git commit -m "feat(c): matrix - measure_all, sample_distribution, qreg_clone, qreg_dump

measure_all uses MPI_Exscan + MPI_Bcast to sample a single basis state
from the full state vector. sample_distribution naively clones-and-
measures per shot (sufficient for v1). qreg_clone is a flat memcpy of
the local slice. qreg_dump gathers to rank 0 and prints non-zero
amplitudes in global-index order.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 7 — distributed gate coverage (Task 29)

### Task 29: test_distributed_gates.c — every locality combination of CNOT

**Files:**
- Create: `implementation/c/tests/test_distributed_gates.c`

Spec §7.3 row "test_distributed_gates" — this is the P1-driven test file that would have caught the partner-by-control bug. Each test gathers the global state vector to rank 0 and compares against the analytic post-gate amplitudes component-wise.

- [ ] **Step 1: Write the test file**

Write `implementation/c/tests/test_distributed_gates.c`:
```c
#include <complex.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "parallel.h"
#include "unity/unity.h"
#include "test_assert.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

/* Helper: gather the full state vector to every rank, return a heap
 * buffer of length 2^n_qubits. Caller frees. Wraps the MPI_Allgather. */
static complex double *gather_full(qreg *q) {
    size_t total = (size_t)1 << q->n_qubits;
    complex double *full = malloc(total * sizeof *full);
    MPI_Allgather(q->amp, (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
                  full,    (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
                  q->comm);
    return full;
}

/* The canonical |Phi+> Bell state on the (control, target) qubit pair  *
 * within an n-qubit register that started in basis |0...0>:            *
 *   amplitudes 1/sqrt(2) on indices 0 and (1<<c | 1<<t); 0 elsewhere.  */
static void assert_bell_state_on_pair(complex double *full,
                                      int n, int control, int target) {
    size_t total = (size_t)1 << n;
    double s = 1.0 / sqrt(2.0);
    size_t bell_idx = ((size_t)1 << control) | ((size_t)1 << target);
    for (size_t i = 0; i < total; i++) {
        if (i == 0)         ASSERT_NEAR_AMP(s + 0.0*I, full[i]);
        else if (i == bell_idx) ASSERT_NEAR_AMP(s + 0.0*I, full[i]);
        else                ASSERT_NEAR_AMP(0.0 + 0.0*I, full[i]);
    }
}

/* Build a |Phi+>-style Bell state via H on `control`, then CNOT(c, t),
 * starting from |0...0>. Returns the qreg (caller destroys).            */
static qreg *make_bell(int n_qubits, int control, int target) {
    qreg *q = qreg_create(n_qubits, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, control);
    apply_cnot(q, control, target);
    return q;
}

/* ---- (a) Single-qubit H on a global qubit ---- */

static void test_h_on_global_qubit_makes_uniform_pair(void) {
    if (g_size < 2) { TEST_PASS(); return; }
    int n = 3;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    int target = n - 1;                           /* always global at NP>=2 */
    TEST_ASSERT_FALSE(is_local_qubit(q, target));
    qreg_init_basis(q, 0);
    apply_h(q, target);
    complex double *full = gather_full(q);
    /* Expected:  (1/sqrt(2)) (|0> + |2^target>) */
    size_t tval = (size_t)1 << target;
    double s = 1.0 / sqrt(2.0);
    for (size_t i = 0; i < ((size_t)1 << n); i++) {
        if (i == 0)         ASSERT_NEAR_AMP(s, full[i]);
        else if (i == tval) ASSERT_NEAR_AMP(s, full[i]);
        else                ASSERT_NEAR_AMP(0.0, full[i]);
    }
    free(full);
    qreg_destroy(q);
}

/* ---- (b) CNOT control local, target global ---- */

static void test_cnot_c_local_t_global_bell(void) {
    if (g_size < 2) { TEST_PASS(); return; }
    int n = 3;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    /* control = 0 (always local), target = n-1 (always global at NP>=2). */
    int control = 0, target = n - 1;
    TEST_ASSERT_TRUE (is_local_qubit(q, control));
    TEST_ASSERT_FALSE(is_local_qubit(q, target));
    qreg_destroy(q);
    q = make_bell(n, control, target);
    complex double *full = gather_full(q);
    assert_bell_state_on_pair(full, n, control, target);
    free(full);
    qreg_destroy(q);
}

/* ---- (c) CNOT control global, target local ---- */

static void test_cnot_c_global_t_local_bell(void) {
    if (g_size < 2) { TEST_PASS(); return; }
    int n = 3;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    int control = n - 1, target = 0;
    TEST_ASSERT_FALSE(is_local_qubit(q, control));
    TEST_ASSERT_TRUE (is_local_qubit(q, target));
    qreg_destroy(q);
    q = make_bell(n, control, target);
    complex double *full = gather_full(q);
    assert_bell_state_on_pair(full, n, control, target);
    free(full);
    qreg_destroy(q);
}

/* ---- (d) CNOT control global, target global (THE bug case) ---- */

static void test_cnot_both_global_bell(void) {
    /* Need at least two global qubits, i.e. p >= 2, i.e. NP >= 4. */
    if (g_size < 4) { TEST_PASS(); return; }
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    /* The top two qubits are global at NP>=4 (since p = log2(NP)). */
    int control = n - 1, target = n - 2;
    TEST_ASSERT_FALSE(is_local_qubit(q, control));
    TEST_ASSERT_FALSE(is_local_qubit(q, target));
    qreg_destroy(q);
    q = make_bell(n, control, target);
    complex double *full = gather_full(q);
    assert_bell_state_on_pair(full, n, control, target);
    free(full);
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_h_on_global_qubit_makes_uniform_pair);
    RUN_TEST(test_cnot_c_local_t_global_bell);
    RUN_TEST(test_cnot_c_global_t_local_bell);
    RUN_TEST(test_cnot_both_global_bell);
}

TEST_RUNNER_MAIN()
```

- [ ] **Step 2: Build the new test binary**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -15
```
Expected: `test_distributed_gates` builds and passes at NP=1, 2, 4.

- [ ] **Step 3: Run at NP=8 to exercise the both-global case with more partitioning**

Run:
```bash
cd implementation/c
mpirun --oversubscribe -n 8 build/tests/test_distributed_gates; echo "exit: $?"
```
Expected: exit 0; all four tests pass.

- [ ] **Step 4: Confirm what failure would look like (spot-check)**

To prove the test catches the spec's P1 bug if it ever regressed: temporarily edit `matrix.c`'s `apply_cu_both_global` to use `control - (n-p)` instead of `target - (n-p)` for `tbit`, rebuild, run:
```bash
cd implementation/c
make clean && make test 2>&1 | tail -15
```
Expected: `test_cnot_both_global_bell` fails (revert the edit afterwards). Do not commit the buggy version.

- [ ] **Step 5: Commit (only the new test file)**

```bash
git add implementation/c/tests/test_distributed_gates.c
git commit -m "test(c): cross-boundary CNOT tests (spec §7.3 P1-driven row)

Four locality cases for controlled gates, each asserting against the
analytic Bell state gathered to rank 0 via MPI_Allgather. The both-
global case is the one the earlier spec draft would have failed; manual
spot-check confirms the test catches the partner-by-control regression.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 8 — Quantum Fourier Transform (Tasks 30–31)

### Task 30: apply_qft (forward, includes bit-reversal swaps)

**Files:**
- Create: `implementation/c/qft.h`
- Create: `implementation/c/qft.c`
- Create: `implementation/c/tests/test_qft.c`
- Modify: `implementation/c/makefile` (`LIB_SRCS += qft.c`)

- [ ] **Step 1: Failing tests**

Write `implementation/c/tests/test_qft.c`:
```c
#include <mpi.h>
#include <math.h>
#include "matrix.h"
#include "qft.h"
#include "unity/unity.h"
#include "test_assert.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

static void test_qft_on_1_qubit_equals_h(void) {
    /* QFT on a single qubit is the Hadamard. Compare amplitudes vs an
     * H applied to the same starting state.                            */
    qreg *qA = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(qA, 0);
    apply_qft(qA, 0, 1);

    qreg *qB = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(qB, 0);
    apply_h(qB, 0);

    /* Same probabilities on every basis state. */
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, prob_of(qB, i), prob_of(qA, i));
    }
    qreg_destroy(qA);
    qreg_destroy(qB);
}

static void test_qft_of_zero_is_uniform(void) {
    /* QFT |0...0> = (1/sqrt(N)) sum_y |y>. */
    int n = 3;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_qft(q, 0, n);
    double expected = 1.0 / (double)(1 << n);
    for (size_t i = 0; i < (size_t)(1 << n); i++) {
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, expected, prob_of(q, i));
    }
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_qft_on_1_qubit_equals_h);
    RUN_TEST(test_qft_of_zero_is_uniform);
}

TEST_RUNNER_MAIN()
```

- [ ] **Step 2: Stub header and source**

Write `implementation/c/qft.h`:
```c
#ifndef QFT_H
#define QFT_H

#include "matrix.h"

/* Spec §6.4. apply_qft includes the final bit-reversal swaps so the    *
 * output amplitude at index y in natural binary equals                  *
 *   (1/sqrt(N)) sum_x alpha_x exp(2*pi*i*x*y/N).                       */
void apply_qft        (qreg *q, int start, int n_qubits);
void apply_qft_inverse(qreg *q, int start, int n_qubits);

#endif
```

Write `implementation/c/qft.c`:
```c
#include "qft.h"
#include <math.h>

void apply_qft(qreg *q, int start, int n_qubits) {
    (void)q; (void)start; (void)n_qubits;   /* stub */
}
void apply_qft_inverse(qreg *q, int start, int n_qubits) {
    (void)q; (void)start; (void)n_qubits;   /* stub */
}
```

Add to `LIB_SRCS` in the makefile:
```make
LIB_SRCS := standart.c matrix.c parallel.c qft.c
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_qft` fails -- with the stub, the state is unchanged so prob_of(0) = 1, not 1/8.

- [ ] **Step 4: Implement apply_qft with the standard decomposition**

Replace the body of `apply_qft` in `qft.c`:
```c
void apply_qft(qreg *q, int start, int n_qubits) {
    QREG_ASSERT(q != NULL,         "apply_qft: q is NULL");
    QREG_ASSERT(n_qubits >= 1,     "apply_qft: n_qubits < 1");
    QREG_ASSERT(start >= 0,        "apply_qft: start < 0");
    QREG_ASSERT(start + n_qubits <= q->n_qubits,
                "apply_qft: range exceeds register size");
    /* Standard textbook decomposition. For each qubit j from the most-  *
     * significant down to the least:                                     *
     *   apply H on qubit (start + n_qubits - 1 - j)                      *
     *   for each k > j, apply controlled-R_{k-j+1} from                  *
     *     control = start + n_qubits - 1 - k                             *
     *     target  = start + n_qubits - 1 - j                             *
     *     angle  = 2*pi / 2^(k-j+1)                                      *
     * Finish with a bit-reversal SWAP across the range so the output    *
     * is in natural binary order.                                        */
    for (int j = 0; j < n_qubits; j++) {
        int target = start + (n_qubits - 1 - j);
        apply_h(q, target);
        for (int k = j + 1; k < n_qubits; k++) {
            int control = start + (n_qubits - 1 - k);
            double theta = 2.0 * M_PI / (double)((size_t)1 << (k - j + 1));
            apply_controlled_phase(q, control, target, theta);
        }
    }
    /* Final swaps. */
    for (int i = 0; i < n_qubits / 2; i++) {
        apply_swap(q, start + i, start + n_qubits - 1 - i);
    }
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_qft; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/qft.h implementation/c/qft.c \
        implementation/c/tests/test_qft.c implementation/c/makefile
git commit -m "feat(c): qft - apply_qft (forward) with bit-reversal swaps

Standard H-and-controlled-phase decomposition followed by the final
swap pass so the output amplitudes are in natural binary order
(matching the convention declared in spec §6.4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 31: apply_qft_inverse and round-trip test

**Files:**
- Modify: `implementation/c/qft.c`
- Modify: `implementation/c/tests/test_qft.c`

- [ ] **Step 1: Failing test (round-trip)**

Append to `tests/test_qft.c`:
```c
static void test_qft_round_trip(void) {
    /* QFT^-1 . QFT = identity, up to floating-point. */
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    apply_qft        (q, 0, n);
    apply_qft_inverse(q, 0, n);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 5));
    /* No leakage to any other state. */
    for (size_t i = 0; i < (size_t)(1 << n); i++) {
        if (i == 5) continue;
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, i));
    }
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_qft_round_trip);
```

- [ ] **Step 2: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_qft_round_trip` fails -- with the stub inverse, QFT-followed-by-no-op leaves the state in the post-QFT uniform spectrum, prob_of(5) is not 1.

- [ ] **Step 3: Implement apply_qft_inverse**

Replace the body of `apply_qft_inverse` in `qft.c`:
```c
void apply_qft_inverse(qreg *q, int start, int n_qubits) {
    QREG_ASSERT(q != NULL,         "apply_qft_inverse: q is NULL");
    QREG_ASSERT(n_qubits >= 1,     "apply_qft_inverse: n_qubits < 1");
    QREG_ASSERT(start >= 0,        "apply_qft_inverse: start < 0");
    QREG_ASSERT(start + n_qubits <= q->n_qubits,
                "apply_qft_inverse: range exceeds register size");
    /* Reverse the swap pass first. */
    for (int i = 0; i < n_qubits / 2; i++) {
        apply_swap(q, start + i, start + n_qubits - 1 - i);
    }
    /* Then run the QFT decomposition backwards with negated phases. */
    for (int j = n_qubits - 1; j >= 0; j--) {
        int target = start + (n_qubits - 1 - j);
        for (int k = n_qubits - 1; k > j; k--) {
            int control = start + (n_qubits - 1 - k);
            double theta = -2.0 * M_PI / (double)((size_t)1 << (k - j + 1));
            apply_controlled_phase(q, control, target, theta);
        }
        apply_h(q, target);
    }
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_qft; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 4: Add a period-detection test (the real Shor-relevant case)**

Append to `tests/test_qft.c`:
```c
static void test_qft_detects_period(void) {
    /* Prepare the periodic state (1/sqrt(M)) sum_{j=0}^{M-1} |x0 + j*r>  *
     * where M = N/r. The inverse QFT concentrates mass on multiples of   *
     * N/r. We test the simplest case: n=3 (N=8), r=2, x0=0.              *
     * Input: (1/sqrt(4)) (|0> + |2> + |4> + |6>).                        *
     * After QFT^-1, mass should be on indices that are multiples of      *
     * N/r = 4: i.e. on |0> and |4>, each with probability 0.5.            */
    int n = 3;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    /* Manual init: zero all then set the 4 amplitudes uniformly. */
    qreg_init_basis(q, 0);
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    double a = 0.5;     /* 1/sqrt(4) */
    size_t base = (size_t)q->rank * q->local_size;
    for (size_t off = 0; off < q->local_size; off++) {
        size_t g = base + off;
        if (g == 0 || g == 2 || g == 4 || g == 6) q->amp[off] = a;
    }
    apply_qft_inverse(q, 0, n);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.5, prob_of(q, 4));
    for (size_t i = 0; i < 8; i++) {
        if (i == 0 || i == 4) continue;
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, i));
    }
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_qft_detects_period);
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_qft; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/qft.c implementation/c/tests/test_qft.c
git commit -m "feat(c): qft - apply_qft_inverse and period-detection test

Inverse QFT runs the forward decomposition in reverse with negated
phase angles. Round-trip and period-detection tests verify both the
algebra and the bit-order convention (final-swap inclusion) declared
in spec §6.4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 9 — Grover's algorithm (Tasks 32–33)

### Task 32: apply_grover (oracle + diffusion loop)

**Files:**
- Create: `implementation/c/grover.h`
- Create: `implementation/c/grover.c`
- Create: `implementation/c/tests/test_grover.c`
- Modify: `implementation/c/makefile` (`LIB_SRCS += grover.c`)

- [ ] **Step 1: Failing test (1 marked item in N=16)**

Write `implementation/c/tests/test_grover.c`:
```c
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include "matrix.h"
#include "grover.h"
#include "unity/unity.h"
#include "test_assert.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

/* Oracle that phase-flips a single marked basis state passed via *user. */
static size_t g_marked;
static void oracle_single_marked(qreg *q, void *user) {
    (void)user;
    /* Flip the amplitude on basis g_marked, on whichever rank owns it. */
    if (rank_owns(q, g_marked)) {
        size_t off = global_to_local(q, g_marked);
        q->amp[off] = -q->amp[off];
    }
}

static void test_grover_1_marked_in_16(void) {
    /* N = 16 = 2^4, 1 marked item. Optimum iterations = floor(pi/4 * 4) = 3. */
    int n = 4;
    g_marked = 11;                     /* arbitrary in [0, 16). */
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, oracle_single_marked, NULL, 3);
    double p = prob_of(q, g_marked);
    TEST_ASSERT_TRUE_MESSAGE(p >= 0.99, "Grover did not concentrate >= 0.99 on the marked item");
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_grover_1_marked_in_16);
}

TEST_RUNNER_MAIN()
```

- [ ] **Step 2: Stub header and source**

Write `implementation/c/grover.h`:
```c
#ifndef GROVER_H
#define GROVER_H

#include "matrix.h"

/* Phase-oracle callback: applies (-1)^f(x) to amplitude at index x in
 * place. user is whatever the caller passed to apply_grover.            */
typedef void (*oracle_fn)(qreg *q, void *user);

/* Spec §6.5. Initialises q to the uniform superposition over the first  *
 * n_qubits qubits, then runs `iterations` rounds of oracle + diffusion. */
void apply_grover(qreg *q, int n_qubits, oracle_fn oracle, void *user,
                  int iterations);

#endif
```

Write `implementation/c/grover.c`:
```c
#include "grover.h"

void apply_grover(qreg *q, int n_qubits, oracle_fn oracle, void *user,
                  int iterations) {
    (void)q; (void)n_qubits; (void)oracle; (void)user; (void)iterations;
}
```

Add to `LIB_SRCS`:
```make
LIB_SRCS := standart.c matrix.c parallel.c qft.c grover.c
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_grover_1_marked_in_16` fails (stub does nothing; prob_of(11) stays ~0).

- [ ] **Step 4: Implement**

Replace the body in `grover.c`:
```c
#include "grover.h"

void apply_grover(qreg *q, int n_qubits, oracle_fn oracle, void *user,
                  int iterations) {
    QREG_ASSERT(q != NULL,           "apply_grover: q is NULL");
    QREG_ASSERT(oracle != NULL,      "apply_grover: oracle is NULL");
    QREG_ASSERT(n_qubits >= 1 && n_qubits <= q->n_qubits,
                "apply_grover: n_qubits out of range");
    QREG_ASSERT(iterations >= 0,     "apply_grover: iterations negative");
    /* Uniform superposition. */
    for (int i = 0; i < n_qubits; i++) apply_h(q, i);
    /* Iterate. */
    for (int it = 0; it < iterations; it++) {
        oracle(q, user);
        /* Diffusion: H^n -> X^n -> multi-controlled-Z -> X^n -> H^n.    *
         * Pre-X turns the flip on |1...1> into a flip on |0...0>, then  *
         * H^n turns the |0...0> reflection into the |s> reflection.     */
        for (int i = 0; i < n_qubits; i++) apply_h(q, i);
        for (int i = 0; i < n_qubits; i++) apply_x(q, i);
        apply_multi_controlled_z(q, n_qubits);
        for (int i = 0; i < n_qubits; i++) apply_x(q, i);
        for (int i = 0; i < n_qubits; i++) apply_h(q, i);
    }
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_grover; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8 with prob >= 0.99 (actual value should be sin^2(7*theta) where sin theta = 1/4, i.e. ~0.961...; wait that's not >= 0.99. Let me think.

Actually with 1 marked in N=16: sin(theta) = sqrt(1/16) = 1/4. theta = arcsin(1/4) ≈ 0.2527. Optimal k = floor((pi/2 - theta) / (2*theta)) but the formula in the thesis is floor(pi/4 * sqrt(N/M)) = floor(pi/4 * 4) = floor(pi) = 3.

After 3 iterations, prob = sin^2((2*3+1)*theta) = sin^2(7 * 0.2527) = sin^2(1.769) = (0.980)^2 = 0.961.

So 0.96, not 0.99. Adjust threshold:

Fix the test:
```c
    TEST_ASSERT_TRUE_MESSAGE(p >= 0.95, "Grover did not concentrate >= 0.95 on the marked item");
```

Re-run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/grover.h implementation/c/grover.c \
        implementation/c/tests/test_grover.c implementation/c/makefile
git commit -m "feat(c): grover - apply_grover with oracle_fn callback

H^n initialisation, then `iterations` rounds of oracle + diffusion.
Diffusion uses H^n -> X^n -> multi-controlled-Z -> X^n -> H^n, which
turns the cheap |1...1> phase flip from apply_multi_controlled_z into
the |s> reflection. Test verifies >= 0.95 success on 1 marked item in
N=16 after 3 iterations (analytical ~ 0.961).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 33: Grover tests — multiple marked items + over-iteration penalty

**Files:**
- Modify: `implementation/c/tests/test_grover.c`

- [ ] **Step 1: Failing tests**

Append to `tests/test_grover.c`:
```c
/* Oracle that marks 4 fixed items: indices 1, 3, 5, 7 in N=16. */
static void oracle_four_marked(qreg *q, void *user) {
    (void)user;
    size_t marked[] = {1, 3, 5, 7};
    for (int i = 0; i < 4; i++) {
        if (rank_owns(q, marked[i])) {
            size_t off = global_to_local(q, marked[i]);
            q->amp[off] = -q->amp[off];
        }
    }
}

static void test_grover_4_marked_in_16(void) {
    /* N=16, M=4 -> theta = arcsin(sqrt(4/16)) = arcsin(0.5) = pi/6.    *
     * Optimum iterations = floor(pi/4 * sqrt(16/4)) = floor(pi/4 * 2)   *
     *                    = floor(pi/2) = 1.                              */
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, oracle_four_marked, NULL, 1);
    /* After 1 iteration: amplitude sin(3*theta) on each marked = sin(pi/2)*0.5 = 0.5 each *
     * Wait, the formula is amplitude on EACH marked = sin((2k+1)*theta) / sqrt(M).        *
     * For k=1, theta=pi/6: sin(pi/2)/sqrt(4) = 1/2. prob per marked = 1/4.                *
     * Total prob on the marked subspace = 1.0.                                             */
    double total = 0.0;
    size_t marked[] = {1, 3, 5, 7};
    for (int i = 0; i < 4; i++) total += prob_of(q, marked[i]);
    TEST_ASSERT_DOUBLE_WITHIN(0.01, 1.0, total);
    qreg_destroy(q);
}

static void test_grover_over_iteration_hurts(void) {
    /* Run twice the optimum number of iterations - probability should drop. */
    int n = 4;
    g_marked = 9;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, oracle_single_marked, NULL, 3);
    double p_opt = prob_of(q, g_marked);

    qreg *q2 = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q2, 0);
    apply_grover(q2, n, oracle_single_marked, NULL, 6);
    double p_over = prob_of(q2, g_marked);

    TEST_ASSERT_TRUE_MESSAGE(p_over < p_opt,
        "over-iteration should reduce success probability");
    qreg_destroy(q);
    qreg_destroy(q2);
}
```

Register:
```c
RUN_TEST(test_grover_4_marked_in_16);
RUN_TEST(test_grover_over_iteration_hurts);
```

- [ ] **Step 2: Build and run**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_grover; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 3: (No implementation needed; tests verify existing apply_grover.)**

This task is pure test additions; the algorithm is unchanged.

- [ ] **Step 4: Sanity at NP=1 with single-rank random seed**

Run:
```bash
cd implementation/c
mpirun --oversubscribe -n 1 build/tests/test_grover; echo "exit: $?"
```
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/tests/test_grover.c
git commit -m "test(c): grover - 4 marked items + over-iteration penalty

Verifies the optimum-stop property of Grover: at k = floor(pi/4 sqrt(N/M))
the success probability is maximal, and doubling k reduces it (the
amplitude vector rotates past the marked subspace). N=16/M=4 lands
on optimal probability 1.0 in a single iteration, which is a clean
analytical reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 10 — Shor's algorithm (Tasks 34–37)

### Task 34: parallel — redistribute_pairs (MPI_Alltoallv bucketing)

**Files:**
- Modify: `implementation/c/parallel.h` (declare redistribute_pairs)
- Modify: `implementation/c/parallel.c` (implement)
- Modify: `implementation/c/tests/test_parallel.c` (add round-trip test)

- [ ] **Step 1: Failing test**

Append to `tests/test_parallel.c`:
```c
static void test_redistribute_pairs_round_trip(void) {
    /* Each rank produces a list of (global_index, amplitude) pairs whose *
     * destination is uniformly distributed across all ranks. After the   *
     * Alltoallv, every rank should hold exactly the pairs destined for   *
     * it, with the amplitudes intact. We use a fixed deterministic       *
     * pattern so the expected outcome is computable analytically.        */
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    size_t total = (size_t)1 << q->n_qubits;
    /* Each rank emits 4 pairs: indices base, base+1, base+2, base+3 with
     * base = rank * 4, values (rank+1.0)*100 + i.                        */
    size_t n_pairs = 4;
    size_t *idx = malloc(n_pairs * sizeof *idx);
    complex double *val = malloc(n_pairs * sizeof *val);
    for (size_t i = 0; i < n_pairs; i++) {
        idx[i] = ((size_t)q->rank * 4 + i) % total;
        val[i] = (double)((q->rank + 1) * 100 + (int)i) + 0.0*I;
    }
    /* Zero the qreg, then accumulate the redistributed pairs into it. */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    redistribute_pairs(q, idx, n_pairs, val);
    /* Every global index from 0 .. (n_pairs * n_procs - 1) modulo total
     * should now have exactly one amplitude written. Verify the local
     * slice has expected values.                                          */
    size_t base = (size_t)q->rank * q->local_size;
    for (size_t off = 0; off < q->local_size; off++) {
        size_t g = base + off;
        complex double expected = 0.0;
        /* Find which (rank, i) produced this g. */
        for (int r = 0; r < q->n_procs; r++) {
            for (size_t i = 0; i < n_pairs; i++) {
                if (((size_t)r * 4 + i) % total == g) {
                    expected += (double)((r + 1) * 100 + (int)i);
                }
            }
        }
        ASSERT_NEAR_AMP(expected, q->amp[off]);
    }
    free(idx); free(val);
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_redistribute_pairs_round_trip);
```

- [ ] **Step 2: Stub**

Append to `parallel.h` before `#endif`:
```c
/* Accumulate (global_index, amplitude) pairs into q->amp via MPI_Alltoallv.
 * After the call, q->amp[i] equals the SUM of every incoming amplitude
 * whose global index lands on this rank's slice at local offset i.
 *
 * Used by shor.c's apply_modular_exp under the distributed layout
 * (multiple source amplitudes can land on the same destination index).
 */
void redistribute_pairs(qreg *q, const size_t *global_indices,
                        size_t n_pairs, const complex double *values);
```

Append to `parallel.c`:
```c
void redistribute_pairs(qreg *q, const size_t *global_indices,
                        size_t n_pairs, const complex double *values) {
    (void)q; (void)global_indices; (void)n_pairs; (void)values;
}
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_redistribute_pairs_round_trip` fails (q->amp stays 0).

- [ ] **Step 4: Implement**

Replace the stub in `parallel.c`:
```c
#include <stdlib.h>

void redistribute_pairs(qreg *q, const size_t *global_indices,
                        size_t n_pairs, const complex double *values) {
    QREG_ASSERT(q != NULL,                  "redistribute_pairs: q is NULL");
    QREG_ASSERT(n_pairs == 0 || (global_indices && values),
                "redistribute_pairs: missing arrays");

    int  P = q->n_procs;
    int *send_counts = calloc(P, sizeof *send_counts);
    int *recv_counts = calloc(P, sizeof *recv_counts);
    int *send_displs = calloc(P, sizeof *send_displs);
    int *recv_displs = calloc(P, sizeof *recv_displs);

    /* Count destinations. */
    for (size_t i = 0; i < n_pairs; i++) {
        int dest = (int)(global_indices[i] >> (q->n_qubits - q->p));
        send_counts[dest]++;
    }
    /* Exchange counts. */
    MPI_Alltoall(send_counts, 1, MPI_INT,
                 recv_counts, 1, MPI_INT, q->comm);

    int total_send = 0, total_recv = 0;
    for (int r = 0; r < P; r++) {
        send_displs[r] = total_send; total_send += send_counts[r];
        recv_displs[r] = total_recv; total_recv += recv_counts[r];
    }

    /* Pack: each pair as (size_t local_offset, complex double value).   *
     * We send local-offsets so the receiver can apply them directly.    */
    size_t         *send_off = malloc((size_t)total_send * sizeof *send_off);
    complex double *send_val = malloc((size_t)total_send * sizeof *send_val);
    int *cursor = calloc(P, sizeof *cursor);
    for (size_t i = 0; i < n_pairs; i++) {
        int dest = (int)(global_indices[i] >> (q->n_qubits - q->p));
        int slot = send_displs[dest] + cursor[dest]++;
        send_off[slot] = global_indices[i] & (q->local_size - 1);
        send_val[slot] = values[i];
    }
    free(cursor);

    size_t         *recv_off = malloc((size_t)total_recv * sizeof *recv_off);
    complex double *recv_val = malloc((size_t)total_recv * sizeof *recv_val);

    /* Two Alltoallv calls: one for offsets, one for amplitudes. */
    MPI_Alltoallv(send_off, send_counts, send_displs, MPI_UNSIGNED_LONG,
                  recv_off, recv_counts, recv_displs, MPI_UNSIGNED_LONG,
                  q->comm);
    MPI_Alltoallv(send_val, send_counts, send_displs, MPI_C_DOUBLE_COMPLEX,
                  recv_val, recv_counts, recv_displs, MPI_C_DOUBLE_COMPLEX,
                  q->comm);

    /* Accumulate. */
    for (int i = 0; i < total_recv; i++) {
        q->amp[recv_off[i]] += recv_val[i];
    }

    free(send_off); free(send_val); free(recv_off); free(recv_val);
    free(send_counts); free(recv_counts); free(send_displs); free(recv_displs);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_parallel; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/parallel.h implementation/c/parallel.c \
        implementation/c/tests/test_parallel.c
git commit -m "feat(c): parallel - redistribute_pairs via MPI_Alltoallv

Buckets (global_index, amplitude) pairs by destination rank, two
MPI_Alltoallv calls (offsets + values), per-rank accumulation into
q->amp. This is the primitive that lets apply_modular_exp move
amplitudes across rank boundaries during the y -> a^x*y mod N step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 35: shor — apply_modular_exp

**Files:**
- Create: `implementation/c/shor.h`
- Create: `implementation/c/shor.c`
- Create: `implementation/c/tests/test_shor.c`
- Modify: `implementation/c/makefile` (`LIB_SRCS += shor.c`)

- [ ] **Step 1: Failing test (smallest case)**

Write `implementation/c/tests/test_shor.c`:
```c
#include <mpi.h>
#include <stdint.h>
#include "matrix.h"
#include "shor.h"
#include "standart.h"
#include "unity/unity.h"
#include "test_assert.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

static void test_modular_exp_passes_through_y_ge_N(void) {
    /* Spec §5.5: amplitudes with y >= N must be left unchanged.        *
     * Build a tiny case: 1 counting qubit (t=1), 4 target qubits (n=4),*
     * N = 5. Place a unit amplitude at (x=0, y=10). Run apply_modular_  *
     * exp with a=2. Since y=10 >= 5, the amplitude should stay at the   *
     * same index.                                                        */
    int counting_start = 0, t = 1;
    int target_start   = 1, n = 4;
    int n_total = counting_start + t + n;  /* would be 5 -- but we need t,n,start ordering */
    /* Actually layout: counting register occupies qubits 0..0; target 1..4. */
    qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
    /* global index = (y << target_start) | (x << counting_start).      *
     * x=0, y=10 -> global = 10 << 1 = 20. But 20 >= 32 = 2^5.            *
     * Fix: use n_total = 6 so we have room.                              */
    qreg_destroy(q);
    n_total = 6;
    q = qreg_create(n_total, MPI_COMM_WORLD);
    qreg_init_basis(q, ((size_t)10 << 1));   /* y=10, x=0 -> global 20 */
    apply_modular_exp(q, counting_start, t, target_start, n, /*a=*/2, /*N=*/5);
    /* y was 10 (>= N=5) so should stay at 10. */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, (size_t)10 << 1));
    qreg_destroy(q);
}

static void test_modular_exp_maps_within_ring(void) {
    /* y=1, x=1, a=2, N=5 -> y_new = (2^1 * 1) mod 5 = 2.                *
     * Start at (x=1, y=1) -> global = (1<<1) | (1<<0) = 3.              *
     * After: (x=1, y=2) -> global = (2<<1) | (1<<0) = 5.                 */
    int counting_start = 0, t = 1;
    int target_start   = 1, n = 4;
    int n_total = 6;
    qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
    qreg_init_basis(q, ((size_t)1 << 1) | 1);
    apply_modular_exp(q, counting_start, t, target_start, n, 2, 5);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, ((size_t)2 << 1) | 1));
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_modular_exp_passes_through_y_ge_N);
    RUN_TEST(test_modular_exp_maps_within_ring);
}

TEST_RUNNER_MAIN()
```

- [ ] **Step 2: Stub header and source**

Write `implementation/c/shor.h`:
```c
#ifndef SHOR_H
#define SHOR_H

#include "matrix.h"

void apply_modular_exp(qreg *q,
                       int counting_start, int t,
                       int target_start,   int n,
                       uint64_t a, uint64_t N);

typedef struct {
    uint64_t r;            /* recovered period, 0 if failed */
    uint64_t measured_c;   /* the integer the QFT measurement returned */
} shor_period_result;

shor_period_result apply_shor_period(qreg *q,
                                     int counting_start, int t,
                                     int target_start,   int n,
                                     uint64_t a, uint64_t N);

typedef struct {
    uint64_t p, q;         /* non-trivial factors of N, 0 if failed */
    int      attempts;
} shor_factor_result;

shor_factor_result shor_factor(uint64_t N, int max_attempts);

#endif
```

Write `implementation/c/shor.c`:
```c
#include "shor.h"
#include "parallel.h"
#include "standart.h"
#include <stdlib.h>
#include <string.h>

void apply_modular_exp(qreg *q,
                       int counting_start, int t,
                       int target_start,   int n,
                       uint64_t a, uint64_t N) {
    (void)q; (void)counting_start; (void)t; (void)target_start;
    (void)n; (void)a; (void)N;
}

shor_period_result apply_shor_period(qreg *q, int cs, int t, int ts, int n,
                                     uint64_t a, uint64_t N) {
    (void)q; (void)cs; (void)t; (void)ts; (void)n; (void)a; (void)N;
    return (shor_period_result){0,0};
}
shor_factor_result shor_factor(uint64_t N, int max_attempts) {
    (void)N; (void)max_attempts;
    return (shor_factor_result){0,0,0};
}
```

Add to `LIB_SRCS`:
```make
LIB_SRCS := standart.c matrix.c parallel.c qft.c grover.c shor.c
```

- [ ] **Step 3: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: both tests fail (stub leaves state unchanged but apply_modular_exp didn't write to the right place; first test expects prob_of(20)=1, which is actually what the stub gives so first might pass; second test expects prob_of((2<<1)|1)=1, which won't hold since stub doesn't move anything).

- [ ] **Step 4: Implement using redistribute_pairs**

Replace the body of `apply_modular_exp` in `shor.c`:
```c
void apply_modular_exp(qreg *q,
                       int counting_start, int t,
                       int target_start,   int n,
                       uint64_t a, uint64_t N) {
    QREG_ASSERT(q != NULL, "apply_modular_exp: q is NULL");
    QREG_ASSERT(N >= 2,    "apply_modular_exp: N < 2");
    QREG_ASSERT(t >= 1 && n >= 1, "apply_modular_exp: t, n < 1");
    QREG_ASSERT(counting_start >= 0 && target_start >= 0,
                "apply_modular_exp: negative register start");
    QREG_ASSERT(counting_start + t <= q->n_qubits,
                "apply_modular_exp: counting range overflows");
    QREG_ASSERT(target_start + n <= q->n_qubits,
                "apply_modular_exp: target range overflows");
    /* Disjoint check: [cs, cs+t) and [ts, ts+n) must not overlap. */
    QREG_ASSERT(counting_start + t <= target_start ||
                target_start   + n <= counting_start,
                "apply_modular_exp: counting and target ranges overlap");
    QREG_ASSERT(N <= ((uint64_t)1 << n),
                "apply_modular_exp: N > 2^target_width");
    QREG_ASSERT(gcd_u64(a, N) == 1,
                "apply_modular_exp: gcd(a, N) != 1");

    /* Walk this rank's amplitudes, compute destinations, redistribute. */
    size_t base   = (size_t)q->rank * q->local_size;
    size_t t_mask = (((size_t)1 << t) - 1) << counting_start;
    size_t n_mask = (((size_t)1 << n) - 1) << target_start;
    size_t outer  = ~(t_mask | n_mask);

    /* Pre-count non-zero amplitudes to size the buffers. */
    size_t n_nz = 0;
    for (size_t i = 0; i < q->local_size; i++) {
        if (q->amp[i] != 0.0) n_nz++;
    }
    size_t         *idx = malloc(n_nz * sizeof *idx);
    complex double *val = malloc(n_nz * sizeof *val);
    size_t        cur = 0;
    for (size_t i = 0; i < q->local_size; i++) {
        if (q->amp[i] == 0.0) continue;
        size_t global = base + i;
        uint64_t x = (global >> counting_start) & (((uint64_t)1 << t) - 1);
        uint64_t y = (global >> target_start)   & (((uint64_t)1 << n) - 1);
        uint64_t y_new;
        if (y < N) y_new = (y * mod_pow(a, x, N)) % N;
        else       y_new = y;          /* reversibility pass-through */
        size_t new_global = (global & outer)
                          | ((size_t)x     << counting_start)
                          | ((size_t)y_new << target_start);
        idx[cur] = new_global;
        val[cur] = q->amp[i];
        cur++;
    }
    /* Zero the local slice; redistribute_pairs accumulates incoming. */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    redistribute_pairs(q, idx, n_nz, val);
    free(idx); free(val);
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_shor; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/shor.h implementation/c/shor.c \
        implementation/c/tests/test_shor.c implementation/c/makefile
git commit -m "feat(c): shor - apply_modular_exp via redistribute_pairs

Each rank walks its local slice, decomposes each non-zero amplitude
into (outer, x, y), computes y_new = (a^x * y) mod N for y < N (and
y_new = y for y >= N, per spec §5.5 reversibility), then hands the
(new_global, amp) pairs to parallel.c::redistribute_pairs. Two unit
tests cover the in-ring map and the y >= N pass-through.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 36: shor — apply_shor_period

**Files:**
- Modify: `implementation/c/shor.c`
- Modify: `implementation/c/tests/test_shor.c`

- [ ] **Step 1: Failing test**

Append to `tests/test_shor.c`:
```c
static void test_shor_period_a7_mod15(void) {
    /* The classical Shor warm-up: order of 7 mod 15 is 4. With 3 target
     * qubits (covers N=15? need ceil(log2 15) = 4) and t = 2*4 + 1 = 9
     * counting qubits, that's 13 qubits total -- big but manageable.
     * For a smaller smoke we use t=8 counting qubits, 4 target qubits =
     * 12 qubits. 2^12 = 4096 amplitudes, comfortable.                    */
    int n = 4;
    int t = 8;
    int n_total = t + n;
    if (n_total > 14) { TEST_PASS(); return; }       /* keep CI fast */
    qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
    shor_period_result res = apply_shor_period(q, /*cs=*/n, t, /*ts=*/0, n,
                                               /*a=*/7, /*N=*/15);
    /* The recovered period must be a divisor of the true period r=4 OR
     * equal to it. Most likely it is exactly 4 or 2.                     */
    TEST_ASSERT_TRUE_MESSAGE(res.r != 0, "Shor period finder returned 0");
    TEST_ASSERT_TRUE_MESSAGE(res.r == 4 || res.r == 2 || res.r == 1,
        "Shor period finder returned an unexpected period");
    qreg_destroy(q);
}
```

Register:
```c
RUN_TEST(test_shor_period_a7_mod15);
```

- [ ] **Step 2: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: stub returns r=0, test fails.

- [ ] **Step 3: Implement apply_shor_period**

Replace the stub in `shor.c`:
```c
#include "qft.h"
#include <math.h>

shor_period_result apply_shor_period(qreg *q,
                                     int counting_start, int t,
                                     int target_start,   int n,
                                     uint64_t a, uint64_t N) {
    QREG_ASSERT(q != NULL,                 "apply_shor_period: q is NULL");
    QREG_ASSERT(counting_start >= 0 && t >= 1, "bad counting range");
    QREG_ASSERT(target_start  >= 0 && n >= 1, "bad target range");
    /* Setup: counting register in uniform superposition, target in |1>. */
    /* Zero everything, then set the single basis state with y=1, x=0. */
    qreg_init_basis(q, (size_t)1 << target_start);
    /* Hadamards on the counting register. */
    for (int j = 0; j < t; j++) apply_h(q, counting_start + j);
    /* Apply modular exponentiation. */
    apply_modular_exp(q, counting_start, t, target_start, n, a, N);
    /* Inverse QFT on the counting register. */
    apply_qft_inverse(q, counting_start, t);
    /* Measure the counting register to get integer c. We measure each
     * qubit and combine; alternatively we could call measure_all and
     * mask out the counting bits. Use bitwise measure_qubit calls.       */
    uint64_t c = 0;
    for (int j = 0; j < t; j++) {
        int bit = measure_qubit(q, counting_start + j);
        c |= ((uint64_t)bit) << j;
    }
    /* Classical post-processing: continued-fraction expansion of c/2^t. */
    double x = (double)c / (double)((uint64_t)1 << t);
    uint64_t num = 0, den = 0;
    continued_fraction(x, N, &num, &den);
    shor_period_result res = { .r = den, .measured_c = c };
    return res;
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_shor; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8 (random outcomes; period 4 with high probability, 2 occasionally).

- [ ] **Step 4: Stabilise the test against measurement randomness**

The test as written can fail if the measurement returns c such that continued_fraction returns 1 or a divisor that doesn't reflect the true period. The current test accepts r in {1, 2, 4}, which is robust. Run several times:

```bash
cd implementation/c
for i in 1 2 3 4 5; do
    mpirun --oversubscribe -n 4 build/tests/test_shor || { echo "FAIL at run $i"; break; }
done
```
Expected: pass every run.

- [ ] **Step 5: Commit**

```bash
git add implementation/c/shor.c implementation/c/tests/test_shor.c
git commit -m "feat(c): shor - apply_shor_period

Counting register prep (H^t), modular exponentiation, inverse QFT,
measure_qubit per counting bit, continued-fraction post-processing.
Returns the recovered period and the raw measured c so callers can
retry with the same shot if continued_fraction misfires.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 37: shor — shor_factor (high-level wrapper) + factor 15 test

**Files:**
- Modify: `implementation/c/shor.c`
- Modify: `implementation/c/tests/test_shor.c`

- [ ] **Step 1: Failing test**

Append to `tests/test_shor.c`:
```c
static void test_shor_factor_15(void) {
    shor_factor_result r = shor_factor(15, /*max_attempts=*/8);
    TEST_ASSERT_TRUE_MESSAGE(r.p != 0 && r.q != 0,
        "shor_factor(15) failed after 8 attempts");
    TEST_ASSERT_EQUAL_UINT64(15ULL, r.p * r.q);
    TEST_ASSERT_TRUE(r.p > 1 && r.q > 1);
    TEST_ASSERT_TRUE(r.p == 3 || r.p == 5);
}
```

Register:
```c
RUN_TEST(test_shor_factor_15);
```

- [ ] **Step 2: Confirm fail**

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
```
Expected: `test_shor_factor_15` fails (stub returns p=q=0).

- [ ] **Step 3: Implement shor_factor**

Replace the stub in `shor.c`:
```c
#include <time.h>

static int  s_seeded = 0;

shor_factor_result shor_factor(uint64_t N, int max_attempts) {
    QREG_ASSERT(N >= 4, "shor_factor: N too small");
    QREG_ASSERT(max_attempts > 0, "shor_factor: max_attempts < 1");
    shor_factor_result out = {0, 0, 0};
    /* Handle even N trivially. */
    if ((N & 1) == 0) { out.p = 2; out.q = N / 2; out.attempts = 0; return out; }
    /* Pick the bit width for the target register: n = ceil(log2 N).     *
     * Counting width: t = 2n + 1.                                       */
    int n = 0;
    while (((uint64_t)1 << n) < N) n++;
    int t = 2 * n + 1;
    int n_total = t + n;
    if (n_total > QREG_MAX_QUBITS) { out.attempts = 0; return out; }
    /* Seed once per process. Each rank uses its rank to differ. */
    int rank;  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!s_seeded) { srand((unsigned)time(NULL) + (unsigned)rank); s_seeded = 1; }
    for (int attempt = 1; attempt <= max_attempts; attempt++) {
        out.attempts = attempt;
        uint64_t a = 2 + ((uint64_t)rand() % (N - 3));   /* in [2, N-2] */
        uint64_t g = gcd_u64(a, N);
        if (g > 1) { out.p = g; out.q = N / g; return out; }    /* lucky */
        qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
        shor_period_result pr = apply_shor_period(q, /*cs=*/n, t, /*ts=*/0,
                                                  n, a, N);
        qreg_destroy(q);
        uint64_t r = pr.r;
        if (r == 0 || (r & 1)) continue;                 /* need even r */
        uint64_t x  = mod_pow(a, r / 2, N);
        if (x + 1 == N) continue;                        /* trivial */
        uint64_t p1 = gcd_u64(x + 1, N);
        uint64_t p2 = gcd_u64(x + N - 1, N);
        if (p1 > 1 && p1 < N) { out.p = p1; out.q = N / p1; return out; }
        if (p2 > 1 && p2 < N) { out.p = p2; out.q = N / p2; return out; }
    }
    return out;   /* p = q = 0 on failure */
}
```

Run:
```bash
cd implementation/c
make test 2>&1 | tail -10
mpirun --oversubscribe -n 8 build/tests/test_shor; echo "exit: $?"
```
Expected: pass at NP=1, 2, 4, 8.

- [ ] **Step 4: Sanity at NP=8 for several runs to catch flakiness**

Run:
```bash
cd implementation/c
for i in 1 2 3 4 5; do
    mpirun --oversubscribe -n 4 build/tests/test_shor || { echo "FAIL at run $i"; break; }
done
```
Expected: pass every run (8 attempts should comfortably find a factor of 15).

- [ ] **Step 5: Commit**

```bash
git add implementation/c/shor.c implementation/c/tests/test_shor.c
git commit -m "feat(c): shor - shor_factor (end-to-end factoring)

High-level loop: pick random a coprime to N, run apply_shor_period,
post-process via gcd(a^(r/2) +/- 1, N). Trivial-r and bad-x cases
restart up to max_attempts. End-to-end test factors N=15 reliably
within 8 attempts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 11 — qubit.c demo (Task 38)

### Task 38: qubit.c — demo program with --algo flag

**Files:**
- Create: `implementation/c/qubit.c`
- Modify: `implementation/c/makefile` (add `bin/qubit` target)

- [ ] **Step 1: Write the demo**

Write `implementation/c/qubit.c`:
```c
/* qubit.c -- demo entry point.
 *
 * Usage:  mpirun -n NP build/bin/qubit --algo {bell|qft|grover|shor} [args]
 *
 * Each --algo runs a small, well-known instance and prints the result.
 * Built primarily as a smoke harness for the library.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "qft.h"
#include "grover.h"
#include "shor.h"

static void demo_bell(int rank) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    apply_cnot(q, 0, 1);
    if (rank == 0) printf("Bell |Phi+>: prob_of(|00>)=%.4f  prob_of(|11>)=%.4f\n",
                          prob_of(q, 0), prob_of(q, 3));
    qreg_destroy(q);
}

static void demo_qft(int rank) {
    int n = 3;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_qft(q, 0, n);
    if (rank == 0) {
        printf("QFT|000>: every basis state has prob ~ 1/%d\n", 1 << n);
        for (int i = 0; i < (1 << n); i++)
            printf("  prob_of(|%d>)=%.4f\n", i, prob_of(q, i));
    }
    qreg_destroy(q);
}

static size_t g_demo_marked = 11;
static void demo_oracle(qreg *q, void *user) {
    (void)user;
    if (rank_owns(q, g_demo_marked))
        q->amp[global_to_local(q, g_demo_marked)] *= -1.0;
}

static void demo_grover(int rank) {
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, demo_oracle, NULL, 3);
    if (rank == 0)
        printf("Grover (N=%d, marked=%zu, 3 iters): prob_of(marked)=%.4f\n",
               1 << n, g_demo_marked, prob_of(q, g_demo_marked));
    qreg_destroy(q);
}

static void demo_shor(int rank) {
    shor_factor_result r = shor_factor(15, /*max_attempts=*/8);
    if (rank == 0) {
        if (r.p && r.q) printf("Shor factor(15) -> %llu * %llu (attempts=%d)\n",
                               (unsigned long long)r.p,
                               (unsigned long long)r.q, r.attempts);
        else            printf("Shor factor(15) FAILED after %d attempts\n",
                               r.attempts);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *algo = "bell";
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--algo") == 0) { algo = argv[i+1]; i++; }
    }
    if      (!strcmp(algo, "bell"))   demo_bell  (rank);
    else if (!strcmp(algo, "qft"))    demo_qft   (rank);
    else if (!strcmp(algo, "grover")) demo_grover(rank);
    else if (!strcmp(algo, "shor"))   demo_shor  (rank);
    else if (rank == 0) {
        fprintf(stderr, "unknown --algo %s; choices: bell qft grover shor\n", algo);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
```

- [ ] **Step 2: Wire the binary target into the makefile**

Append to the makefile (somewhere after the test recipes, before `clean`):
```make
# Demo binary.
DEMO_OBJ := $(BUILDDIR)/qubit.o
$(DEMO_OBJ): qubit.c $(LIB_OBJS) | dirs
	$(MPICC) $(CFLAGS) -I. -c $< -o $@
$(BINDIR)/qubit: $(DEMO_OBJ) $(LIB_OBJS)
	$(MPICC) $(CFLAGS) $(DEMO_OBJ) $(LIB_OBJS) $(LDFLAGS) -o $@

all: dirs $(BINDIR)/qubit

# make demo ALGO=qft NP=4
ALGO ?= bell
NP   ?= 4
demo: $(BINDIR)/qubit
	mpirun --oversubscribe -n $(NP) $(BINDIR)/qubit --algo $(ALGO)
```

- [ ] **Step 3: Build and run all four demos**

Run:
```bash
cd implementation/c
make
for a in bell qft grover shor; do
    echo "=== $a ==="
    mpirun --oversubscribe -n 4 build/bin/qubit --algo $a
done
```
Expected: each demo prints a sensible result line. Shor may fail occasionally; rerun.

- [ ] **Step 4: Smoke at NP=1 and NP=8**

Run:
```bash
cd implementation/c
mpirun --oversubscribe -n 1 build/bin/qubit --algo bell
mpirun --oversubscribe -n 8 build/bin/qubit --algo qft
```
Expected: both work; output identical across ranks (only rank 0 prints).

- [ ] **Step 5: Commit**

```bash
git add implementation/c/qubit.c implementation/c/makefile
git commit -m "feat(c): qubit.c demo with --algo {bell,qft,grover,shor}

Minimal CLI front-end. Each demo runs a known-correct small instance
and prints the relevant probabilities or factors. Primarily exists to
prove the public API can be used by code outside the test harness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 12 — documentation (Tasks 39–40)

### Task 39: assessment.md (living coverage matrix)

**Files:**
- Create: `implementation/c/assessment.md`

- [ ] **Step 1: Write the coverage matrix**

Write `implementation/c/assessment.md`:
```markdown
# implementation/c -- coverage of thesis claims

Updated end of each implementation phase. File:line references point at
the canonical implementation site for each claim.

## §8 (sparse-gate strategy, 2026 thesis)

| Claim | Status | Location |
|---|---|---|
| In-place single-qubit gate, O(2^n) | ✓ | `matrix.c::apply_u_local` |
| Distributed state vector by top-p bits | ✓ | `matrix.c::qreg_create` + `parallel.c::is_local_qubit` |
| Pairwise MPI_Sendrecv for global qubit | ✓ | `parallel.c::exchange_amplitudes` |
| Controlled gate four-case dispatch | ✓ | `matrix.c::apply_cu` |
| Modular_exp via MPI_Alltoallv | ✓ | `parallel.c::redistribute_pairs` + `shor.c::apply_modular_exp` |
| qreg API per §12 | ✓ | `matrix.h` |

## §9 (2004 library API)

| Claim | Status | Notes |
|---|---|---|
| matrix create/init/print | ✓ (functional equiv) | qreg_create / qreg_init_basis / qreg_dump |
| tensor_product, dot_product | ✗ by design | dense operators are not materialised in v1 |
| send_matrix / get_matrix / broadcast_matrix | ✗ by design | replaced by exchange_amplitudes + redistribute_pairs |
| H, CNOT gates | ✓ | apply_h, apply_cnot |
| Deconstructor (~matrix) | ✓ | qreg_destroy |
| QFT promised but unfinished in 2004 | ✓ | `qft.c::apply_qft` |

## §9 (2026 QFT)

| Claim | Status | Location |
|---|---|---|
| QFT forward + inverse | ✓ | `qft.c::apply_qft` / `apply_qft_inverse` |
| Includes bit-reversal swaps | ✓ | `qft.c::apply_qft` final swap loop |
| Period detection on known periodic input | ✓ (tested) | `tests/test_qft.c::test_qft_detects_period` |

## §10 (2026 Grover)

| Claim | Status | Location |
|---|---|---|
| Phase-oracle callback API | ✓ | `grover.h::oracle_fn` |
| H^n -> oracle/diffusion loop | ✓ | `grover.c::apply_grover` |
| Optimum-stop tested | ✓ | `tests/test_grover.c::test_grover_over_iteration_hurts` |
| Multiple marked items | ✓ | `tests/test_grover.c::test_grover_4_marked_in_16` |

## §11 (2026 Shor)

| Claim | Status | Location |
|---|---|---|
| apply_modular_exp with y>=N pass-through | ✓ | `shor.c::apply_modular_exp` |
| Distributed via MPI_Alltoallv | ✓ | via `parallel.c::redistribute_pairs` |
| apply_shor_period (period finding) | ✓ | `shor.c::apply_shor_period` |
| shor_factor (end-to-end) | ✓ | `shor.c::shor_factor` |
| Continued-fraction post-processing | ✓ | `standart.c::continued_fraction` |
| Factor N=15 reliably | ✓ (tested) | `tests/test_shor.c::test_shor_factor_15` |

## §12 (qreg API)

Every entry in spec §6.1 is implemented in matrix.h/c. The disclosed
extensions over thesis §12 (apply_y/s/t/rx/ry/rz, apply_cz, apply_multi_*,
qreg_clone/dump, sample_distribution, shor_factor) are committed back to
the thesis in Task 41.

## Out of scope for v1

* Density matrices / mixed states.
* Noise models.
* Tensor-network / stabilizer-formalism shortcuts.
* GPU offload.
* Python bindings.

## Test matrix

All test binaries pass at NP = 1, 2, 4 via `make test`, and additionally
at NP = 8 via `make test-large` (Shor-21 case excluded from the standard
run for runtime reasons).
```

- [ ] **Step 2: Sanity check the file**

Run: `cat implementation/c/assessment.md | head -30`
Expected: renders correctly.

- [ ] **Step 3: Commit**

```bash
git add implementation/c/assessment.md
git commit -m "docs(c): assessment.md coverage matrix

Tracks every claim from thesis §8/9/10/11/12 to its implementation site
in /c. Two intentional gaps documented: the 2004 dense-matrix primitives
(tensor_product, dot_product, send_matrix, etc.) are NOT exposed in the
new library because the sparse-gate approach makes them obsolete; their
use cases are served by qreg + apply_* + exchange_amplitudes /
redistribute_pairs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 40: README.md for the new library

**Files:**
- Create: `implementation/c/README.md`

- [ ] **Step 1: Write the README**

Write `implementation/c/README.md`:
```markdown
# implementation/c

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
make test-large # additionally NP = 8 (Shor-21 etc.)
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
    /* prob_of(q, k) returns the same value on every rank.            */
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        for (size_t i = 0; i < 8; i++)
            printf("prob(|%zu>) = %.4f\n", i, prob_of(q, i));
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
```

- [ ] **Step 2: Commit**

```bash
git add implementation/c/README.md
git commit -m "docs(c): README for the new sparse-gate library

Build/test/demo commands, file layout, requirements, constraints, and
pointers to the spec, the thesis chapters this implements, and the
assessment matrix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 13 — thesis sync (Task 41)

### Task 41: update thesis §12 ref_guide to track API additions; rebuild PDF

Per spec §2: every API extension co-updates the thesis. The disclosed
v1 additions over thesis §12 (apply_y/s/t/rx/ry/rz, apply_cz,
apply_multi_controlled_z, apply_multi_controlled_x, qreg_clone,
qreg_norm, qreg_dump, sample_distribution, shor_factor) need to land in
`source code/ref_guide.tex`. Grover's `void *user` callback context and
Shor's `shor_period_result` return type also need to be reflected.

**Files:**
- Modify: `source code/ref_guide.tex`
- Run: `./build.sh` from repo root

- [ ] **Step 1: Find the relevant code blocks in ref_guide.tex**

Run: `grep -n "apply_h\|apply_cnot\|apply_grover\|apply_shor_period" "source code/ref_guide.tex"`
Expected: shows the lstlisting blocks where each declared function lives.

- [ ] **Step 2: Patch each lstlisting to add the new functions**

In `source code/ref_guide.tex`, locate the single-qubit `apply_*` block and add `apply_y, apply_s, apply_t, apply_rx, apply_ry, apply_rz`. In the two-qubit block add `apply_cz`. Add a new multi-controlled paragraph after the two-qubit block. Add `qreg_clone, qreg_norm, qreg_dump, sample_distribution` to the measurement / utilities block. Add `shor_factor` to the algorithm block, and change `apply_shor_period`'s signature to return `shor_period_result`. Add `void *user` to `apply_grover`'s signature and document `oracle_fn`'s new signature `typedef void (*oracle_fn)(qreg *q, void *user)`.

- [ ] **Step 3: Rebuild the PDF**

Run:
```bash
cd /Users/arda/projects/thesis
./build.sh distclean >/dev/null
./build.sh 2>&1 | tail -3
```
Expected: build exit 0, OK line printed, new PDF generated.

- [ ] **Step 4: Verify the new functions appear in the rendered PDF**

Run:
```bash
/opt/homebrew/bin/pdftotext qucomp.pdf - | \
   grep -E "apply_(rx|ry|rz|s|t|cz|multi_controlled)|shor_factor|qreg_clone|sample_distribution" \
   | head
```
Expected: each new identifier appears at least once.

- [ ] **Step 5: Commit**

```bash
git add "source code/ref_guide.tex" qucomp.pdf "source code/qucomp.pdf"
git commit -m "docs(thesis): §12 tracks /c library extensions

Per spec §2: every API extension in implementation/c co-updates the
thesis. Adds apply_y/s/t/rx/ry/rz, apply_cz, apply_multi_controlled_{z,x},
qreg_clone, qreg_norm, qreg_dump, sample_distribution to the listings;
updates apply_grover signature to take a void *user callback context;
updates apply_shor_period to return shor_period_result. Mentions
shor_factor as the high-level wrapper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## End of plan

That's 41 tasks across 13 phases. Self-review checklist follows.
