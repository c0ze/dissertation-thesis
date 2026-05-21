#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>     /* getenv */
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
     * Layout: 6 total qubits = 1 counting + 5 target. counting at      *
     * bits [0..0], target at bits [1..5]. N = 5.                       *
     * Place amplitude at (x=0, y=10). 10 >= 5 so should stay.          */
    int counting_start = 0, t = 1;
    int target_start   = 1, n = 5;
    int n_total = 6;
    qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
    qreg_init_basis(q, ((size_t)10 << 1));
    apply_modular_exp(q, counting_start, t, target_start, n, /*a=*/2, /*N=*/5);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, (size_t)10 << 1));
    qreg_destroy(q);
}

static void test_modular_exp_maps_within_ring(void) {
    /* y=1, x=1, a=2, N=5 -> y_new = (2^1 * 1) mod 5 = 2.
     * Start at (x=1, y=1) -> global = (1<<1) | 1 = 3.
     * After: (x=1, y=2) -> global = (2<<1) | 1 = 5.                     */
    int counting_start = 0, t = 1;
    int target_start   = 1, n = 5;
    int n_total = 6;
    qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
    qreg_init_basis(q, ((size_t)1 << 1) | 1);
    apply_modular_exp(q, counting_start, t, target_start, n, 2, 5);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, ((size_t)2 << 1) | 1));
    qreg_destroy(q);
}

static void test_shor_period_a7_mod15(void) {
    /* Order of 7 mod 15 is 4. Use t=8 counting + n=4 target = 12 qubits.
     * That's 2^12 = 4096 amplitudes, comfortable.                        */
    int n = 4;
    int t = 8;
    int n_total = t + n;
    if (n_total > 14) { TEST_PASS(); return; }
    qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
    shor_period_result res = apply_shor_period(q, /*cs=*/n, t, /*ts=*/0, n,
                                               /*a=*/7, /*N=*/15);
    /* Recovered period must be a divisor of true r=4 OR equal to it. */
    TEST_ASSERT_TRUE_MESSAGE(res.r != 0, "Shor period finder returned 0");
    TEST_ASSERT_TRUE_MESSAGE(res.r == 4 || res.r == 2 || res.r == 1,
        "Shor period finder returned an unexpected period");
    qreg_destroy(q);
}

static void test_shor_factor_15(void) {
    shor_factor_result r = shor_factor(15, /*max_attempts=*/8);
    TEST_ASSERT_TRUE_MESSAGE(r.p != 0 && r.q != 0,
        "shor_factor(15) failed after 8 attempts");
    TEST_ASSERT_EQUAL_UINT64(15ULL, r.p * r.q);
    TEST_ASSERT_TRUE(r.p > 1 && r.q > 1);
    TEST_ASSERT_TRUE(r.p == 3 || r.p == 5);
}

/* ---- CI-additions: orbit table, second a value, repeated factor ---- */

static void test_modular_exp_orbit_a2_mod5(void) {
    /* a=2, N=5: 2^0=1, 2^1=2, 2^2=4, 2^3=3 (mod 5). Place a unit
     * amplitude at each (x, y=1) and verify it lands at (x, 2^x mod 5).*/
    int counting_start = 0, t = 2;     /* x in [0..3]                  */
    int target_start   = 2, n = 3;     /* y in [0..7] (>=5 reserved)    */
    int n_total = 5;
    uint64_t expected_y[4] = {1, 2, 4, 3};
    for (int x = 0; x < 4; x++) {
        qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
        size_t initial = ((size_t)1 << target_start) | ((size_t)x << counting_start);
        qreg_init_basis(q, initial);
        apply_modular_exp(q, counting_start, t, target_start, n, 2, 5);
        size_t final = ((size_t)expected_y[x] << target_start)
                     | ((size_t)x            << counting_start);
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, final));
        qreg_destroy(q);
    }
}

static void test_shor_period_a4_mod15(void) {
    /* Order of 4 mod 15 is 2 (4^2 = 16 = 1 mod 15). Smaller register:
     * t=4 counting + n=4 target = 8 qubits. Continued-fraction
     * recovery of r=2 is essentially certain at t=4 (2^4 = 16 >> r^2=4). */
    int n = 4, t = 4;
    qreg *q = qreg_create(t + n, MPI_COMM_WORLD);
    shor_period_result res = apply_shor_period(q, /*cs=*/n, t, /*ts=*/0, n,
                                               /*a=*/4, /*N=*/15);
    TEST_ASSERT_TRUE_MESSAGE(res.r != 0, "period finder returned 0");
    TEST_ASSERT_TRUE_MESSAGE(res.r == 2 || res.r == 1,
        "expected recovered period to divide 2");
    qreg_destroy(q);
}

static void test_shor_factor_15_repeated(void) {
    /* Run shor_factor(15) several times; each run should succeed
     * within 8 attempts and always return {3, 5} (in either order).
     * Catches regressions where one of the period-finding paths
     * silently breaks under a particular random a.                     */
    for (int trial = 0; trial < 3; trial++) {
        shor_factor_result r = shor_factor(15, /*max_attempts=*/8);
        TEST_ASSERT_TRUE_MESSAGE(r.p != 0 && r.q != 0,
            "shor_factor(15) failed within 8 attempts");
        TEST_ASSERT_EQUAL_UINT64(15ULL, r.p * r.q);
        TEST_ASSERT_TRUE((r.p == 3 && r.q == 5) || (r.p == 5 && r.q == 3));
    }
}

/* 16-qubit register (n=5 target + t=11 counting). Benchmarks at
 * ~6-20 ms per call on Apple Silicon depending on NP. Gated behind
 * the RUN_SHOR_21 environment variable so `make test` (the common-
 * case CI loop) stays uncoupled from algorithm-end tests; `make
 * test-large` sets it.
 *
 * Test uses apply_shor_period with a FIXED a=2 (not the high-level
 * shor_factor which picks a randomly) so the test is deterministic
 * apart from the inherent measurement randomness inside QFT readout.
 * The true period of 2 mod 21 is 6, so the continued-fraction step
 * will recover one of its divisors {1, 2, 3, 6} -- any of these is
 * a passing result. This isolates the test from rand-induced flakes
 * that could come from shor_factor's random a-selection inner loop. */
static void test_shor_period_a2_mod21(void) {
    if (getenv("RUN_SHOR_21") == NULL) {
        TEST_PASS();   /* skip in the normal test path */
        return;
    }
    int n = 5;                /* ceil(log2 21) */
    int t = 11;               /* 2*n + 1, the standard Shor counting width */
    int n_total = t + n;
    qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
    shor_period_result res = apply_shor_period(q, /*cs=*/n, t, /*ts=*/0,
                                               n, /*a=*/2, /*N=*/21);
    qreg_destroy(q);
    TEST_ASSERT_TRUE_MESSAGE(res.r != 0,
        "Shor period finder returned r=0 (no recovery)");
    /* True period of 2 mod 21 is 6 (2,4,8,16,11,1). Continued fraction
     * will give r dividing 6. */
    TEST_ASSERT_TRUE_MESSAGE(
        res.r == 1 || res.r == 2 || res.r == 3 || res.r == 6,
        "expected recovered period to divide 6");
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_modular_exp_passes_through_y_ge_N);
    RUN_TEST(test_modular_exp_maps_within_ring);
    RUN_TEST(test_shor_period_a7_mod15);
    RUN_TEST(test_shor_factor_15);
    RUN_TEST(test_modular_exp_orbit_a2_mod5);
    RUN_TEST(test_shor_period_a4_mod15);
    RUN_TEST(test_shor_factor_15_repeated);
    RUN_TEST(test_shor_period_a2_mod21);
}

TEST_RUNNER_MAIN()
