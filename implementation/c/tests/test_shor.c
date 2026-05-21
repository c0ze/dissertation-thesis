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

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_modular_exp_passes_through_y_ge_N);
    RUN_TEST(test_modular_exp_maps_within_ring);
    RUN_TEST(test_shor_period_a7_mod15);
    RUN_TEST(test_shor_factor_15);
}

TEST_RUNNER_MAIN()
