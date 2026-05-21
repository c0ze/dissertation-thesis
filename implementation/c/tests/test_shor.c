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

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_modular_exp_passes_through_y_ge_N);
    RUN_TEST(test_modular_exp_maps_within_ring);
}

TEST_RUNNER_MAIN()
