#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include "matrix.h"
#include "parallel.h"
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
    if (rank_owns(q, g_marked)) {
        size_t off = global_to_local(q, g_marked);
        q->amp[off] = -q->amp[off];
    }
}

static void test_grover_1_marked_in_16(void) {
    /* N = 16 = 2^4, 1 marked item. Optimum iterations = floor(pi/4 * 4) = 3.
     * Analytical success prob after 3 iterations = sin^2(7*theta) where
     * sin(theta) = 1/4, ~ 0.961. Threshold 0.95.                          */
    int n = 4;
    g_marked = 11;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, oracle_single_marked, NULL, 3);
    double p = prob_of(q, g_marked);
    TEST_ASSERT_TRUE_MESSAGE(p >= 0.95, "Grover did not concentrate >= 0.95 on the marked item");
    qreg_destroy(q);
}

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
    /* N=16, M=4 -> theta = arcsin(sqrt(4/16)) = arcsin(0.5) = pi/6.
     * Optimum iterations = floor(pi/4 * sqrt(16/4)) = floor(pi/4 * 2)
     *                    = floor(pi/2) = 1.
     * Total prob on the marked subspace after k=1 should be ~ 1.0.        */
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, oracle_four_marked, NULL, 1);
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

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_grover_1_marked_in_16);
    RUN_TEST(test_grover_4_marked_in_16);
    RUN_TEST(test_grover_over_iteration_hurts);
}

TEST_RUNNER_MAIN()
