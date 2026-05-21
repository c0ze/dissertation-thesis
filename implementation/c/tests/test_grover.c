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

/* ---- CI-additions: zero-iteration, optimum formula, all-marked ---- */

static void test_grover_zero_iterations_is_uniform(void) {
    /* With 0 iterations apply_grover should leave the register in the
     * uniform superposition produced by the H^n pre-step.              */
    int n = 4;
    g_marked = 11;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, oracle_single_marked, NULL, 0);
    double expected = 1.0 / (double)(1 << n);
    for (size_t i = 0; i < (size_t)(1 << n); i++) {
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, expected, prob_of(q, i));
    }
    qreg_destroy(q);
}

static void test_grover_probability_matches_formula(void) {
    /* Analytical prob after k iterations on 1 marked item in N=16:
     *   p_k = sin^2((2k+1)*theta) with sin(theta) = 1/4.                *
     * Tolerance 1e-6 accounts for the ~50 gates of accumulated FP.     */
    int n = 4;
    g_marked = 5;
    double sin_theta = 0.25;
    double theta     = asin(sin_theta);
    int   k_values[] = {1, 2, 3};
    for (size_t i = 0; i < sizeof(k_values)/sizeof(k_values[0]); i++) {
        int k = k_values[i];
        qreg *q = qreg_create(n, MPI_COMM_WORLD);
        qreg_init_basis(q, 0);
        apply_grover(q, n, oracle_single_marked, NULL, k);
        double expected = pow(sin((2*k + 1) * theta), 2);
        double actual   = prob_of(q, g_marked);
        TEST_ASSERT_DOUBLE_WITHIN(1e-6, expected, actual);
        qreg_destroy(q);
    }
}

static void oracle_mark_all(qreg *q, void *user) {
    (void)user;
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = -q->amp[i];
}

static void test_grover_all_marked_one_iter(void) {
    /* If every state is marked, the oracle is a global phase and the
     * diffusion's effect on the uniform state is just to add a global
     * sign too. Probabilities should remain uniform after any number
     * of iterations.                                                   */
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, oracle_mark_all, NULL, 2);
    double expected = 1.0 / (double)(1 << n);
    for (size_t i = 0; i < (size_t)(1 << n); i++) {
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, expected, prob_of(q, i));
    }
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_grover_1_marked_in_16);
    RUN_TEST(test_grover_4_marked_in_16);
    RUN_TEST(test_grover_over_iteration_hurts);
    RUN_TEST(test_grover_zero_iterations_is_uniform);
    RUN_TEST(test_grover_probability_matches_formula);
    RUN_TEST(test_grover_all_marked_one_iter);
}

TEST_RUNNER_MAIN()
