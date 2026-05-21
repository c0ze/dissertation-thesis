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

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_grover_1_marked_in_16);
}

TEST_RUNNER_MAIN()
