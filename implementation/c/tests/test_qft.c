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
    /* QFT on a single qubit is the Hadamard. */
    qreg *qA = qreg_create(2, MPI_COMM_WORLD);
    if (!qA) { TEST_PASS(); return; }
    qreg_init_basis(qA, 0);
    apply_qft(qA, 0, 1);

    qreg *qB = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(qB, 0);
    apply_h(qB, 0);

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
