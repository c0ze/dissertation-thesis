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

static void test_qft_detects_period(void) {
    /* Prepare (1/sqrt(4)) (|0> + |2> + |4> + |6>). After QFT^-1, mass
     * concentrates on multiples of N/r = 8/2 = 4: |0> and |4>, each with
     * probability 0.5.                                                    */
    int n = 3;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
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

/* ---- CI-additions: norm preservation, larger n, linearity ---- */

static void test_qft_preserves_norm_on_basis_state(void) {
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 9);            /* arbitrary non-zero basis state */
    apply_qft(q, 0, n);
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}

static void test_qft_on_4_qubits_uniform_from_zero(void) {
    /* QFT |0...0> on n=4 yields the uniform superposition over 16
     * basis states, each with prob 1/16. Distinct from the 3-qubit
     * test, this exercises an extra round of controlled-phase nesting. */
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_qft(q, 0, n);
    double expected = 1.0 / (double)(1 << n);
    for (size_t i = 0; i < (size_t)(1 << n); i++) {
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, expected, prob_of(q, i));
    }
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}

static void test_qft_round_trip_on_basis_states(void) {
    /* Run QFT-then-inverse on several different basis inputs and
     * confirm each returns to its starting state. Spot-checks that the
     * inverse covers more than the one shot in test_qft_round_trip.   */
    int n = 4;
    size_t bases[] = {0, 1, 3, 7, 10, 15};
    for (size_t b = 0; b < sizeof(bases)/sizeof(bases[0]); b++) {
        qreg *q = qreg_create(n, MPI_COMM_WORLD);
        qreg_init_basis(q, bases[b]);
        apply_qft        (q, 0, n);
        apply_qft_inverse(q, 0, n);
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, bases[b]));
        for (size_t i = 0; i < (size_t)(1 << n); i++) {
            if (i == bases[b]) continue;
            TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 0.0, prob_of(q, i));
        }
        qreg_destroy(q);
    }
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_qft_on_1_qubit_equals_h);
    RUN_TEST(test_qft_of_zero_is_uniform);
    RUN_TEST(test_qft_round_trip);
    RUN_TEST(test_qft_detects_period);
    RUN_TEST(test_qft_preserves_norm_on_basis_state);
    RUN_TEST(test_qft_on_4_qubits_uniform_from_zero);
    RUN_TEST(test_qft_round_trip_on_basis_states);
}

TEST_RUNNER_MAIN()
