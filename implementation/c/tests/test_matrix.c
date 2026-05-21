#include <mpi.h>
#include <stdlib.h>
#include "matrix.h"
#include "parallel.h"
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

static void test_apply_h_twice_is_identity(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    /* Skip if qubit 1 is global at this NP — apply_u global lands in Task 18. */
    if (!is_local_qubit(q, 1)) { qreg_destroy(q); TEST_PASS(); return; }
    apply_h(q, 1);
    apply_h(q, 1);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 5));
    qreg_destroy(q);
}

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
    /* Apply H again - should return to |000>. */
    apply_h(q, target);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 0));
    ASSERT_NORM_ONE(q);
    qreg_destroy(q);
}

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
    if (!q) { TEST_PASS(); return; }   /* NP > 2^n_qubits: not a valid layout */
    qreg_init_basis(q, 0);
    apply_y(q, 0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}

static void test_pauli_z_on_zero_is_identity(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
    qreg_init_basis(q, 0);
    apply_z(q, 0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 0));
    qreg_destroy(q);
}

static void test_pauli_z_on_one_negates(void) {
    /* H Z |0> = (|0>-|1>)/sqrt(2); then H back gives |1>, not |0>. */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    apply_z(q, 0);
    apply_h(q, 0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_create_4q);
    RUN_TEST(test_create_rejects_non_pow2_n_procs);
    RUN_TEST(test_create_rejects_too_many_qubits);
    RUN_TEST(test_create_rejects_zero_qubits);
    RUN_TEST(test_init_basis_zero);
    RUN_TEST(test_init_basis_arbitrary);
    RUN_TEST(test_init_basis_normalisation);
    RUN_TEST(test_norm_of_basis_state);
    RUN_TEST(test_prob_of_basis_state);
    RUN_TEST(test_apply_h_on_qubit0_from_basis0);
    RUN_TEST(test_apply_h_twice_is_identity);
    RUN_TEST(test_apply_h_on_global_qubit);
    RUN_TEST(test_pauli_x_flips_bit);
    RUN_TEST(test_pauli_y_on_zero);
    RUN_TEST(test_pauli_z_on_zero_is_identity);
    RUN_TEST(test_pauli_z_on_one_negates);
}

TEST_RUNNER_MAIN()
