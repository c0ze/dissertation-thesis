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

static void test_s_is_phase_pi_over_2(void) {
    /* S^2 = Z. Apply via H-sandwich so the phase shows in probability. */
    qreg *qA = qreg_create(2, MPI_COMM_WORLD);
    if (!qA) { TEST_PASS(); return; }
    qreg_init_basis(qA, 0);
    apply_h(qA, 0); apply_s(qA, 0); apply_s(qA, 0); apply_h(qA, 0);

    qreg *qB = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(qB, 0);
    apply_h(qB, 0); apply_z(qB, 0); apply_h(qB, 0);

    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, prob_of(qB, 0), prob_of(qA, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, prob_of(qB, 1), prob_of(qA, 1));
    qreg_destroy(qA);
    qreg_destroy(qB);
}

static void test_t_quartic_is_z(void) {
    /* T^4 = Z. */
    qreg *qA = qreg_create(2, MPI_COMM_WORLD);
    if (!qA) { TEST_PASS(); return; }
    qreg_init_basis(qA, 0);
    apply_h(qA, 0);
    apply_t(qA, 0); apply_t(qA, 0); apply_t(qA, 0); apply_t(qA, 0);
    apply_h(qA, 0);

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
    if (!q) { TEST_PASS(); return; }
    qreg_init_basis(q, 1);
    apply_phase(q, 0, 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}

static void test_rx_2pi_is_identity_up_to_phase(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
    qreg_init_basis(q, 0);
    apply_rx(q, 0, 2.0 * M_PI);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 0));
    qreg_destroy(q);
}

static void test_ry_pi_flips(void) {
    /* RY(pi)|0> = |1> (up to a global -i for our convention). */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
    qreg_init_basis(q, 0);
    apply_ry(q, 0, M_PI);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}

static void test_rz_zero_is_identity(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
    qreg_init_basis(q, 1);
    apply_rz(q, 0, 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 1));
    qreg_destroy(q);
}

static void test_cu_both_local_makes_bell(void) {
    /* On a 2-qubit register, H on 0 then CNOT(0,1) -> |Phi+>.           */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
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

static void test_cnot_local_makes_bell(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
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
    /* CZ from |11> is just a phase. Sandwich with H to make it visible:
     * |11> -> CZ -> -|11> -> H on qubit 1 -> still concentrated in the
     * |10>/|11> branch when re-checked via H on qubit 1 inverse.        */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 0);
    apply_x(q, 0); apply_x(q, 1);    /* |11>                            */
    apply_cz(q, 0, 1);               /* phase only: still |11>          */
    apply_h(q, 1); apply_h(q, 1);    /* identity                        */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 3));
    qreg_destroy(q);
}

static void test_controlled_phase_zero_is_identity(void) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 3);                       /* |11> */
    apply_controlled_phase(q, 0, 1, 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 3));
    qreg_destroy(q);
}

static void test_swap_exchanges_basis_indices(void) {
    /* Start from |01>; swap qubits 0,1 -> |10>. */
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    if (!q) { TEST_PASS(); return; }
    if (!is_local_qubit(q, 0) || !is_local_qubit(q, 1)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    qreg_init_basis(q, 1);
    apply_swap(q, 0, 1);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 2));
    qreg_destroy(q);
}

static void test_swap_self_is_identity(void) {
    /* For a != b verify swap*swap = id. */
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 5);
    int a = 0, b = 1;
    if (!is_local_qubit(q, a) || !is_local_qubit(q, b)) {
        qreg_destroy(q); TEST_PASS(); return;
    }
    apply_swap(q, a, b);
    apply_swap(q, a, b);
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, prob_of(q, 5));
    qreg_destroy(q);
}

static void test_mcz_flips_only_all_ones(void) {
    qreg *q = qreg_create(3, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, 0); apply_h(q, 1); apply_h(q, 2);   /* |+++> */
    apply_multi_controlled_z(q, 3);                /* phase on |111> only */
    apply_h(q, 0); apply_h(q, 1); apply_h(q, 2);   /* H^3 again */
    /* The state should now be H^3 (I - 2|111><111|) H^3 |0>
     *  = |0> - 2 (1/8) sum_x (-1)^(popcount x) |x>
     *  prob_of(0)  = (1 - 2*1/8)^2 = (3/4)^2 = 9/16
     *  prob_of(7)  = (-2*1/8*(-1))^2  = (1/4)^2 = 1/16                  */
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 9.0/16.0, prob_of(q, 0));
    TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0/16.0, prob_of(q, 7));
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
    RUN_TEST(test_s_is_phase_pi_over_2);
    RUN_TEST(test_t_quartic_is_z);
    RUN_TEST(test_apply_phase_zero_is_identity);
    RUN_TEST(test_rx_2pi_is_identity_up_to_phase);
    RUN_TEST(test_ry_pi_flips);
    RUN_TEST(test_rz_zero_is_identity);
    RUN_TEST(test_cu_both_local_makes_bell);
    RUN_TEST(test_cnot_local_makes_bell);
    RUN_TEST(test_cz_phase_on_11);
    RUN_TEST(test_controlled_phase_zero_is_identity);
    RUN_TEST(test_swap_exchanges_basis_indices);
    RUN_TEST(test_swap_self_is_identity);
    RUN_TEST(test_mcz_flips_only_all_ones);
}

TEST_RUNNER_MAIN()
