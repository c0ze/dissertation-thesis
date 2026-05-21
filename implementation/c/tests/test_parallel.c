#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "parallel.h"
#include "unity/unity.h"
#include "test_assert.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

static void test_locality_classification(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    /* With 4 qubits, p = log2(n_procs).                                 *
     * - At NP=1: every qubit is local.                                  *
     * - At NP=2: qubits 0..2 local, qubit 3 global.                     *
     * - At NP=4: qubits 0,1 local; qubits 2,3 global.                   */
    for (int k = 0; k < q->n_qubits - q->p; k++)
        TEST_ASSERT_TRUE(is_local_qubit(q, k));
    for (int k = q->n_qubits - q->p; k < q->n_qubits; k++)
        TEST_ASSERT_FALSE(is_local_qubit(q, k));
    qreg_destroy(q);
}

static void test_partner_for_global_qubit(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    if (q->n_procs >= 2) {
        /* The top-most global qubit (n-1) partners by toggling bit (p-1). */
        int top_global = q->n_qubits - 1;
        int expected_partner = q->rank ^ (1 << (top_global - (q->n_qubits - q->p)));
        TEST_ASSERT_EQUAL_INT(expected_partner, partner_for(q, top_global));
    }
    qreg_destroy(q);
}

static void test_global_local_round_trip(void) {
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    /* For each global index this rank owns, global -> local -> global must round-trip. */
    size_t base = (size_t)q->rank * q->local_size;
    for (size_t off = 0; off < q->local_size; off++) {
        size_t g = base + off;
        TEST_ASSERT_TRUE(rank_owns(q, g));
        TEST_ASSERT_EQUAL_size_t(off, global_to_local(q, g));
        TEST_ASSERT_EQUAL_size_t(g,   local_to_global(q, off));
    }
    qreg_destroy(q);
}

static void test_exchange_round_trip(void) {
    /* Skip in single-process mode -- nothing to exchange with. */
    if (g_size == 1) {
        TEST_PASS();
        return;
    }
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    /* Fill local slice with a known per-rank, per-index pattern. */
    for (size_t i = 0; i < q->local_size; i++) {
        q->amp[i] = (double)(q->rank * 1000 + (int)i) + 0.5*I;
    }
    /* Partner along the most-significant global qubit. */
    int k       = q->n_qubits - 1;
    int partner = partner_for(q, k);
    /* Snapshot what we expect to receive: the partner's pattern. */
    complex double *expected = malloc(q->local_size * sizeof *expected);
    for (size_t i = 0; i < q->local_size; i++) {
        expected[i] = (double)(partner * 1000 + (int)i) + 0.5*I;
    }
    /* Snapshot what we expect to keep in q->amp afterwards: ours unchanged. */
    complex double *ours = malloc(q->local_size * sizeof *ours);
    memcpy(ours, q->amp, q->local_size * sizeof *ours);

    complex double *recv = malloc(q->local_size * sizeof *recv);
    exchange_amplitudes(q, partner, recv);
    for (size_t i = 0; i < q->local_size; i++) {
        ASSERT_NEAR_AMP(expected[i], recv[i]);
        ASSERT_NEAR_AMP(ours[i],     q->amp[i]);
    }
    free(expected); free(ours); free(recv);
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_locality_classification);
    RUN_TEST(test_partner_for_global_qubit);
    RUN_TEST(test_global_local_round_trip);
    RUN_TEST(test_exchange_round_trip);
}

TEST_RUNNER_MAIN()
