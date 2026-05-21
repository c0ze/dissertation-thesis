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

static void test_redistribute_pairs_round_trip(void) {
    /* Each rank produces a list of (global_index, amplitude) pairs whose
     * destination is uniformly distributed across all ranks. After the
     * Alltoallv, every rank should hold exactly the pairs destined for
     * it, with the amplitudes intact.                                    */
    qreg *q = qreg_create(4, MPI_COMM_WORLD);
    size_t total = (size_t)1 << q->n_qubits;
    /* Each rank emits 4 pairs: indices base, base+1, base+2, base+3 with
     * base = rank * 4, values (rank+1.0)*100 + i.                        */
    size_t n_pairs = 4;
    size_t *idx = malloc(n_pairs * sizeof *idx);
    complex double *val = malloc(n_pairs * sizeof *val);
    for (size_t i = 0; i < n_pairs; i++) {
        idx[i] = ((size_t)q->rank * 4 + i) % total;
        val[i] = (double)((q->rank + 1) * 100 + (int)i) + 0.0*I;
    }
    /* Zero the qreg, then accumulate the redistributed pairs into it. */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    redistribute_pairs(q, idx, n_pairs, val);
    /* Every global index from 0 .. (n_pairs * n_procs - 1) modulo total
     * should now have exactly one amplitude written. Verify the local
     * slice has expected values.                                          */
    size_t base = (size_t)q->rank * q->local_size;
    for (size_t off = 0; off < q->local_size; off++) {
        size_t g = base + off;
        complex double expected = 0.0;
        /* Find which (rank, i) produced this g. */
        for (int r = 0; r < q->n_procs; r++) {
            for (size_t i = 0; i < n_pairs; i++) {
                if (((size_t)r * 4 + i) % total == g) {
                    expected += (double)((r + 1) * 100 + (int)i);
                }
            }
        }
        ASSERT_NEAR_AMP(expected, q->amp[off]);
    }
    free(idx); free(val);
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_locality_classification);
    RUN_TEST(test_partner_for_global_qubit);
    RUN_TEST(test_global_local_round_trip);
    RUN_TEST(test_exchange_round_trip);
    RUN_TEST(test_redistribute_pairs_round_trip);
}

TEST_RUNNER_MAIN()
