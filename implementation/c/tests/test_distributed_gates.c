#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "parallel.h"
#include "standart.h"
#include "unity/unity.h"
#include "test_assert.h"
#include "test_runner.h"

static int g_rank, g_size, g_p;

void setUp(void)    {}
void tearDown(void) {}

/* Helper: gather the full state vector to every rank, return a heap
 * buffer of length 2^n_qubits. Caller frees. Wraps the MPI_Allgather. */
static complex double *gather_full(qreg *q) {
    size_t total = (size_t)1 << q->n_qubits;
    complex double *full = malloc(total * sizeof *full);
    MPI_Allgather(q->amp, (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
                  full,    (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
                  q->comm);
    return full;
}

/* The canonical |Phi+> Bell state on the (control, target) qubit pair  *
 * within an n-qubit register that started in basis |0...0>:            *
 *   amplitudes 1/sqrt(2) on indices 0 and (1<<c | 1<<t); 0 elsewhere.  */
static void assert_bell_state_on_pair(complex double *full,
                                      int n, int control, int target) {
    size_t total = (size_t)1 << n;
    double s = 1.0 / sqrt(2.0);
    size_t bell_idx = ((size_t)1 << control) | ((size_t)1 << target);
    for (size_t i = 0; i < total; i++) {
        if (i == 0)         ASSERT_NEAR_AMP(s + 0.0*I, full[i]);
        else if (i == bell_idx) ASSERT_NEAR_AMP(s + 0.0*I, full[i]);
        else                ASSERT_NEAR_AMP(0.0 + 0.0*I, full[i]);
    }
}

/* Build a |Phi+>-style Bell state via H on `control`, then CNOT(c, t),
 * starting from |0...0>. Returns the qreg (caller destroys).            */
static qreg *make_bell(int n_qubits, int control, int target) {
    qreg *q = qreg_create(n_qubits, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, control);
    apply_cnot(q, control, target);
    return q;
}

/* We choose n based on g_p = log2(NP) so each test exercises the    *
 * advertised locality split at every supported NP (1, 2, 4, 8...). *
 * In particular:                                                    *
 *   - tests (a)..(c) need n >= g_p + 2 so qubit 0 stays local while *
 *     qubit n-1 stays global;                                       *
 *   - test  (d)    needs n >= g_p + 1 with g_p >= 2 (NP >= 4) so    *
 *     qubits n-1 and n-2 are both in the global region.             */

/* ---- (a) Single-qubit H on a global qubit ---- */

static void test_h_on_global_qubit_makes_uniform_pair(void) {
    if (g_size < 2) { TEST_PASS(); return; }
    int n = g_p + 2;                              /* >= 3 at NP=2 */
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    int target = n - 1;                           /* always global */
    TEST_ASSERT_FALSE(is_local_qubit(q, target));
    qreg_init_basis(q, 0);
    apply_h(q, target);
    complex double *full = gather_full(q);
    /* Expected:  (1/sqrt(2)) (|0> + |2^target>) */
    size_t tval = (size_t)1 << target;
    double s = 1.0 / sqrt(2.0);
    for (size_t i = 0; i < ((size_t)1 << n); i++) {
        if (i == 0)         ASSERT_NEAR_AMP(s, full[i]);
        else if (i == tval) ASSERT_NEAR_AMP(s, full[i]);
        else                ASSERT_NEAR_AMP(0.0, full[i]);
    }
    free(full);
    qreg_destroy(q);
}

/* ---- (b) CNOT control local, target global ---- */

static void test_cnot_c_local_t_global_bell(void) {
    if (g_size < 2) { TEST_PASS(); return; }
    int n = g_p + 2;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    /* control = 0 (always local at n >= p+1), target = n-1 (always global). */
    int control = 0, target = n - 1;
    TEST_ASSERT_TRUE (is_local_qubit(q, control));
    TEST_ASSERT_FALSE(is_local_qubit(q, target));
    qreg_destroy(q);
    q = make_bell(n, control, target);
    complex double *full = gather_full(q);
    assert_bell_state_on_pair(full, n, control, target);
    free(full);
    qreg_destroy(q);
}

/* ---- (c) CNOT control global, target local ---- */

static void test_cnot_c_global_t_local_bell(void) {
    if (g_size < 2) { TEST_PASS(); return; }
    int n = g_p + 2;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    int control = n - 1, target = 0;
    TEST_ASSERT_FALSE(is_local_qubit(q, control));
    TEST_ASSERT_TRUE (is_local_qubit(q, target));
    qreg_destroy(q);
    q = make_bell(n, control, target);
    complex double *full = gather_full(q);
    assert_bell_state_on_pair(full, n, control, target);
    free(full);
    qreg_destroy(q);
}

/* ---- (d) CNOT control global, target global (THE bug case) ---- */

static void test_cnot_both_global_bell(void) {
    /* Need at least two global qubits, i.e. p >= 2, i.e. NP >= 4. */
    if (g_size < 4) { TEST_PASS(); return; }
    int n = g_p + 2;                              /* >= 4 at NP=4 */
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    /* Top two qubits are always global. */
    int control = n - 1, target = n - 2;
    TEST_ASSERT_FALSE(is_local_qubit(q, control));
    TEST_ASSERT_FALSE(is_local_qubit(q, target));
    qreg_destroy(q);
    q = make_bell(n, control, target);
    complex double *full = gather_full(q);
    assert_bell_state_on_pair(full, n, control, target);
    free(full);
    qreg_destroy(q);
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    g_p = ilog2_u32((uint32_t)g_size);
    RUN_TEST(test_h_on_global_qubit_makes_uniform_pair);
    RUN_TEST(test_cnot_c_local_t_global_bell);
    RUN_TEST(test_cnot_c_global_t_local_bell);
    RUN_TEST(test_cnot_both_global_bell);
}

TEST_RUNNER_MAIN()
