/* qubit.c -- demo entry point.
 *
 * Usage:  mpirun -n NP build/bin/qubit --algo {bell|qft|grover|shor} [args]
 *
 * Each --algo runs a small, well-known instance and prints the result.
 * Built primarily as a smoke harness for the library.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "parallel.h"
#include "qft.h"
#include "grover.h"
#include "shor.h"

static void demo_bell(int rank) {
    qreg *q = qreg_create(2, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_h(q, 0);
    apply_cnot(q, 0, 1);
    /* prob_of is a collective; every rank must call it. */
    double p00 = prob_of(q, 0);
    double p11 = prob_of(q, 3);
    if (rank == 0) printf("Bell |Phi+>: prob_of(|00>)=%.4f  prob_of(|11>)=%.4f\n",
                          p00, p11);
    qreg_destroy(q);
}

static void demo_qft(int rank) {
    int n = 3;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_qft(q, 0, n);
    /* prob_of is a collective; every rank must call it. */
    double probs[8];
    for (int i = 0; i < (1 << n); i++) probs[i] = prob_of(q, i);
    if (rank == 0) {
        printf("QFT|000>: every basis state has prob ~ 1/%d\n", 1 << n);
        for (int i = 0; i < (1 << n); i++)
            printf("  prob_of(|%d>)=%.4f\n", i, probs[i]);
    }
    qreg_destroy(q);
}

static size_t g_demo_marked = 11;
static void demo_oracle(qreg *q, void *user) {
    (void)user;
    if (rank_owns(q, g_demo_marked))
        q->amp[global_to_local(q, g_demo_marked)] *= -1.0;
}

static void demo_grover(int rank) {
    int n = 4;
    qreg *q = qreg_create(n, MPI_COMM_WORLD);
    qreg_init_basis(q, 0);
    apply_grover(q, n, demo_oracle, NULL, 3);
    /* prob_of is a collective; every rank must call it. */
    double pm = prob_of(q, g_demo_marked);
    if (rank == 0)
        printf("Grover (N=%d, marked=%zu, 3 iters): prob_of(marked)=%.4f\n",
               1 << n, g_demo_marked, pm);
    qreg_destroy(q);
}

static void demo_shor(int rank) {
    shor_factor_result r = shor_factor(15, /*max_attempts=*/8);
    if (rank == 0) {
        if (r.p && r.q) printf("Shor factor(15) -> %llu * %llu (attempts=%d)\n",
                               (unsigned long long)r.p,
                               (unsigned long long)r.q, r.attempts);
        else            printf("Shor factor(15) FAILED after %d attempts\n",
                               r.attempts);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *algo = "bell";
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--algo") == 0) { algo = argv[i+1]; i++; }
    }
    if      (!strcmp(algo, "bell"))   demo_bell  (rank);
    else if (!strcmp(algo, "qft"))    demo_qft   (rank);
    else if (!strcmp(algo, "grover")) demo_grover(rank);
    else if (!strcmp(algo, "shor"))   demo_shor  (rank);
    else if (rank == 0) {
        fprintf(stderr, "unknown --algo %s; choices: bell qft grover shor\n", algo);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
