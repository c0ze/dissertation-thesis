#ifndef MATRIX_H
#define MATRIX_H

#include <complex.h>
#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Per spec §4.3:  hard cap on n_qubits keeps every  1ULL<<k  shift   *
 * well-defined on 64-bit systems and leaves 4 bits of headroom over  *
 * size_t arithmetic.                                                  */
#define QREG_MAX_QUBITS 60

/* Always-on assert. Survives -DNDEBUG. Uses MPI_Abort so failures on  *
 * one rank do not leave the others hanging in collective calls.       */
#define QREG_ASSERT(cond, msg)                                              \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr,                                                 \
                "QREG_ASSERT failed at %s:%d: %s\n  condition: %s\n",       \
                __FILE__, __LINE__, (msg), #cond);                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

typedef struct {
    complex double *amp;       /* local slice, length local_size           */
    int      n_qubits;         /* global qubit count                       */
    int      rank, n_procs;    /* MPI rank and size (size = 2^p)           */
    int      p;                /* log2(n_procs); top p index bits = rank   */
    size_t   local_size;       /* = 2^(n_qubits - p)                       */
    MPI_Comm comm;
} qreg;

/* Lifecycle */
qreg *qreg_create   (int n_qubits, MPI_Comm comm);
void  qreg_destroy  (qreg *q);
void  qreg_init_basis(qreg *q, size_t basis_state);

/* Reductions used by tests and algorithms */
double qreg_norm(const qreg *q);
double prob_of  (const qreg *q, size_t basis);

/* Single-qubit gates (spec §6.1). */
void apply_u(qreg *q, int target, complex double u[2][2]);
void apply_h(qreg *q, int target);

#endif /* MATRIX_H */
