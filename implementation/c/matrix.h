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

/* C11 does not require <math.h> to expose M_PI, and CI builds with a
 * strict language mode. Keep the library's angle constant project-local. */
#define QREG_PI 3.141592653589793238462643383279502884

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
void apply_x(qreg *q, int target);
void apply_y(qreg *q, int target);
void apply_z(qreg *q, int target);
void apply_s    (qreg *q, int target);
void apply_t    (qreg *q, int target);
void apply_phase(qreg *q, int target, double theta);
void apply_rx(qreg *q, int target, double theta);
void apply_ry(qreg *q, int target, double theta);
void apply_rz(qreg *q, int target, double theta);

void apply_cu(qreg *q, int control, int target, complex double u[2][2]);

void apply_cnot            (qreg *q, int control, int target);
void apply_cz              (qreg *q, int control, int target);
void apply_controlled_phase(qreg *q, int control, int target, double theta);

void apply_swap(qreg *q, int a, int b);

/* Phase-flip the single all-ones amplitude |1...1> on the first n qubits. */
void apply_multi_controlled_z(qreg *q, int n);

/* Generalised Toffoli: flip target iff every control is set.
 *
 * v1 limitation: every control AND the target must be a LOCAL qubit
 *   (index < n_qubits - p). The distributed version would itself
 *   decompose into a Toffoli + ancilla ladder and is left for a
 *   follow-up. Passing a global control or target aborts via
 *   QREG_ASSERT.                                                       */
void apply_multi_controlled_x(qreg *q, const int *controls, int n_controls,
                              int target);

/* Seed the rank-0 stdlib RNG used by measure_qubit / measure_all /
 * sample_distribution. Call once on rank 0 (or on every rank with the
 * same seed) before measurement; otherwise the default C-library
 * sequence is deterministic across runs. shor_factor seeds itself on
 * first use, so a caller that only goes through shor_factor does not
 * need to call this.                                                   */
void qreg_seed(qreg *q, uint64_t seed);

int measure_qubit(qreg *q, int target);

size_t measure_all        (qreg *q);
void   sample_distribution(const qreg *q, size_t *out, int shots);
qreg  *qreg_clone         (const qreg *q);
void   qreg_dump          (const qreg *q, FILE *f);   /* rank 0 prints global state */

#endif /* MATRIX_H */
