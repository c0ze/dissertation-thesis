#include "matrix.h"
#include "standart.h"
#include <stdlib.h>
#include <string.h>

qreg *qreg_create(int n_qubits, MPI_Comm comm) {
    if (n_qubits < 1 || n_qubits > QREG_MAX_QUBITS) return NULL;

    int n_procs, rank;
    MPI_Comm_size(comm, &n_procs);
    MPI_Comm_rank(comm, &rank);
    if (!is_power_of_two(n_procs))       return NULL;
    if ((size_t)n_procs > ((size_t)1 << n_qubits)) return NULL;

    qreg *q = (qreg *)malloc(sizeof *q);
    if (!q) return NULL;
    q->n_qubits   = n_qubits;
    q->n_procs    = n_procs;
    q->rank       = rank;
    q->p          = ilog2_u32((uint32_t)n_procs);
    q->local_size = (size_t)1 << (n_qubits - q->p);
    q->comm       = comm;
    q->amp        = (complex double *)calloc(q->local_size, sizeof *q->amp);
    if (!q->amp) { free(q); return NULL; }
    return q;
}

void qreg_destroy(qreg *q) {
    if (!q) return;
    free(q->amp);
    free(q);
}

void qreg_init_basis(qreg *q, size_t basis_state) {
    QREG_ASSERT(q != NULL, "qreg_init_basis: q is NULL");
    QREG_ASSERT(basis_state < ((size_t)1 << q->n_qubits),
                "qreg_init_basis: basis_state out of range");
    /* zero everything */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    /* set the owning rank's amplitude to 1 */
    int owning_rank = (int)(basis_state >> (q->n_qubits - q->p));
    if (q->rank == owning_rank) {
        size_t off = basis_state & (q->local_size - 1);
        q->amp[off] = 1.0;
    }
}

double qreg_norm(const qreg *q) {
    QREG_ASSERT(q != NULL, "qreg_norm: q is NULL");
    double local = 0.0;
    for (size_t i = 0; i < q->local_size; i++) {
        double r = creal(q->amp[i]);
        double im = cimag(q->amp[i]);
        local += r*r + im*im;
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    return global;
}

double prob_of(const qreg *q, size_t basis) {
    QREG_ASSERT(q != NULL, "prob_of: q is NULL");
    QREG_ASSERT(basis < ((size_t)1 << q->n_qubits),
                "prob_of: basis out of range");
    int owning_rank = (int)(basis >> (q->n_qubits - q->p));
    double local = 0.0;
    if (q->rank == owning_rank) {
        size_t off = basis & (q->local_size - 1);
        double r  = creal(q->amp[off]);
        double im = cimag(q->amp[off]);
        local = r*r + im*im;
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    return global;
}
