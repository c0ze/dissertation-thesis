#include "parallel.h"
#include "matrix.h"
#include <string.h>

int is_local_qubit(const qreg *q, int k) {
    QREG_ASSERT(q != NULL, "is_local_qubit: q is NULL");
    QREG_ASSERT(k >= 0 && k < q->n_qubits, "is_local_qubit: k out of range");
    return k < q->n_qubits - q->p;
}

int partner_for(const qreg *q, int k) {
    QREG_ASSERT(q != NULL, "partner_for: q is NULL");
    QREG_ASSERT(k >= q->n_qubits - q->p && k < q->n_qubits,
                "partner_for: k is not a global qubit");
    int bit_in_rank = k - (q->n_qubits - q->p);
    return q->rank ^ (1 << bit_in_rank);
}

int rank_owns(const qreg *q, size_t global_index) {
    QREG_ASSERT(q != NULL, "rank_owns: q is NULL");
    QREG_ASSERT(global_index < ((size_t)1 << q->n_qubits),
                "rank_owns: global_index out of range");
    return (int)(global_index >> (q->n_qubits - q->p)) == q->rank;
}

size_t global_to_local(const qreg *q, size_t global_index) {
    QREG_ASSERT(q != NULL, "global_to_local: q is NULL");
    QREG_ASSERT(rank_owns(q, global_index),
                "global_to_local: this rank does not own global_index");
    return global_index & (q->local_size - 1);
}

size_t local_to_global(const qreg *q, size_t local_index) {
    QREG_ASSERT(q != NULL, "local_to_global: q is NULL");
    QREG_ASSERT(local_index < q->local_size,
                "local_to_global: local_index out of range");
    return ((size_t)q->rank * q->local_size) + local_index;
}

void exchange_amplitudes(qreg *q, int partner_rank, complex double *recv_buf) {
    QREG_ASSERT(q != NULL, "exchange_amplitudes: q is NULL");
    QREG_ASSERT(recv_buf != NULL, "exchange_amplitudes: recv_buf is NULL");
    QREG_ASSERT(partner_rank >= 0 && partner_rank < q->n_procs,
                "exchange_amplitudes: partner_rank out of range");
    QREG_ASSERT(partner_rank != q->rank,
                "exchange_amplitudes: partner is self");
    MPI_Sendrecv(q->amp,    (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
                 partner_rank, 0,
                 recv_buf,  (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
                 partner_rank, 0,
                 q->comm, MPI_STATUS_IGNORE);
}
