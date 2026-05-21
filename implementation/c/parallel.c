#include "parallel.h"
#include "matrix.h"
#include <stdlib.h>
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

void redistribute_pairs(qreg *q, const size_t *global_indices,
                        size_t n_pairs, const complex double *values) {
    QREG_ASSERT(q != NULL,                  "redistribute_pairs: q is NULL");
    QREG_ASSERT(n_pairs == 0 || (global_indices && values),
                "redistribute_pairs: missing arrays");

    int  P = q->n_procs;
    int *send_counts = calloc(P, sizeof *send_counts);
    int *recv_counts = calloc(P, sizeof *recv_counts);
    int *send_displs = calloc(P, sizeof *send_displs);
    int *recv_displs = calloc(P, sizeof *recv_displs);

    /* Count destinations. */
    for (size_t i = 0; i < n_pairs; i++) {
        int dest = (int)(global_indices[i] >> (q->n_qubits - q->p));
        send_counts[dest]++;
    }
    /* Exchange counts. */
    MPI_Alltoall(send_counts, 1, MPI_INT,
                 recv_counts, 1, MPI_INT, q->comm);

    int total_send = 0, total_recv = 0;
    for (int r = 0; r < P; r++) {
        send_displs[r] = total_send; total_send += send_counts[r];
        recv_displs[r] = total_recv; total_recv += recv_counts[r];
    }

    /* Pack: each pair as (size_t local_offset, complex double value).
     * We send local-offsets so the receiver can apply them directly.    */
    size_t         *send_off = malloc((size_t)total_send * sizeof *send_off);
    complex double *send_val = malloc((size_t)total_send * sizeof *send_val);
    int *cursor = calloc(P, sizeof *cursor);
    for (size_t i = 0; i < n_pairs; i++) {
        int dest = (int)(global_indices[i] >> (q->n_qubits - q->p));
        int slot = send_displs[dest] + cursor[dest]++;
        send_off[slot] = global_indices[i] & (q->local_size - 1);
        send_val[slot] = values[i];
    }
    free(cursor);

    size_t         *recv_off = malloc((size_t)total_recv * sizeof *recv_off);
    complex double *recv_val = malloc((size_t)total_recv * sizeof *recv_val);

    /* Two Alltoallv calls: one for offsets, one for amplitudes. */
    MPI_Alltoallv(send_off, send_counts, send_displs, MPI_UNSIGNED_LONG,
                  recv_off, recv_counts, recv_displs, MPI_UNSIGNED_LONG,
                  q->comm);
    MPI_Alltoallv(send_val, send_counts, send_displs, MPI_C_DOUBLE_COMPLEX,
                  recv_val, recv_counts, recv_displs, MPI_C_DOUBLE_COMPLEX,
                  q->comm);

    /* Accumulate. */
    for (int i = 0; i < total_recv; i++) {
        q->amp[recv_off[i]] += recv_val[i];
    }

    free(send_off); free(send_val); free(recv_off); free(recv_val);
    free(send_counts); free(recv_counts); free(send_displs); free(recv_displs);
}
