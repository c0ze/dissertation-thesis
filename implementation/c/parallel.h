#ifndef PARALLEL_H
#define PARALLEL_H

#include "matrix.h"

/* Spec §6.2 — locality + exchange primitives. */

int    is_local_qubit  (const qreg *q, int k);
int    partner_for     (const qreg *q, int k);          /* must be global qubit */
int    rank_owns       (const qreg *q, size_t global_index);
size_t global_to_local (const qreg *q, size_t global_index);
size_t local_to_global (const qreg *q, size_t local_index);

void exchange_amplitudes(qreg *q, int partner_rank,
                         complex double *recv_buf);
/* Sendrecvs q->amp <-> recv_buf with partner_rank in q->comm.
 * recv_buf must be q->local_size complex doubles. */

/* Accumulate (global_index, amplitude) pairs into q->amp via MPI_Alltoallv.
 * After the call, q->amp[i] equals the SUM of every incoming amplitude
 * whose global index lands on this rank's slice at local offset i.
 *
 * Used by shor.c's apply_modular_exp under the distributed layout
 * (multiple source amplitudes can land on the same destination index).
 */
void redistribute_pairs(qreg *q, const size_t *global_indices,
                        size_t n_pairs, const complex double *values);

#endif
