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

#endif
