#ifndef QFT_H
#define QFT_H

#include "matrix.h"

/* Spec §6.4. apply_qft includes the final bit-reversal swaps so the
 * output amplitude at index y in natural binary equals
 *   (1/sqrt(N)) sum_x alpha_x exp(2*pi*i*x*y/N).                       */
void apply_qft        (qreg *q, int start, int n_qubits);
void apply_qft_inverse(qreg *q, int start, int n_qubits);

#endif
