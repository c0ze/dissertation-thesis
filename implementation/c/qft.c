#include "qft.h"
#include <math.h>

void apply_qft(qreg *q, int start, int n_qubits) {
    QREG_ASSERT(q != NULL,         "apply_qft: q is NULL");
    QREG_ASSERT(n_qubits >= 1,     "apply_qft: n_qubits < 1");
    QREG_ASSERT(start >= 0,        "apply_qft: start < 0");
    QREG_ASSERT(start + n_qubits <= q->n_qubits,
                "apply_qft: range exceeds register size");
    /* Standard textbook decomposition. For each qubit j from the most-
     * significant down to the least:
     *   apply H on qubit (start + n_qubits - 1 - j)
     *   for each k > j, apply controlled-R_{k-j+1} from
     *     control = start + n_qubits - 1 - k
     *     target  = start + n_qubits - 1 - j
     *     angle  = 2*pi / 2^(k-j+1)
     * Finish with a bit-reversal SWAP across the range so the output
     * is in natural binary order.                                        */
    for (int j = 0; j < n_qubits; j++) {
        int target = start + (n_qubits - 1 - j);
        apply_h(q, target);
        for (int k = j + 1; k < n_qubits; k++) {
            int control = start + (n_qubits - 1 - k);
            double theta = 2.0 * M_PI / (double)((size_t)1 << (k - j + 1));
            apply_controlled_phase(q, control, target, theta);
        }
    }
    /* Final swaps. */
    for (int i = 0; i < n_qubits / 2; i++) {
        apply_swap(q, start + i, start + n_qubits - 1 - i);
    }
}
void apply_qft_inverse(qreg *q, int start, int n_qubits) {
    (void)q; (void)start; (void)n_qubits;   /* stub */
}
