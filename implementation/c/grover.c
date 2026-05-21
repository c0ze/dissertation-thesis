#include "grover.h"

void apply_grover(qreg *q, int n_qubits, oracle_fn oracle, void *user,
                  int iterations) {
    QREG_ASSERT(q != NULL,           "apply_grover: q is NULL");
    QREG_ASSERT(oracle != NULL,      "apply_grover: oracle is NULL");
    QREG_ASSERT(n_qubits >= 1 && n_qubits <= q->n_qubits,
                "apply_grover: n_qubits out of range");
    QREG_ASSERT(iterations >= 0,     "apply_grover: iterations negative");
    /* Uniform superposition. */
    for (int i = 0; i < n_qubits; i++) apply_h(q, i);
    /* Iterate. */
    for (int it = 0; it < iterations; it++) {
        oracle(q, user);
        /* Diffusion: H^n -> X^n -> multi-controlled-Z -> X^n -> H^n.
         * Pre-X turns the flip on |1...1> into a flip on |0...0>, then
         * H^n turns the |0...0> reflection into the |s> reflection.     */
        for (int i = 0; i < n_qubits; i++) apply_h(q, i);
        for (int i = 0; i < n_qubits; i++) apply_x(q, i);
        apply_multi_controlled_z(q, n_qubits);
        for (int i = 0; i < n_qubits; i++) apply_x(q, i);
        for (int i = 0; i < n_qubits; i++) apply_h(q, i);
    }
}
