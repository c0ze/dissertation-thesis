#ifndef GROVER_H
#define GROVER_H

#include "matrix.h"

/* Phase-oracle callback: applies (-1)^f(x) to amplitude at index x in
 * place. user is whatever the caller passed to apply_grover.            */
typedef void (*oracle_fn)(qreg *q, void *user);

/* Spec §6.5. Applies H to each of the first n_qubits qubits, then runs
 * `iterations` rounds of oracle + diffusion (H^n X^n MCZ X^n H^n).
 *
 * Precondition: the first n_qubits qubits must be in |0...0>. The
 *   Hadamards then produce the uniform superposition the algorithm
 *   needs; if the register is in any other state, the routine performs
 *   the same gate sequence but the meaning of the run is no longer
 *   ``standard Grover.'' Call qreg_init_basis(q, 0) first if in doubt.
 *
 * The remaining qubits, if any, are not touched and act as inert
 * ancilla space.                                                        */
void apply_grover(qreg *q, int n_qubits, oracle_fn oracle, void *user,
                  int iterations);

#endif
