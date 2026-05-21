#ifndef GROVER_H
#define GROVER_H

#include "matrix.h"

/* Phase-oracle callback: applies (-1)^f(x) to amplitude at index x in
 * place. user is whatever the caller passed to apply_grover.            */
typedef void (*oracle_fn)(qreg *q, void *user);

/* Spec §6.5. Initialises q to the uniform superposition over the first
 * n_qubits qubits, then runs `iterations` rounds of oracle + diffusion. */
void apply_grover(qreg *q, int n_qubits, oracle_fn oracle, void *user,
                  int iterations);

#endif
