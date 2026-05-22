#ifndef SHOR_H
#define SHOR_H

#include "matrix.h"
#include <stdint.h>

void apply_modular_exp(qreg *q,
                       int counting_start, int t,
                       int target_start,   int n,
                       uint64_t a, uint64_t N);

typedef struct {
    uint64_t r;            /* candidate period from continued-fraction
                            * recovery, 0 if recovery failed outright.
                            * NOT guaranteed to equal the true order of
                            * a mod N -- see apply_shor_period below.  */
    uint64_t measured_c;   /* the integer the QFT measurement returned */
} shor_period_result;

/* Runs the period-finding circuit (modular_exp + inverse QFT +
 * measurement) and returns the denominator extracted by continued
 * fractions from the measured value.
 *
 * Caveat: the returned `r` is a CANDIDATE. The QFT readout is
 *   stochastic; the continued-fraction step may return a divisor of
 *   the true period, or fail outright when the measured integer is
 *   close to a multiple of 2^t but not aligned to a period boundary.
 *   Callers must verify with the classical check  a^r mod N == 1
 *   before treating `r` as the order. This is what `shor_factor` does
 *   in its outer retry loop; users who call `apply_shor_period`
 *   directly should do the same.                                       */
shor_period_result apply_shor_period(qreg *q,
                                     int counting_start, int t,
                                     int target_start,   int n,
                                     uint64_t a, uint64_t N);

typedef struct {
    uint64_t p, q;         /* non-trivial factors of N, 0 if failed */
    int      attempts;
} shor_factor_result;

shor_factor_result shor_factor(uint64_t N, int max_attempts);

#endif
