#ifndef SHOR_H
#define SHOR_H

#include "matrix.h"
#include <stdint.h>

void apply_modular_exp(qreg *q,
                       int counting_start, int t,
                       int target_start,   int n,
                       uint64_t a, uint64_t N);

typedef struct {
    uint64_t r;            /* recovered period, 0 if failed */
    uint64_t measured_c;   /* the integer the QFT measurement returned */
} shor_period_result;

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
