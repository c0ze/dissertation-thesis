#ifndef TEST_ASSERT_H
#define TEST_ASSERT_H

#include <complex.h>
#include <math.h>
#include "unity/unity.h"

/* Per spec §7.5 — tolerances used across the entire test suite. */
#define AMP_TOL   1e-10   /* per amplitude component */
#define PROB_TOL  1e-9    /* probabilities and sums  */

/* Assert two complex doubles are equal to within AMP_TOL component-wise. */
#define ASSERT_NEAR_AMP(expected, actual)                                  \
    do {                                                                   \
        complex double _e = (expected);                                    \
        complex double _a = (actual);                                      \
        TEST_ASSERT_DOUBLE_WITHIN(AMP_TOL, creal(_e), creal(_a));          \
        TEST_ASSERT_DOUBLE_WITHIN(AMP_TOL, cimag(_e), cimag(_a));          \
    } while (0)

/* Assert |q->amp|^2 sums to 1 across all ranks. */
#define ASSERT_NORM_ONE(q)                                                 \
    do {                                                                   \
        double _n = qreg_norm(q);                                          \
        TEST_ASSERT_DOUBLE_WITHIN(PROB_TOL, 1.0, _n);                      \
    } while (0)

#endif /* TEST_ASSERT_H */
