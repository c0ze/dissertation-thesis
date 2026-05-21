#ifndef STANDART_H
#define STANDART_H

#include <stdint.h>

uint64_t gcd_u64(uint64_t a, uint64_t b);
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod);

/* Find the best rational approximation p/q to x with q <= max_denominator.
 * Writes the numerator and denominator out via *num and *den.
 * Algorithm: standard continued-fraction expansion truncated at the last
 * convergent that satisfies the denominator bound.
 */
void continued_fraction(double x, uint64_t max_denominator,
                        uint64_t *num, uint64_t *den);

#endif
