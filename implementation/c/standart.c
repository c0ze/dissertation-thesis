#include "standart.h"

#include <math.h>

uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 1) return 0;
    __uint128_t result = 1;
    __uint128_t b      = base % mod;
    while (exp > 0) {
        if (exp & 1ULL) {
            result = (result * b) % mod;
        }
        b   = (b * b) % mod;
        exp >>= 1;
    }
    return (uint64_t)result;
}

void continued_fraction(double x, uint64_t max_denominator,
                        uint64_t *num, uint64_t *den) {
    /* h[k]/k[k] are the convergents. Recurrence:
     *   h_{-1}=1, h_{-2}=0; k_{-1}=0, k_{-2}=1
     *   h_k = a_k * h_{k-1} + h_{k-2}
     *   k_k = a_k * k_{k-1} + k_{k-2}
     */
    uint64_t h1 = 1, h2 = 0;
    uint64_t k1 = 0, k2 = 1;
    uint64_t best_h = 0, best_k = 1;
    double   y       = x;
    for (int i = 0; i < 64; i++) {
        double a_d = floor(y);
        if (a_d < 0 || a_d > (double)UINT64_MAX) break;
        uint64_t a  = (uint64_t)a_d;
        /* check overflow of next denominator */
        if (k1 != 0 && a > (UINT64_MAX - k2) / k1) break;
        uint64_t k0 = a * k1 + k2;
        uint64_t h0 = a * h1 + h2;
        if (k0 > max_denominator) break;
        best_h = h0;
        best_k = k0;
        h2 = h1; h1 = h0;
        k2 = k1; k1 = k0;
        double frac = y - a_d;
        if (frac < 1e-18) break;
        y = 1.0 / frac;
    }
    *num = best_h;
    *den = best_k;
}

int is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

int ilog2_u32(uint32_t x) {
    /* Precondition: x is a power of two and nonzero. */
    int r = 0;
    while (x > 1) { x >>= 1; r++; }
    return r;
}
