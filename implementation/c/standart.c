#include "standart.h"

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
