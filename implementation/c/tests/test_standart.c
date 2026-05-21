#include <mpi.h>
#include <stdint.h>
#include "standart.h"
#include "unity/unity.h"
#include "test_runner.h"

void setUp(void)    {}
void tearDown(void) {}

static void test_gcd_basics(void) {
    TEST_ASSERT_EQUAL_UINT64(1,  gcd_u64(1, 1));
    TEST_ASSERT_EQUAL_UINT64(7,  gcd_u64(14, 21));
    TEST_ASSERT_EQUAL_UINT64(6,  gcd_u64(12, 18));
    TEST_ASSERT_EQUAL_UINT64(1,  gcd_u64(7, 11));    /* coprime */
    TEST_ASSERT_EQUAL_UINT64(15, gcd_u64(15, 0));    /* gcd(x,0) = x */
    TEST_ASSERT_EQUAL_UINT64(15, gcd_u64(0, 15));
}

static void test_mod_pow_basics(void) {
    /* a^0 mod N = 1 for any a, N>0 */
    TEST_ASSERT_EQUAL_UINT64(1,  mod_pow(7, 0, 15));
    /* a^1 mod N = a mod N */
    TEST_ASSERT_EQUAL_UINT64(7,  mod_pow(7, 1, 15));
    TEST_ASSERT_EQUAL_UINT64(2,  mod_pow(17, 1, 15));
    /* Known period: 7^4 mod 15 = 1 (the standard Shor example for N=15) */
    TEST_ASSERT_EQUAL_UINT64(4,  mod_pow(7, 2, 15));
    TEST_ASSERT_EQUAL_UINT64(13, mod_pow(7, 3, 15));
    TEST_ASSERT_EQUAL_UINT64(1,  mod_pow(7, 4, 15));
    /* Large exponent without overflow: 2^64 mod 1000003 (a prime) */
    TEST_ASSERT_EQUAL_UINT64(350687, mod_pow(2, 64, 1000003));
}

void register_tests(void) {
    RUN_TEST(test_gcd_basics);
    RUN_TEST(test_mod_pow_basics);
}

TEST_RUNNER_MAIN()
