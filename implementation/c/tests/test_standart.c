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

static void test_continued_fraction_pi(void) {
    uint64_t num, den;
    /* 22/7 is the first famous convergent of pi with denominator <= 100. */
    continued_fraction(3.14159265358979323846, 100, &num, &den);
    TEST_ASSERT_EQUAL_UINT64(22, num);
    TEST_ASSERT_EQUAL_UINT64(7,  den);
    /* 355/113 is the next, the famous Milü, with denominator <= 200. */
    continued_fraction(3.14159265358979323846, 200, &num, &den);
    TEST_ASSERT_EQUAL_UINT64(355, num);
    TEST_ASSERT_EQUAL_UINT64(113, den);
}

static void test_continued_fraction_simple_period(void) {
    uint64_t num, den;
    /* 3/8 should round-trip exactly with max_denom >= 8. */
    continued_fraction(3.0 / 8.0, 16, &num, &den);
    TEST_ASSERT_EQUAL_UINT64(3, num);
    TEST_ASSERT_EQUAL_UINT64(8, den);
    /* 5/16 likewise. */
    continued_fraction(5.0 / 16.0, 32, &num, &den);
    TEST_ASSERT_EQUAL_UINT64(5,  num);
    TEST_ASSERT_EQUAL_UINT64(16, den);
}

static void test_is_power_of_two(void) {
    TEST_ASSERT_TRUE (is_power_of_two(1));
    TEST_ASSERT_TRUE (is_power_of_two(2));
    TEST_ASSERT_TRUE (is_power_of_two(4));
    TEST_ASSERT_TRUE (is_power_of_two(1024));
    TEST_ASSERT_FALSE(is_power_of_two(0));
    TEST_ASSERT_FALSE(is_power_of_two(3));
    TEST_ASSERT_FALSE(is_power_of_two(6));
    TEST_ASSERT_FALSE(is_power_of_two(1023));
}

static void test_ilog2_u32(void) {
    TEST_ASSERT_EQUAL_INT( 0, ilog2_u32(1));
    TEST_ASSERT_EQUAL_INT( 1, ilog2_u32(2));
    TEST_ASSERT_EQUAL_INT( 2, ilog2_u32(4));
    TEST_ASSERT_EQUAL_INT(10, ilog2_u32(1024));
    TEST_ASSERT_EQUAL_INT(20, ilog2_u32(1 << 20));
}

void register_tests(void) {
    RUN_TEST(test_gcd_basics);
    RUN_TEST(test_mod_pow_basics);
    RUN_TEST(test_continued_fraction_pi);
    RUN_TEST(test_continued_fraction_simple_period);
    RUN_TEST(test_is_power_of_two);
    RUN_TEST(test_ilog2_u32);
}

TEST_RUNNER_MAIN()
