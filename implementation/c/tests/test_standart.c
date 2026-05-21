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

void register_tests(void) {
    RUN_TEST(test_gcd_basics);
}

TEST_RUNNER_MAIN()
