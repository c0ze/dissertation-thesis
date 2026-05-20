/* test_smoke.c - sanity check that the MPI test runner harness works
 * end-to-end before any real library code exists. Asserts only things
 * about MPI itself. Removed (or kept) once the first real test lands.
 */
#include <mpi.h>
#include "unity/unity.h"
#include "test_runner.h"

static int g_rank, g_size;

void setUp(void)    {}
void tearDown(void) {}

static void test_mpi_size_is_positive(void) {
    TEST_ASSERT_GREATER_THAN_INT(0, g_size);
}

static void test_mpi_rank_in_range(void) {
    TEST_ASSERT_GREATER_OR_EQUAL_INT(0, g_rank);
    TEST_ASSERT_LESS_THAN_INT(g_size, g_rank);
}

static void test_mpi_size_is_power_of_two(void) {
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, g_size & (g_size - 1),
        "NP must be a power of two for this suite");
}

void register_tests(void) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    RUN_TEST(test_mpi_size_is_positive);
    RUN_TEST(test_mpi_rank_in_range);
    RUN_TEST(test_mpi_size_is_power_of_two);
}

TEST_RUNNER_MAIN()
