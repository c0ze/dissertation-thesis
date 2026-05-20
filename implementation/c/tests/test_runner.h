#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <mpi.h>
#include <stdio.h>
#include "unity/unity.h"

/* Per spec §7.2. Every test_<module>.c expands TEST_RUNNER_MAIN() once at
 * the bottom of the file. The macro produces a complete main() that:
 *   - initialises MPI;
 *   - silences stdout/stderr on rank > 0 before UnityBegin, so only
 *     rank 0's report reaches the user;
 *   - runs the suite via the file-local register_tests() function;
 *   - calls UnityEnd() on EVERY rank (it finalises Unity's failure
 *     count), then MPI_LOR-reduces the per-rank pass/fail bit so any
 *     rank failing surfaces as a non-zero exit;
 *   - finalises MPI.
 */
#define TEST_RUNNER_MAIN()                                                  \
    void register_tests(void);                                              \
    int main(int argc, char **argv) {                                       \
        MPI_Init(&argc, &argv);                                             \
        int _rank;  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);                  \
        if (_rank != 0) {                                                   \
            freopen("/dev/null", "w", stdout);                              \
            freopen("/dev/null", "w", stderr);                              \
        }                                                                   \
        UnityBegin(__FILE__);                                               \
        register_tests();                                                   \
        int _unity_fail  = UnityEnd();                                      \
        int _local_fail  = (_unity_fail != 0) ? 1 : 0;                      \
        int _global_fail = 0;                                               \
        MPI_Allreduce(&_local_fail, &_global_fail, 1, MPI_INT, MPI_LOR,     \
                      MPI_COMM_WORLD);                                      \
        MPI_Finalize();                                                     \
        return _global_fail;                                                \
    }

#endif /* TEST_RUNNER_H */
