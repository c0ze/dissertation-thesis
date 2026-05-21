# ---------------------------------------------------------------------------
# makefile for implementation/c -- sparse-gate MPI quantum simulator
#
# Targets:
#   make                 build libqubit.a (when sources exist) and bin/qubit
#   make test            build and run every tests/test_*.c at NP=1, 2, 4
#   make test-large      runs the suite at NP=1,2,4,8 and sets
#                          RUN_SHOR_21=1 to unlock the Shor(21) factor
#                          test (skipped by default in `make test`)
#   make demo ALGO=qft NP=4
#   make clean / distclean
#
# Requirements:
#   mpicc (OpenMPI 5.x, brew install open-mpi)
#
# This file grows as more source files come online. At Task 5 it does the
# minimum: lays out directories, defines variables, provides empty test
# and clean targets so the structure is in place for later tasks.
# ---------------------------------------------------------------------------

MPICC      ?= mpicc
CFLAGS     ?= -std=c11 -O2 -Wall -Wextra
LDFLAGS    ?= -lm

ifeq ($(DEBUG),1)
    CFLAGS  := -std=c11 -O0 -g -Wall -Wextra -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
endif

# Unity defaults to skipping double-precision assertions; we use
# TEST_ASSERT_DOUBLE_WITHIN throughout the suite (amplitudes, norms,
# probabilities), so we must opt back in.
CFLAGS     += -DUNITY_INCLUDE_DOUBLE

BUILDDIR   := build
BINDIR     := $(BUILDDIR)/bin
TESTDIR    := $(BUILDDIR)/tests

# Library sources will be filled in as later tasks add files.
LIB_SRCS   := standart.c matrix.c parallel.c qft.c grover.c shor.c
LIB_OBJS   := $(LIB_SRCS:%.c=$(BUILDDIR)/%.o)

# Test files. Each one becomes its own binary.
TEST_SRCS  := $(wildcard tests/test_*.c)
TEST_BINS  := $(TEST_SRCS:tests/%.c=$(TESTDIR)/%)

# NP values to exercise.
NPS_SMALL  := 1 2 4
NPS_LARGE  := 1 2 4 8

# Unity vendored.
UNITY_OBJ  := $(BUILDDIR)/unity.o

.PHONY: all test test-large clean distclean dirs demo

all: dirs $(BINDIR)/qubit

dirs:
	@mkdir -p $(BUILDDIR) $(BINDIR) $(TESTDIR)

# Compile a library .c into build/.
$(BUILDDIR)/%.o: %.c | dirs
	$(MPICC) $(CFLAGS) -c $< -o $@

# Unity compiled separately with relaxed warnings.
$(UNITY_OBJ): tests/unity/unity.c | dirs
	$(MPICC) -std=c11 -O2 -w -DUNITY_INCLUDE_DOUBLE -c $< -o $@

# Compile a test source plus the library plus Unity into a test binary.
$(TESTDIR)/%: tests/%.c $(LIB_OBJS) $(UNITY_OBJ) | dirs
	$(MPICC) $(CFLAGS) -I. -Itests -Itests/unity $< $(LIB_OBJS) $(UNITY_OBJ) $(LDFLAGS) -o $@

# Run a single test binary at a list of NP values.  MPI_ENV is an
# optional set of `-x VAR=val` flags that targets can override to
# forward extra environment variables (e.g. RUN_SHOR_21) through
# mpirun to each rank's environment.
MPI_ENV ?=
define run_test_at_np
	@echo "--- $(1) at NP=$(2) ---"
	@mpirun --oversubscribe $(MPI_ENV) -n $(2) $(1) || (echo "FAIL: $(1) at NP=$(2)"; exit 1)

endef

test: $(TEST_BINS)
	$(foreach bin,$(TEST_BINS),$(foreach np,$(NPS_SMALL),$(call run_test_at_np,$(bin),$(np))))
	@echo "OK: all tests passed at NP=$(NPS_SMALL)"

# test-large opts in to slower tests gated behind RUN_SHOR_21 (currently
# the 16-qubit Shor(21) end-to-end factor). The variable is forwarded
# through mpirun via -x so it lands in every rank's environment.
test-large: MPI_ENV := -x RUN_SHOR_21=1
test-large: $(TEST_BINS)
	$(foreach bin,$(TEST_BINS),$(foreach np,$(NPS_LARGE),$(call run_test_at_np,$(bin),$(np))))
	@echo "OK: all tests passed at NP=$(NPS_LARGE) (including RUN_SHOR_21)"

# Demo binary.
DEMO_OBJ := $(BUILDDIR)/qubit.o
$(DEMO_OBJ): qubit.c $(LIB_OBJS) | dirs
	$(MPICC) $(CFLAGS) -I. -c $< -o $@
$(BINDIR)/qubit: $(DEMO_OBJ) $(LIB_OBJS)
	$(MPICC) $(CFLAGS) $(DEMO_OBJ) $(LIB_OBJS) $(LDFLAGS) -o $@

# make demo ALGO=qft NP=4
ALGO ?= bell
NP   ?= 4
demo: $(BINDIR)/qubit
	mpirun --oversubscribe -n $(NP) $(BINDIR)/qubit --algo $(ALGO)

clean:
	rm -rf $(BUILDDIR)

distclean: clean
