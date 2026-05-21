# ---------------------------------------------------------------------------
# makefile - build the 2004 dense-matrix MPI quantum simulator.
#
# Original 2004 makefile was a one-liner targeting LAM-MPI on Linux:
#       mpicc qubit.c matrix.c standart.c parallel.c -lm -o qubit
#
# It no longer builds clean under modern toolchains for two reasons:
#   1. matrix.c forgets to #include <stdlib.h>, so malloc is implicitly
#      declared. ISO C99 made this a hard error.
#   2. standart.h forward-declares Complex_Multiplication but NOT
#      get_biggest_2s_power, so qubit.c also uses it implicitly.
#
# Neither issue can be fixed in tooling without touching the .c/.h files,
# so we demote them back to warnings via -Wno-implicit-function-declaration
# and -Wno-implicit-int. The source itself is unchanged from 2004.
#
# Targets:
#   make            build  build/qubit
#   make clean      remove build/
#   make run        build then mpirun -n 4 build/qubit  (override with NP=...)
#   make check      build, run with NP=4, assert per-rank amplitude counts
#
# The pristine 2004 Linux ELF binary stays in place at ./qubit and is not
# touched by this makefile.
#
# Note on NP:
#   The original main() partitions an 8-row dense matrix across ranks using
#   size_2P = the largest power of two <= NP. Workers from 1..size_2P-1
#   each post MPI_Recv calls; ranks above size_2P also enter the worker
#   path and block forever on receives that never come (the master only
#   sends to size_2P-1 ranks).
#
#   Safe values for this 3-qubit demo are therefore strictly powers of two:
#   NP = 1, 2, 4, or 8. The recipes below enforce this with a fail-fast
#   guard so users see a clear error instead of a hung mpirun.
# ---------------------------------------------------------------------------

MPICC    ?= mpicc
NP       ?= 4
BUILDDIR := build
TARGET   := $(BUILDDIR)/qubit
SOURCES  := qubit.c matrix.c standart.c parallel.c

# C99 needed for VLAs as function-local declarations.
# The two -Wno-... flags are the price of leaving the 2004 source alone.
CFLAGS   := -std=c99 \
            -Wno-implicit-function-declaration \
            -Wno-implicit-int
LDFLAGS  := -lm

.PHONY: all clean run check require-pow2-np

all: $(TARGET)

$(TARGET): $(SOURCES) matrix.h parallel.h standart.h | $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(SOURCES) $(LDFLAGS) -o $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

# Fail fast if NP is not a power of two. The 2004 main() deadlocks on
# non-power-of-two rank counts (workers beyond size_2P-1 hang in MPI_Recv).
require-pow2-np:
	@case "$(NP)" in 1|2|4|8) ;; *) \
	    echo "error: NP=$(NP) is unsafe for this demo." >&2; \
	    echo "       This 3-qubit demo deadlocks on non-power-of-two NP and" >&2; \
	    echo "       wastes ranks beyond 2^qureg_size = 8. Use NP=1, 2, 4, or 8." >&2; \
	    exit 2;; esac

# Run the hardcoded 3-qubit Hadamard demo. Override NP for different ranks.
#   make run             # default 4 ranks
#   make run NP=2        # two ranks
#   make run NP=8        # eight ranks (uses --oversubscribe on small laptops)
run: require-pow2-np $(TARGET)
	mpirun --oversubscribe -n $(NP) $(TARGET)

# Strict verification of the H(*)H(*)H |010> output at NP=4.
#
# Expected exact state vector (8 amplitudes, big-endian basis ordering):
#       (+,+,-,-,+,+,-,-) / sqrt(8)  ~  +-0.354
#
# With NP=4 the rank-0 master is idle and ranks 1, 2, 3 print rows 2-3,
# 4-5, 6-7 respectively. Expected per-rank output:
#       rank 1 : (-0.354), (-0.354)         <- rows 2, 3
#       rank 2 : (+0.354), (+0.354)         <- rows 4, 5
#       rank 3 : (-0.354), (-0.354)         <- rows 6, 7
# Totals: 4 negative amplitudes, 2 positive, 3 rank headers.
#
# stdout from different ranks interleaves under mpirun. Use OpenMPI's
# --tag-output so every line is prefixed with its rank, then assert
# each rank's amplitude signs by prefix rather than by line adjacency.
check: $(TARGET)
	@out=$$(mpirun --oversubscribe --tag-output -n 4 $(TARGET) 2>&1); \
	echo "$$out"; \
	echo "----"; \
	neg=$$(echo "$$out" | grep -cF "(-0.354 + 0.000 i)"); \
	pos=$$(echo "$$out" | grep -cF "(0.354 + 0.000 i)"); \
	r1=$$(echo "$$out" | grep -cE "my rank : 1 "); \
	r2=$$(echo "$$out" | grep -cE "my rank : 2 "); \
	r3=$$(echo "$$out" | grep -cE "my rank : 3 "); \
	r0=$$(echo "$$out" | grep -cE "my rank : 0 "); \
	fail=0; \
	[ "$$neg" = 4 ] || { echo "FAIL: expected 4 negative amplitudes, got $$neg"; fail=1; }; \
	[ "$$pos" = 2 ] || { echo "FAIL: expected 2 positive amplitudes, got $$pos"; fail=1; }; \
	[ "$$r1"  = 1 ] || { echo "FAIL: expected exactly 1 'my rank : 1' header, got $$r1"; fail=1; }; \
	[ "$$r2"  = 1 ] || { echo "FAIL: expected exactly 1 'my rank : 2' header, got $$r2"; fail=1; }; \
	[ "$$r3"  = 1 ] || { echo "FAIL: expected exactly 1 'my rank : 3' header, got $$r3"; fail=1; }; \
	[ "$$r0"  = 0 ] || { echo "FAIL: master should not compute, but emitted 'my rank : 0'"; fail=1; }; \
		echo "$$out" | awk ' \
		    /^\[1,[0-9]+\]<stdout>:[[:space:]]*\((-)?0\.354 \+ 0\.000 i\)/ { \
		        line = $$0; \
		        sub(/^\[1,/, "", line); \
		        sub(/\].*/, "", line); \
		        r = line; \
		        sign = ($$0 ~ /\(-0\.354/) ? "-" : "+"; \
		        observed[r] = observed[r] sign; \
		        next \
		    } \
		    END { \
		        for (r = 1; r <= 3; r++) { \
		            expected = (r == 1) ? "--" : \
		                       (r == 2) ? "++" : \
		                       (r == 3) ? "--" : "??"; \
	            if (observed[r] != expected) { \
	                printf "FAIL: rank %d: expected [%s], got [%s]\n", \
	                       r, expected, observed[r]; \
	                exit 1 \
	            } \
	        } \
	    }' || fail=1; \
	if [ $$fail -eq 0 ]; then echo "OK"; else exit 1; fi
