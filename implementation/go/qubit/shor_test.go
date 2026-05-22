package qubit

import (
	"os"
	"testing"
)

func TestModularExpPassThroughYGeN(t *testing.T) {
	// Layout: 6 total qubits = 1 counting + 5 target. counting at
	// bits [0..0], target at bits [1..5]. N = 5.
	// Place amplitude at (x=0, y=10). 10 >= 5 so should stay.
	q, _ := NewQreg(6)
	q.InitBasis(uint64(10) << 1)
	q.ApplyModularExp(0, 1, 1, 5, 2, 5)
	assertAmpNear(t, complex(1, 0), q.amp[10<<1],
		"pass-through (x=0, y=10>=5)")
}

func TestModularExpOrbitA2Mod5(t *testing.T) {
	// a=2, N=5 powers: 2^0=1, 2^1=2, 2^2=4, 2^3=3 (mod 5).
	// Layout: counting at [0..1] (t=2), target at [2..4] (n=3), total=5.
	expected := []uint64{1, 2, 4, 3}
	for x := 0; x < 4; x++ {
		q, _ := NewQreg(5)
		initial := (uint64(1) << 2) | (uint64(x) << 0) // y=1, x=x
		q.InitBasis(initial)
		q.ApplyModularExp(0, 2, 2, 3, 2, 5)
		final := (expected[x] << 2) | (uint64(x) << 0)
		assertAmpNear(t, complex(1, 0), q.amp[final],
			"orbit map")
	}
}

func TestShorPeriodA7Mod15(t *testing.T) {
	// Order of 7 mod 15 is 4. Use t=8 counting + n=4 target = 12 qubits.
	n := 4
	tBits := 8
	q, _ := NewQreg(tBits+n, WithSeed(1))
	res := q.ApplyShorPeriod(n, tBits, 0, n, 7, 15)
	if res.R == 0 {
		t.Fatalf("period finder returned r=0 (no recovery); raw c=%d", res.MeasuredC)
	}
	// Recovered period must divide true r=4: r in {1, 2, 4}.
	if res.R != 1 && res.R != 2 && res.R != 4 {
		t.Errorf("recovered r=%d not a divisor of 4", res.R)
	}
}

func TestShorFactorRejectsNonPositiveMaxAttempts(t *testing.T) {
	// Negative or zero maxAttempts is a programmer-error input. The
	// function should reject it upfront with {0, 0, 0} instead of
	// silently returning {0, 0, attempts} (which would carry a
	// negative or zero attempts count -- nonsense as a result struct).
	for _, m := range []int{-5, -1, 0} {
		res := ShorFactor(15, m)
		if res.P != 0 || res.Q != 0 || res.Attempts != 0 {
			t.Errorf("ShorFactor(15, %d) = {P:%d, Q:%d, Attempts:%d}, want all zero",
				m, res.P, res.Q, res.Attempts)
		}
	}
}

func TestShorFactorRejectsOversizedN(t *testing.T) {
	// (2^25 + 1) needs a 3*log2(N)+1 ~= 79-qubit register at
	// QregMaxQubits=26. ShorFactor should reject upfront rather than
	// burn random-base effort and fail inside the attempt loop.
	// Returns {0, 0, 0}. Using 2^25+1 (odd) so we bypass the
	// N%2==0 short-circuit at the top of ShorFactor.
	res := ShorFactor((1<<25)+1, 8)
	if res.P != 0 || res.Q != 0 || res.Attempts != 0 {
		t.Errorf("ShorFactor(2^25+1) = {P:%d, Q:%d, Attempts:%d}, want all zero",
			res.P, res.Q, res.Attempts)
	}
}

func TestShorFactorHandlesHugeN(t *testing.T) {
	// N > 2^63 used to hang the old shift-based bit-width loop
	// (uint64(1) << 64 = 0 in Go, so `0 < N` never becomes false).
	// With bits.Len64(N-1) this terminates immediately and the upfront
	// QregMaxQubits check rejects without entering the attempt loop.
	res := ShorFactor(uint64(1)<<63+7, 1)
	if res.P != 0 || res.Q != 0 || res.Attempts != 0 {
		t.Errorf("ShorFactor(2^63+7) = {P:%d, Q:%d, Attempts:%d}, want all zero",
			res.P, res.Q, res.Attempts)
	}
}

func TestShorFactor15(t *testing.T) {
	res := ShorFactor(15, 8)
	if res.P == 0 || res.Q == 0 {
		t.Fatalf("ShorFactor(15) failed after %d attempts", res.Attempts)
	}
	if res.P*res.Q != 15 {
		t.Errorf("ShorFactor(15): p=%d q=%d, p*q=%d, want 15", res.P, res.Q, res.P*res.Q)
	}
	if !(res.P == 3 || res.P == 5) {
		t.Errorf("ShorFactor(15): p=%d, want 3 or 5", res.P)
	}
}

func TestShorFactor15Repeated(t *testing.T) {
	// 3 reps; each should succeed within 8 attempts and produce {3,5}.
	for trial := 0; trial < 3; trial++ {
		res := ShorFactor(15, 8)
		if res.P*res.Q != 15 {
			t.Errorf("trial %d: ShorFactor(15) gave p=%d q=%d", trial, res.P, res.Q)
		}
		if !((res.P == 3 && res.Q == 5) || (res.P == 5 && res.Q == 3)) {
			t.Errorf("trial %d: ShorFactor(15) = (%d, %d), want (3,5) or (5,3)",
				trial, res.P, res.Q)
		}
	}
}

func TestShorPeriodA2Mod21(t *testing.T) {
	if os.Getenv("RUN_SHOR_21") == "" {
		t.Skip("set RUN_SHOR_21=1 to run the 16-qubit Shor-21 period test")
	}
	// True period of 2 mod 21 is 6 (2, 4, 8, 16, 11, 1). Continued
	// fraction therefore lands in {1, 2, 3, 6}. Test fixes a=2 to
	// remove random base-selection.
	n := 5
	tBits := 11
	q, _ := NewQreg(tBits+n, WithSeed(1))
	res := q.ApplyShorPeriod(n, tBits, 0, n, 2, 21)
	if res.R == 0 {
		t.Fatalf("Shor-21: period finder returned r=0")
	}
	switch res.R {
	case 1, 2, 3, 6:
		// pass
	default:
		t.Errorf("Shor-21: r=%d not a divisor of 6", res.R)
	}
}
