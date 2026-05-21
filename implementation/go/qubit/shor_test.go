package qubit

import "testing"

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
