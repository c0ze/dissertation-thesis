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
