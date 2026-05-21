package qubit

import (
	"math"
	"testing"
)

func TestApplyQFTOnSingleQubitEqualsH(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyQFT(0, 1)
	inv2 := complex(1/math.Sqrt2, 0)
	assertAmpNear(t, inv2, q.amp[0], "QFT|0> on 1 qubit amp[0]")
	assertAmpNear(t, inv2, q.amp[1], "QFT|0> on 1 qubit amp[1]")
}

func TestQFTOfZeroIsUniformSuperposition(t *testing.T) {
	n := 4
	q, _ := NewQreg(n)
	q.InitBasis(0)
	q.ApplyQFT(0, n)
	want := complex(1/math.Sqrt(float64(int(1)<<n)), 0)
	for i, a := range q.amp {
		assertAmpNear(t, want, a, "QFT|0> uniform amp")
		_ = i
	}
}

func TestQFTRoundTrip(t *testing.T) {
	n := 4
	for basis := uint64(0); basis < uint64(1<<n); basis++ {
		q, _ := NewQreg(n)
		q.InitBasis(basis)
		q.ApplyQFT(0, n)
		q.ApplyQFTInverse(0, n)
		for i, a := range q.amp {
			want := complex(0, 0)
			if uint64(i) == basis {
				want = complex(1, 0)
			}
			assertAmpNear(t, want, a, "QFT round-trip amp")
		}
	}
}

func TestQFTDetectsPeriod(t *testing.T) {
	// Build a state with period 4 over 4 qubits: |0> + |4> + |8> + |12>,
	// normalised. After QFT we expect non-trivial probability mass on
	// indices that are multiples of N/period = 16/4 = 4: indices 0, 4,
	// 8, 12.
	n := 4
	q, _ := NewQreg(n)
	for i := range q.amp {
		q.amp[i] = 0
	}
	amp := complex(0.5, 0)
	q.amp[0], q.amp[4], q.amp[8], q.amp[12] = amp, amp, amp, amp
	q.ApplyQFT(0, n)
	// Sum probability at multiples of 4; should be near 1.
	var pPeak float64
	for k := 0; k < 16; k += 4 {
		pPeak += q.ProbOf(uint64(k))
	}
	if pPeak < 0.99 {
		t.Errorf("QFT period detection: peak prob = %v, want >= 0.99", pPeak)
	}
}
