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

func TestQFTOnSubrangePreservesOutsideBits(t *testing.T) {
	// ApplyQFT(start=1, n=3) on a 5-qubit register should:
	//   - QFT the middle 3 qubits (indices 1, 2, 3)
	//   - leave qubits 0 and 4 unchanged
	//
	// Input: |10110> = basis index 0b10110 = 22.
	//   bit 0 = 0  (outside subrange)
	//   bits 1..3 = 011 = 3 (inside subrange, input to QFT)
	//   bit 4 = 1  (outside subrange)
	//
	// Expected output amplitudes: for every basis k in [0, 8), set
	// bit 0 = 0 and bit 4 = 1, then put k into bits 1..3. The
	// amplitude at that position should be exp(2 pi i * 3 * k / 8) / sqrt(8).
	// Every other basis state must have zero amplitude.
	q, _ := NewQreg(5)
	q.InitBasis(22) // 0b10110
	q.ApplyQFT(1, 3)

	N := float64(8)
	invSqrt := 1.0 / math.Sqrt(N)
	const subrangeInput = 3 // bits 1..3 of input

	outsideMask := uint64(0b10001) // bit 0 + bit 4 set per input
	for i := uint64(0); i < 32; i++ {
		// Decompose i into (outside bits) and (subrange bits in [1,3]).
		outside := i & outsideMask
		k := (i >> 1) & 0b111

		var want complex128
		if outside == 0b10000 { // bit 4 = 1, bit 0 = 0 from input
			theta := 2.0 * math.Pi * float64(subrangeInput) * float64(k) / N
			want = complex(invSqrt*math.Cos(theta), invSqrt*math.Sin(theta))
		} else {
			want = 0
		}
		assertAmpNear(t, want, q.amp[i],
			"QFT subrange[1,3] of |10110>")
	}
}

func TestQFTOnBasisOneMatchesAnalyticPhases(t *testing.T) {
	// QFT|1> on 3 qubits = (1/sqrt(8)) sum_k exp(2 pi i k / 8) |k>.
	//
	// Phase-sensitive: we check both real and imaginary parts of every
	// amplitude against the analytical DFT. Probability tests miss
	// phase errors -- a faulty QFT that scrambled the phases but
	// preserved magnitudes would pass TestQFTOfZeroIsUniformSuperposition
	// without complaint. This catches them.
	n := 3
	q, _ := NewQreg(n)
	q.InitBasis(1)
	q.ApplyQFT(0, n)

	N := float64(int(1) << n)
	invSqrtN := 1.0 / math.Sqrt(N)
	for k := 0; k < int(N); k++ {
		theta := 2.0 * math.Pi * float64(k) / N
		want := complex(invSqrtN*math.Cos(theta), invSqrtN*math.Sin(theta))
		assertAmpNear(t, want, q.amp[k], "QFT|1> analytic phase amp")
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
