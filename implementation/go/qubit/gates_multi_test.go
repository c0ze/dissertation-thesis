package qubit

import (
	"math"
	"testing"
)

func TestApplyMultiControlledZFlipsOnlyAllOnes(t *testing.T) {
	q, _ := NewQreg(3)
	// Build (|0>+|1>)^3 / sqrt(8)
	q.InitBasis(0)
	for i := 0; i < 3; i++ {
		q.ApplyH(i)
	}
	q.ApplyMultiControlledZ(3)
	// Only amp[7] should be negated; everything else stays at 1/sqrt(8).
	inv := complex(1/math.Sqrt(8), 0)
	for i := 0; i < 8; i++ {
		want := inv
		if i == 7 {
			want = -inv
		}
		assertAmpNear(t, want, q.amp[i], "MCZ amp")
	}
}

func TestApplyMultiControlledXAsToffoli(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(3) // |011>: controls=1,1, target initially 0
	q.ApplyMultiControlledX([]int{0, 1}, 2)
	// Toffoli flips target -> |111> = 7
	assertAmpNear(t, complex(1, 0), q.amp[7], "Toffoli on |011>: amp[7]")
	assertAmpNear(t, complex(0, 0), q.amp[3], "Toffoli on |011>: amp[3]")
}

func TestApplyMultiControlledXSkipsWhenAnyControlZero(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(1) // |001>: control 0 is 1, control 1 is 0
	q.ApplyMultiControlledX([]int{0, 1}, 2)
	// Should not flip: |001> stays
	assertAmpNear(t, complex(1, 0), q.amp[1], "Toffoli on |001>: amp[1] unchanged")
}
