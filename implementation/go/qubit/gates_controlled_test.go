package qubit

import (
	"math"
	"testing"
)

func TestApplyCNOTLeavesZeroControlAlone(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(2) // |10>: control=0, target=1
	q.ApplyCNOT(0, 1)
	assertAmpNear(t, complex(1, 0), q.amp[2], "CNOT control=0: amp[2]")
}

func TestApplyCNOTFlipsTargetWhenControlIsOne(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(1) // |01>: control=1, target=0
	q.ApplyCNOT(0, 1)
	// Flips target (bit 1) -> |11> = 3
	assertAmpNear(t, complex(1, 0), q.amp[3], "CNOT control=1: amp[3]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "CNOT control=1: amp[1]")
}

func TestBellStateFromHCNOT(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(0)
	q.ApplyH(0)
	q.ApplyCNOT(0, 1)
	// |Phi+> = (|00> + |11>) / sqrt(2)
	inv2 := complex(1/math.Sqrt2, 0)
	assertAmpNear(t, inv2, q.amp[0], "Bell amp[0]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "Bell amp[1]")
	assertAmpNear(t, complex(0, 0), q.amp[2], "Bell amp[2]")
	assertAmpNear(t, inv2, q.amp[3], "Bell amp[3]")
}

func TestApplyCUPanicsOnControlEqualsTarget(t *testing.T) {
	q, _ := NewQreg(2)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for ApplyCU control == target")
		}
	}()
	q.ApplyCU(1, 1, [2][2]complex128{{1, 0}, {0, 1}})
}
