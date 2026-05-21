package qubit

import (
	"math"
	"testing"
)

func TestApplyHTwiceIsIdentity(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(5) // |101>
	q.ApplyH(1)
	q.ApplyH(1)
	for i, a := range q.amp {
		want := complex(0, 0)
		if i == 5 {
			want = complex(1, 0)
		}
		assertAmpNear(t, want, a, "H^2 on |101>")
	}
}

func TestApplyHOnZeroProducesPlusState(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyH(0)
	// Expect (|0> + |1>) / sqrt(2)
	inv2 := complex(1.0/math.Sqrt2, 0)
	assertAmpNear(t, inv2, q.amp[0], "H|0> amp[0]")
	assertAmpNear(t, inv2, q.amp[1], "H|0> amp[1]")
}

func TestApplyUTargetOutOfRangePanics(t *testing.T) {
	q, _ := NewQreg(2)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for ApplyU target=5 on 2-qubit register")
		}
	}()
	q.ApplyU(5, [2][2]complex128{{1, 0}, {0, 1}})
}

func TestApplyXFlips(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(1) // |01>
	q.ApplyX(0)    // flip bit 0 -> |00>
	assertAmpNear(t, complex(1, 0), q.amp[0], "X on |01> amp[0]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "X on |01> amp[1]")
}

func TestApplyZOnOneNegates(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(1)
	q.ApplyZ(0)
	assertAmpNear(t, complex(-1, 0), q.amp[1], "Z|1> amp[1]")
}

func TestApplyYOnZero(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyY(0)
	// Y|0> = i|1>
	assertAmpNear(t, complex(0, 0), q.amp[0], "Y|0> amp[0]")
	assertAmpNear(t, complex(0, 1), q.amp[1], "Y|0> amp[1]")
}
