package qubit

import (
	"math/cmplx"
	"testing"
)

const AmpTol = 1e-10
const ProbTol = 1e-9

func TestNewQregValid(t *testing.T) {
	for _, n := range []int{1, 2, 8, 16} {
		q, err := NewQreg(n)
		if err != nil {
			t.Fatalf("NewQreg(%d) returned err: %v", n, err)
		}
		if q.NQubits() != n {
			t.Errorf("NewQreg(%d).NQubits() = %d", n, q.NQubits())
		}
		if got, want := len(q.amp), 1<<n; got != want {
			t.Errorf("amp len = %d, want %d", got, want)
		}
	}
}

func TestNewQregRejectsOutOfRange(t *testing.T) {
	cases := []int{0, -1, QregMaxQubits + 1, 100}
	for _, n := range cases {
		_, err := NewQreg(n)
		if err == nil {
			t.Errorf("NewQreg(%d): want error, got nil", n)
		}
	}
}

func TestNewQregAppliesOptions(t *testing.T) {
	q, err := NewQreg(4, WithWorkers(2), WithSeed(42))
	if err != nil {
		t.Fatal(err)
	}
	if q.workers != 2 {
		t.Errorf("WithWorkers(2): q.workers = %d", q.workers)
	}
	if q.rng == nil {
		t.Errorf("WithSeed: q.rng is nil")
	}
}

func TestInitBasisCollapsesToBasisState(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(5) // |101>
	for i, a := range q.amp {
		want := complex(0, 0)
		if i == 5 {
			want = complex(1, 0)
		}
		if cmplx.Abs(a-want) > AmpTol {
			t.Errorf("amp[%d] = %v, want %v", i, a, want)
		}
	}
}

func TestInitBasisRejectsOutOfRange(t *testing.T) {
	q, _ := NewQreg(3)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for InitBasis(8)")
		}
	}()
	q.InitBasis(8) // out of range for 3-qubit register
}
