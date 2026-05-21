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

func TestAmplitudeReadsBack(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(2)
	if got := q.Amplitude(2); got != complex(1, 0) {
		t.Errorf("Amplitude(2) = %v, want (1+0i)", got)
	}
	if got := q.Amplitude(3); got != complex(0, 0) {
		t.Errorf("Amplitude(3) = %v, want (0+0i)", got)
	}
}

func TestAmplitudePanicsOutOfRange(t *testing.T) {
	q, _ := NewQreg(2)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for Amplitude(4)")
		}
	}()
	q.Amplitude(4)
}

func TestAmplitudesCopyIsIndependent(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(0)
	c := q.AmplitudesCopy()
	c[0] = complex(99, 0) // mutate the copy
	if q.amp[0] != complex(1, 0) {
		t.Errorf("AmplitudesCopy returned an aliased slice: q.amp[0] = %v", q.amp[0])
	}
}

func TestProbOfBasisState(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(3)
	if got := q.ProbOf(3); got != 1.0 {
		t.Errorf("ProbOf(3) = %v, want 1.0", got)
	}
	if got := q.ProbOf(0); got != 0.0 {
		t.Errorf("ProbOf(0) = %v, want 0.0", got)
	}
}

func TestNormOnBasisStateIsOne(t *testing.T) {
	q, _ := NewQreg(4)
	q.InitBasis(7)
	if got := q.Norm(); abs(got-1.0) > ProbTol {
		t.Errorf("Norm = %v, want 1.0", got)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
