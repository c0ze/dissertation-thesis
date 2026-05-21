package qubit

import "testing"

func TestMeasureQubitOnBasisStateIsDeterministic(t *testing.T) {
	q, _ := NewQreg(3, WithSeed(1))
	q.InitBasis(5) // |101>: bit 0 = 1, bit 1 = 0, bit 2 = 1
	if got := q.MeasureQubit(0); got != 1 {
		t.Errorf("MeasureQubit(0) on |101> = %d, want 1", got)
	}
	q.InitBasis(5)
	if got := q.MeasureQubit(1); got != 0 {
		t.Errorf("MeasureQubit(1) on |101> = %d, want 0", got)
	}
	q.InitBasis(5)
	if got := q.MeasureQubit(2); got != 1 {
		t.Errorf("MeasureQubit(2) on |101> = %d, want 1", got)
	}
}

func TestMeasureQubitCollapsesAndRenormalises(t *testing.T) {
	q, _ := NewQreg(1, WithSeed(42))
	q.InitBasis(0)
	q.ApplyH(0) // |+>
	q.MeasureQubit(0)
	// After collapse, norm should be 1.0.
	if got := q.Norm(); abs(got-1.0) > ProbTol {
		t.Errorf("Norm after measure = %v, want 1.0", got)
	}
	// And exactly one of amp[0], amp[1] should be |1| and the other 0.
	a0sq := real(q.amp[0])*real(q.amp[0]) + imag(q.amp[0])*imag(q.amp[0])
	a1sq := real(q.amp[1])*real(q.amp[1]) + imag(q.amp[1])*imag(q.amp[1])
	if abs(a0sq+a1sq-1.0) > ProbTol {
		t.Errorf("post-collapse: a0sq+a1sq = %v, want 1.0", a0sq+a1sq)
	}
	if !((abs(a0sq-1) < ProbTol && abs(a1sq) < ProbTol) ||
		(abs(a0sq) < ProbTol && abs(a1sq-1) < ProbTol)) {
		t.Errorf("post-collapse not a basis state: a0sq=%v, a1sq=%v", a0sq, a1sq)
	}
}

func TestMeasureAllOnBasisStateIsDeterministic(t *testing.T) {
	q, _ := NewQreg(3, WithSeed(1))
	q.InitBasis(6)
	if got := q.MeasureAll(); got != 6 {
		t.Errorf("MeasureAll on |6> = %d, want 6", got)
	}
}

func TestMeasureAllCollapses(t *testing.T) {
	q, _ := NewQreg(2, WithSeed(7))
	q.InitBasis(0)
	q.ApplyH(0)
	q.ApplyH(1)
	got := q.MeasureAll()
	// Post-measure, q.amp[got] should be ~1 and everyone else 0.
	for i, a := range q.amp {
		mag := real(a)*real(a) + imag(a)*imag(a)
		if i == int(got) {
			if abs(mag-1.0) > ProbTol {
				t.Errorf("post-measure amp[%d] magnitude = %v, want 1", i, mag)
			}
		} else {
			if mag > ProbTol {
				t.Errorf("post-measure amp[%d] magnitude = %v, want 0", i, mag)
			}
		}
	}
}

func TestCloneIsIndependent(t *testing.T) {
	q, _ := NewQreg(3, WithSeed(7))
	q.InitBasis(2)
	c := q.Clone()
	c.ApplyX(0) // mutate the clone
	// Original amp[2] should still be 1.
	assertAmpNear(t, complex(1, 0), q.amp[2], "original after clone mutation")
}

func TestSampleDistributionPanicsOnNegativeShots(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(0)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for shots=-1")
		}
	}()
	out := make([]uint64, 10)
	q.SampleDistribution(out, -1)
}

func TestSampleDistributionPreservesOriginal(t *testing.T) {
	q, _ := NewQreg(2, WithSeed(11))
	q.InitBasis(0)
	q.ApplyH(0)
	q.ApplyH(1) // uniform over {0,1,2,3}
	out := make([]uint64, 1000)
	q.SampleDistribution(out, 1000)
	// Count each outcome; with 1000 shots over 4 outcomes, each should
	// be roughly 250 +/- 60 (3-sigma).
	counts := [4]int{}
	for _, v := range out {
		if v >= 4 {
			t.Fatalf("got outcome %d outside [0,4)", v)
		}
		counts[v]++
	}
	for i, c := range counts {
		if c < 150 || c > 350 {
			t.Errorf("outcome %d count = %d, expected near 250", i, c)
		}
	}
	// Original should still be uniform (norm 1, four equal amplitudes).
	if abs(q.Norm()-1.0) > ProbTol {
		t.Errorf("original norm after sampling = %v, want 1.0", q.Norm())
	}
}
