package qubit

import (
	"math"
	"testing"
)

func TestGrover1Marked16Qubits(t *testing.T) {
	// 4-qubit search, 1 marked state at |1010> = 10
	n := 4
	target := uint64(10)
	q, _ := NewQreg(n)
	q.InitBasis(0)
	oracle := func(q *Qreg, user interface{}) {
		mark := user.(uint64)
		// Flip phase of |mark> via -1 on amp[mark].
		q.amp[mark] = -q.amp[mark]
	}
	iters := int(math.Pi / 4 * math.Sqrt(float64(int(1)<<n))) // ~3 for n=4
	q.ApplyGrover(n, oracle, target, iters)
	if got := q.ProbOf(target); got < 0.9 {
		t.Errorf("Grover prob[%d] = %v, want >= 0.9 after %d iters", target, got, iters)
	}
}

func TestGroverOverIterationDropsAccuracy(t *testing.T) {
	n := 4
	target := uint64(10)
	q, _ := NewQreg(n)
	q.InitBasis(0)
	oracle := func(q *Qreg, user interface{}) {
		mark := user.(uint64)
		q.amp[mark] = -q.amp[mark]
	}
	q.ApplyGrover(n, oracle, target, 20) // way past optimum
	// Past optimum, probability of the marked state oscillates. Just
	// assert it's not greater than at the optimum.
	if got := q.ProbOf(target); got > 0.95 {
		t.Errorf("over-iteration kept prob high (%v); expected oscillation", got)
	}
}
