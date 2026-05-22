// This file deliberately uses `package qubit_test` instead of
// `package qubit` so it can only see the package's exported API. It
// exists to catch API regressions where an exported function's
// documented usage pattern actually relies on unexported fields, which
// same-package tests would silently miss.
//
// Right now it covers the Grover oracle pattern (which used to tell
// callers to do `q.amp[mark] = -q.amp[mark]` and would have broken
// for any external user) and the NewQreg-already-initialised
// invariant. Add a new test here every time the README documents a
// usage pattern that an external user is expected to follow.
package qubit_test

import (
	"math"
	"testing"

	"github.com/c0ze/dissertation-thesis/implementation/go/qubit"
)

func TestNewQregExternallyVisibleNormIsOne(t *testing.T) {
	// NewQreg must produce a valid quantum state straight away (norm 1,
	// concentrated on |0...0>) so external users can apply gates
	// without an explicit InitBasis(0). This is the invariant the
	// README documents; verify it through the exported API only.
	q, err := qubit.NewQreg(3)
	if err != nil {
		t.Fatalf("NewQreg(3): unexpected error: %v", err)
	}
	if got, want := q.Norm(), 1.0; math.Abs(got-want) > 1e-12 {
		t.Fatalf("fresh register Norm() = %v, want 1", got)
	}
	if got, want := q.ProbOf(0), 1.0; math.Abs(got-want) > 1e-12 {
		t.Fatalf("fresh register ProbOf(0) = %v, want 1", got)
	}
	for i := uint64(1); i < 8; i++ {
		if got := q.ProbOf(i); got > 1e-12 {
			t.Fatalf("fresh register ProbOf(%d) = %v, want 0", i, got)
		}
	}
}

func TestGroverWithExternalOracleViaFlipPhase(t *testing.T) {
	// Reproduces the README's documented external-Grover pattern:
	// import the package, define an oracle that calls the EXPORTED
	// FlipPhase method, run ApplyGrover. This would have failed before
	// FlipPhase existed because q.amp is unexported.
	n := 4
	N := uint64(1) << uint(n)
	mark := uint64(11) // arbitrary marked basis state

	q, err := qubit.NewQreg(n)
	if err != nil {
		t.Fatalf("NewQreg(%d): %v", n, err)
	}

	oracle := func(q *qubit.Qreg, user any) {
		q.FlipPhase(user.(uint64))
	}

	iters := int(math.Pi / 4.0 * math.Sqrt(float64(N))) // textbook optimum
	q.ApplyGrover(n, oracle, mark, iters)

	// Expected amplification: P(mark) ~ 0.96 for n=4, 1 marked, 3 iters.
	if got := q.ProbOf(mark); got < 0.9 {
		t.Fatalf("Grover via external FlipPhase oracle: ProbOf(%d) = %v, want >= 0.9", mark, got)
	}
}
