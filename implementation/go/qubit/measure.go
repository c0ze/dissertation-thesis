package qubit

import (
	"fmt"
	"io"
	"math"
)

// MeasureQubit performs a projective measurement on the target qubit
// in the computational basis, returning 0 or 1. The register is
// collapsed onto the measured branch and renormalised.
//
// Uses q.rng (seeded by time.Now() at construction, or via WithSeed).
func (q *Qreg) MeasureQubit(target int) int {
	assert(target >= 0 && target < q.nQubits,
		"MeasureQubit: target=%d out of [0, %d)", target, q.nQubits)
	tBit := uint(target)
	// p0 = sum over basis states with bit=0 of |amp|^2
	var p0 float64
	for i, a := range q.amp {
		if uint(i)>>tBit&1 == 0 {
			p0 += real(a)*real(a) + imag(a)*imag(a)
		}
	}
	// Sample with [0, 1).
	u := q.rng.Float64()
	outcome := 0
	if u >= p0 {
		outcome = 1
	}
	// Project + renormalise.
	var norm float64
	if outcome == 0 {
		norm = math.Sqrt(p0)
	} else {
		norm = math.Sqrt(1.0 - p0)
	}
	if norm == 0 {
		// Numerical edge: cannot happen if the sample is consistent,
		// but defend against it rather than dividing by zero.
		return outcome
	}
	for i, a := range q.amp {
		bit := int(uint(i) >> tBit & 1)
		if bit == outcome {
			q.amp[i] = complex(real(a)/norm, imag(a)/norm)
		} else {
			q.amp[i] = 0
		}
	}
	return outcome
}

// Dump writes |i>: amp_i lines for every basis index with nonzero
// amplitude. Diagnostic only.
func (q *Qreg) Dump(w io.Writer) {
	for i, a := range q.amp {
		if real(a) != 0 || imag(a) != 0 {
			fmt.Fprintf(w, "|%d>: %v\n", i, a)
		}
	}
}
