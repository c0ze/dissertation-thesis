package qubit

import (
	"fmt"
	"io"
	"math"
	"math/rand"
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

// MeasureAll samples a full basis index from the |amp|^2 distribution
// and collapses the register onto |outcome>. Returns the outcome as a
// uint64 basis index.
func (q *Qreg) MeasureAll() uint64 {
	u := q.rng.Float64()
	var cum float64
	chosen := uint64(len(q.amp) - 1) // default to last index for numerical edge
	for i, a := range q.amp {
		cum += real(a)*real(a) + imag(a)*imag(a)
		if u < cum {
			chosen = uint64(i)
			break
		}
	}
	for i := range q.amp {
		q.amp[i] = 0
	}
	q.amp[chosen] = complex(1, 0)
	return chosen
}

// Clone returns an independent Qreg with the same amplitudes, worker
// count, and a freshly seeded RNG. The original and the clone do not
// share any mutable state.
func (q *Qreg) Clone() *Qreg {
	c := &Qreg{
		amp:     make([]complex128, len(q.amp)),
		nQubits: q.nQubits,
		workers: q.workers,
		rng:     rand.New(rand.NewSource(q.rng.Int63())),
	}
	copy(c.amp, q.amp)
	return c
}

// SampleDistribution runs `shots` independent measurements on a clone
// of the register and writes the outcomes into out[0..shots). The
// original q is unmodified. len(out) must be >= shots.
func (q *Qreg) SampleDistribution(out []uint64, shots int) {
	assert(len(out) >= shots,
		"SampleDistribution: len(out)=%d < shots=%d", len(out), shots)
	for s := 0; s < shots; s++ {
		c := q.Clone()
		out[s] = c.MeasureAll()
	}
}
