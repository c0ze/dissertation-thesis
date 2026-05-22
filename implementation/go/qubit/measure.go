package qubit

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
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
	// Clamp tiny floating-point drift so p0 stays a probability. A
	// normalised state has p0 in [0, 1]; rounding can push it just
	// past either edge over a deep gate sequence, and we'd otherwise
	// take an unintended branch in the outcome decision below.
	if p0 < 0 {
		p0 = 0
	} else if p0 > 1 {
		p0 = 1
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
// uint64 basis index. Panics if the register has zero norm (e.g. a
// fresh allocation that has never been initialised, or one that has
// been multiplied by a non-unitary matrix).
func (q *Qreg) MeasureAll() uint64 {
	norm := q.Norm()
	assert(norm > 0,
		"MeasureAll: zero-norm state (forgot InitBasis, or non-unitary gate?)")
	u := q.rng.Float64() * norm
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
// share any mutable amplitude state.
//
// Note: cloning consumes one value from q.rng to seed the clone, so
// the source register's sampling stream advances by one step per
// Clone() call. If you need a pure non-mutating snapshot, copy
// AmplitudesCopy() into a fresh NewQreg yourself.
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

// SampleDistribution draws `shots` independent samples from the |amp|^2
// distribution and writes the outcomes into out[0..shots). The
// register is NOT collapsed -- the underlying amplitudes are read
// once into a cumulative-distribution table and every shot binary-
// searches into that table. This is O(2^n + shots * log 2^n) instead
// of the O(shots * 2^n) you'd get from clone-then-measure-per-shot.
//
// len(out) must be >= shots. Consumes `shots` values from q.rng.
func (q *Qreg) SampleDistribution(out []uint64, shots int) {
	assert(shots >= 0,
		"SampleDistribution: shots=%d must be >= 0", shots)
	assert(len(out) >= shots,
		"SampleDistribution: len(out)=%d < shots=%d", len(out), shots)
	if shots == 0 {
		return
	}
	cdf := make([]float64, len(q.amp))
	var total float64
	for i, a := range q.amp {
		total += real(a)*real(a) + imag(a)*imag(a)
		cdf[i] = total
	}
	assert(total > 0,
		"SampleDistribution: zero-norm state (forgot InitBasis?)")
	for s := 0; s < shots; s++ {
		u := q.rng.Float64() * total
		idx := sort.Search(len(cdf), func(i int) bool { return cdf[i] > u })
		if idx == len(cdf) {
			idx = len(cdf) - 1
		}
		out[s] = uint64(idx)
	}
}
