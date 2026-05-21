package qubit

import (
	"math/rand"
	"time"
)

// ShorPeriodResult is the return value of ApplyShorPeriod.
type ShorPeriodResult struct {
	R         uint64 // recovered period (0 on failure)
	MeasuredC uint64 // raw counting-register outcome (for debugging)
}

// ShorFactorResult is the return value of ShorFactor.
type ShorFactorResult struct {
	P, Q     uint64 // non-trivial factors of N (0 on failure)
	Attempts int    // number of period-finding rounds used
}

// ApplyModularExp applies the modular-exponentiation oracle:
//
//	(x, y) -> (x, a^x * y mod N)    for y < N
//	(x, y) -> (x, y)                 for y >= N
//
// where x lives in counting bits [countingStart, countingStart+t) and
// y lives in target bits [targetStart, targetStart+n).
//
// This is a permutation of computational basis states, so workers
// writing to disjoint output cells never collide.
func (q *Qreg) ApplyModularExp(countingStart, t, targetStart, n int,
	a, N uint64) {
	assert(t >= 1 && n >= 1,
		"ApplyModularExp: t=%d n=%d must be >= 1", t, n)
	assert(countingStart >= 0 && countingStart+t <= q.nQubits,
		"ApplyModularExp: counting range [%d, %d) out of [0, %d)",
		countingStart, countingStart+t, q.nQubits)
	assert(targetStart >= 0 && targetStart+n <= q.nQubits,
		"ApplyModularExp: target range [%d, %d) out of [0, %d)",
		targetStart, targetStart+n, q.nQubits)
	// counting and target must not overlap
	cEnd := countingStart + t
	tEnd := targetStart + n
	assert(cEnd <= targetStart || tEnd <= countingStart,
		"ApplyModularExp: counting [%d, %d) and target [%d, %d) overlap",
		countingStart, cEnd, targetStart, tEnd)
	assert(N >= 2, "ApplyModularExp: N=%d must be >= 2", N)
	assert(N <= uint64(1)<<uint(n),
		"ApplyModularExp: N=%d exceeds target capacity 2^%d", N, n)
	assert(GCD(a, N) == 1, "ApplyModularExp: GCD(a=%d, N=%d) != 1", a, N)

	newAmp := make([]complex128, len(q.amp))
	tMask := (uint64(1) << uint(t)) - 1
	nMask := (uint64(1) << uint(n)) - 1
	clearMask := (tMask << countingStart) | (nMask << targetStart)

	q.parallelOverIndices(len(q.amp), func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			if amp[i] == 0 {
				continue
			}
			x := (uint64(i) >> countingStart) & tMask
			y := (uint64(i) >> targetStart) & nMask
			var yNew uint64
			if y < N {
				yNew = MulMod(y, ModPow(a, x, N), N)
			} else {
				yNew = y
			}
			outer := uint64(i) &^ clearMask
			iNew := outer | (x << countingStart) | (yNew << targetStart)
			newAmp[iNew] = amp[i] // safe: permutation, no contention
		}
	})
	q.amp = newAmp
}

// ApplyShorPeriod runs Shor's quantum period-finding subroutine:
//
//  1. InitBasis(1 << targetStart) -- counting=0, target=1
//  2. H^t on counting register
//  3. ApplyModularExp(counting, target, a, N)
//  4. ApplyQFTInverse on counting register
//  5. Measure counting register -> integer c in [0, 2^t)
//  6. Recover candidate period r via continued-fraction expansion of c/2^t
//
// Returns the recovered period and the raw measured counting value
// (for debugging). r=0 indicates failure (continued-fraction did not
// produce a denominator in the valid range).
func (q *Qreg) ApplyShorPeriod(countingStart, t, targetStart, n int,
	a, N uint64) ShorPeriodResult {
	// Init: counting=0, target=1 -- so amp at basis (1 << targetStart) = 1.
	q.InitBasis(uint64(1) << uint(targetStart))
	for i := 0; i < t; i++ {
		q.ApplyH(countingStart + i)
	}
	q.ApplyModularExp(countingStart, t, targetStart, n, a, N)
	q.ApplyQFTInverse(countingStart, t)
	full := q.MeasureAll()
	tMask := (uint64(1) << uint(t)) - 1
	c := (full >> uint(countingStart)) & tMask
	// Continued-fraction recovery: r = denominator of best approximation
	// of c/2^t with denominator <= N.
	x := float64(c) / float64(uint64(1)<<uint(t))
	_, den := ContinuedFraction(x, N)
	return ShorPeriodResult{R: den, MeasuredC: c}
}

// ShorFactor attempts to factor N via Shor's algorithm. Picks random
// bases a coprime to N until period finding produces a usable r
// (non-zero, even, and a^{r/2} != -1 mod N), then derives a factor
// via gcd(a^{r/2} +/- 1, N). Returns {0, 0, attempts} on total failure.
//
// Allocates its own Qreg (size 2*ceil(log2 N) + 1, the standard width).
// This is the one entry point that is a package function rather than a
// method on *Qreg -- it has no "current register" to be a method on.
func ShorFactor(N uint64, maxAttempts int) ShorFactorResult {
	if N < 2 {
		return ShorFactorResult{P: 0, Q: 0, Attempts: 0}
	}
	if N%2 == 0 {
		return ShorFactorResult{P: 2, Q: N / 2, Attempts: 0}
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	n := 0
	for (uint64(1) << uint(n)) < N {
		n++
	}
	tBits := 2*n + 1
	nTotal := tBits + n
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		// Pick random a in [2, N-1] coprime to N.
		var a uint64
		for {
			a = uint64(rng.Intn(int(N-2))) + 2
			if g := GCD(a, N); g != 1 {
				// Lucky shortcut: gcd already a non-trivial factor.
				return ShorFactorResult{P: g, Q: N / g, Attempts: attempt}
			}
			break
		}
		q, err := NewQreg(nTotal)
		if err != nil {
			return ShorFactorResult{P: 0, Q: 0, Attempts: attempt}
		}
		res := q.ApplyShorPeriod(n, tBits, 0, n, a, N)
		r := res.R
		if r == 0 || r%2 != 0 {
			continue
		}
		half := ModPow(a, r/2, N)
		if half == N-1 {
			continue
		}
		p := GCD(half+1, N)
		if p > 1 && p < N {
			return ShorFactorResult{P: p, Q: N / p, Attempts: attempt}
		}
		p = GCD(half-1, N)
		if p > 1 && p < N {
			return ShorFactorResult{P: p, Q: N / p, Attempts: attempt}
		}
	}
	return ShorFactorResult{P: 0, Q: 0, Attempts: maxAttempts}
}
