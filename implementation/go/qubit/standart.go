package qubit

// GCD returns gcd(a, b) using Euclid's algorithm. gcd(0, n) = n.
func GCD(a, b uint64) uint64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// addMod returns (a + b) mod mod without overflowing uint64. Plain
// (a+b) % mod can overflow when a+b >= 2^64, which happens for any mod
// above 2^63. We subtract from the modulus instead of adding past it.
//
// Precondition: a < mod, b < mod. Callers in MulMod maintain this.
func addMod(a, b, mod uint64) uint64 {
	// mod - b is well-defined and < mod since b < mod.
	// If a >= mod - b, then a + b >= mod, so wrap by subtracting.
	if a >= mod-b {
		return a - (mod - b)
	}
	return a + b
}

// MulMod returns (a * b) mod mod using double-and-add over addMod.
// Safe for any mod < 2^64 (the entire representable range); the loop
// invariant `result < mod && a < mod` is preserved by addMod every
// iteration, so the body never overflows regardless of mod's magnitude.
func MulMod(a, b, mod uint64) uint64 {
	if mod == 0 {
		return 0
	}
	var result uint64
	a %= mod
	for b > 0 {
		if b&1 == 1 {
			result = addMod(result, a, mod)
		}
		a = addMod(a, a, mod) // doubling step
		b >>= 1
	}
	return result
}

// ModPow returns base^exp mod mod using square-and-multiply over MulMod.
// 0^0 conventionally returns 1.
func ModPow(base, exp, mod uint64) uint64 {
	if mod == 1 {
		return 0
	}
	result := uint64(1)
	base %= mod
	for exp > 0 {
		if exp&1 == 1 {
			result = MulMod(result, base, mod)
		}
		exp >>= 1
		base = MulMod(base, base, mod)
	}
	return result
}

// ContinuedFraction approximates x by num/den with den <= maxDenom,
// using the standard continued-fraction expansion. Used by Shor's
// period finder to recover r from the QFT readout c/2^t.
func ContinuedFraction(x float64, maxDenom uint64) (num, den uint64) {
	if x <= 0 {
		return 0, 1
	}
	var (
		h0, h1 uint64 = 0, 1
		k0, k1 uint64 = 1, 0
	)
	for i := 0; i < 64; i++ {
		ai := uint64(x)
		hNew := ai*h1 + h0
		kNew := ai*k1 + k0
		if kNew > maxDenom {
			return h1, k1
		}
		h0, h1 = h1, hNew
		k0, k1 = k1, kNew
		frac := x - float64(ai)
		if frac < 1e-12 {
			return h1, k1
		}
		x = 1.0 / frac
	}
	return h1, k1
}

// IsPowerOfTwo returns true for positive powers of two: 1, 2, 4, 8, ...
func IsPowerOfTwo(x int) bool {
	return x > 0 && x&(x-1) == 0
}

// Ilog2 returns floor(log2(x)) for x >= 1. Undefined for x == 0.
func Ilog2(x uint32) int {
	n := 0
	for x > 1 {
		x >>= 1
		n++
	}
	return n
}
