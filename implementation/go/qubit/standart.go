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
