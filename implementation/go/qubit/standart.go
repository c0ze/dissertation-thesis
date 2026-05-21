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
