package qubit

// GCD returns gcd(a, b) using Euclid's algorithm. gcd(0, n) = n.
func GCD(a, b uint64) uint64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
