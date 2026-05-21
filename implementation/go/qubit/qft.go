package qubit

import "math"

// ApplyQFT applies the quantum Fourier transform to qubits
// [start, start+n). Includes the final bit-reversal swaps so output
// amplitudes are in natural binary order (same convention as /c).
//
// Big-endian convention: qubit start+n-1 is the most-significant.
func (q *Qreg) ApplyQFT(start, n int) {
	assert(start >= 0 && start+n <= q.nQubits && n >= 1,
		"ApplyQFT: start=%d n=%d out of range for nQubits=%d",
		start, n, q.nQubits)
	for i := n - 1; i >= 0; i-- {
		q.ApplyH(start + i)
		for j := i - 1; j >= 0; j-- {
			theta := math.Pi / float64(int(1)<<uint(i-j))
			q.ApplyControlledPhase(start+j, start+i, theta)
		}
	}
	// Bit-reversal swaps.
	for i := 0; i < n/2; i++ {
		q.ApplySWAP(start+i, start+n-1-i)
	}
}

// ApplyQFTInverse applies the inverse QFT on qubits [start, start+n).
// Includes the bit-reversal swaps at the start so the input is in the
// same natural-binary order ApplyQFT produces.
func (q *Qreg) ApplyQFTInverse(start, n int) {
	assert(start >= 0 && start+n <= q.nQubits && n >= 1,
		"ApplyQFTInverse: start=%d n=%d out of range for nQubits=%d",
		start, n, q.nQubits)
	// Reverse the swaps first.
	for i := 0; i < n/2; i++ {
		q.ApplySWAP(start+i, start+n-1-i)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			theta := -math.Pi / float64(int(1)<<uint(i-j))
			q.ApplyControlledPhase(start+j, start+i, theta)
		}
		q.ApplyH(start + i)
	}
}
