package qubit

// ApplyMultiControlledZ negates the amplitude where the lowest n qubits
// are all 1. Equivalent to a phase oracle marking |1...1> in the first
// n qubits; used by Grover's diffusion step.
func (q *Qreg) ApplyMultiControlledZ(n int) {
	assert(n > 0 && n <= q.nQubits,
		"ApplyMultiControlledZ: n=%d out of (0, %d]", n, q.nQubits)
	mask := uint64((1 << uint(n)) - 1)
	// Iterate over amp indices; flip any whose lowest n bits are all 1.
	nIdx := len(q.amp)
	q.parallelOverIndices(nIdx, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			if uint64(i)&mask == mask {
				amp[i] = -amp[i]
			}
		}
	})
}

// ApplyMultiControlledX flips the target qubit when every control qubit is 1.
// Generalises Toffoli; len(controls) == 2 is Toffoli, == 1 is CNOT.
func (q *Qreg) ApplyMultiControlledX(controls []int, target int) {
	assert(target >= 0 && target < q.nQubits,
		"ApplyMultiControlledX: target=%d out of [0, %d)", target, q.nQubits)
	var cMask uint64
	for _, c := range controls {
		assert(c >= 0 && c < q.nQubits,
			"ApplyMultiControlledX: control=%d out of [0, %d)", c, q.nQubits)
		assert(c != target,
			"ApplyMultiControlledX: control %d == target", c)
		cMask |= 1 << uint(c)
	}
	tBit := uint(target)
	tMask := uint64(1) << tBit
	nPairs := 1 << (q.nQubits - 1)
	q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			lower := i & ((1 << tBit) - 1)
			upper := (i >> tBit) << (tBit + 1)
			i0 := uint64(upper | lower)
			if i0&cMask != cMask {
				continue
			}
			i1 := i0 | tMask
			amp[i0], amp[i1] = amp[i1], amp[i0]
		}
	})
}
