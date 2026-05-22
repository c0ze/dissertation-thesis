package qubit

// OracleFn is the user-supplied phase oracle. It receives the register
// (already prepared in the current Grover step) and a user payload
// (mark index, predicate, etc.); it must flip the sign of every
// amplitude whose basis state satisfies the user's predicate. Most
// callers implement this with a single q.amp[mark] = -q.amp[mark] or a
// loop over targets.
type OracleFn func(q *Qreg, user interface{})

// ApplyGrover applies H to each of the first nQubits qubits, then runs
// `iterations` rounds of oracle + diffusion.
//
// Step 1: applies H^n to produce the uniform superposition.
// Each iteration: oracle (phase mark) + diffusion (2|s><s| - I).
//
// The diffusion operator is implemented as: H^n, X^n,
// multi-controlled-Z, X^n, H^n.
//
// Precondition: the first nQubits qubits must be in |0...0> on entry.
// The Hadamards then produce the uniform superposition the algorithm
// needs; if the register is in any other state, ApplyGrover still runs
// the same gate sequence but the result is no longer "standard Grover."
// Call q.InitBasis(0) first if in doubt. The remaining qubits (indices
// >= nQubits) are not touched and act as inert ancilla space.
func (q *Qreg) ApplyGrover(nQubits int, oracle OracleFn, user interface{},
	iterations int) {
	assert(nQubits > 0 && nQubits <= q.nQubits,
		"ApplyGrover: nQubits=%d out of (0, %d]", nQubits, q.nQubits)
	for i := 0; i < nQubits; i++ {
		q.ApplyH(i)
	}
	for it := 0; it < iterations; it++ {
		oracle(q, user)
		// Diffusion: 2|s><s| - I where |s> is the uniform superposition.
		// Implementation: H^n, X^n, MCZ_n, X^n, H^n.
		for i := 0; i < nQubits; i++ {
			q.ApplyH(i)
		}
		for i := 0; i < nQubits; i++ {
			q.ApplyX(i)
		}
		q.ApplyMultiControlledZ(nQubits)
		for i := 0; i < nQubits; i++ {
			q.ApplyX(i)
		}
		for i := 0; i < nQubits; i++ {
			q.ApplyH(i)
		}
	}
}
