package qubit

import "math"

// ApplyCU applies the controlled-U gate: when the control bit is 1,
// apply the 2x2 unitary u to the target bit; otherwise leave the
// amplitude pair unchanged.
//
// Shared memory means the four locality cases /c needs (both local,
// both global, control-local-target-global, target-local-control-global)
// collapse into a single uniform loop.
func (q *Qreg) ApplyCU(control, target int, u [2][2]complex128) {
	assert(control >= 0 && control < q.nQubits,
		"ApplyCU: control=%d out of [0, %d)", control, q.nQubits)
	assert(target >= 0 && target < q.nQubits,
		"ApplyCU: target=%d out of [0, %d)", target, q.nQubits)
	assert(control != target,
		"ApplyCU: control == target == %d", control)
	nPairs := 1 << (q.nQubits - 1)
	cMask := 1 << uint(control)
	tBit := uint(target)
	q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			lower := i & ((1 << tBit) - 1)
			upper := (i >> tBit) << (tBit + 1)
			i0 := upper | lower
			if i0&cMask == 0 {
				continue // control bit must be 1
			}
			i1 := i0 | (1 << tBit)
			a0, a1 := amp[i0], amp[i1]
			amp[i0] = u[0][0]*a0 + u[0][1]*a1
			amp[i1] = u[1][0]*a0 + u[1][1]*a1
		}
	})
}

// ApplyCNOT applies the controlled-NOT gate.
//
//	CNOT = ApplyCU with X = [[0,1],[1,0]]
func (q *Qreg) ApplyCNOT(control, target int) {
	q.ApplyCU(control, target, [2][2]complex128{
		{0, 1},
		{1, 0},
	})
}

// ApplyCZ applies the controlled-Z gate.
func (q *Qreg) ApplyCZ(control, target int) {
	q.ApplyCU(control, target, [2][2]complex128{
		{1, 0},
		{0, -1},
	})
}

// ApplyControlledPhase applies a phase rotation by theta on the |11> branch.
func (q *Qreg) ApplyControlledPhase(control, target int, theta float64) {
	q.ApplyCU(control, target, [2][2]complex128{
		{1, 0},
		{0, complex(math.Cos(theta), math.Sin(theta))},
	})
}

// ApplySWAP exchanges the contents of two qubits. SWAP = CNOT(a,b)
// then CNOT(b,a) then CNOT(a,b). Same decomposition as /c.
func (q *Qreg) ApplySWAP(a, b int) {
	assert(a != b, "ApplySWAP: a == b == %d", a)
	q.ApplyCNOT(a, b)
	q.ApplyCNOT(b, a)
	q.ApplyCNOT(a, b)
}
