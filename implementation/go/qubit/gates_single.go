package qubit

import "math"

// ApplyU applies the single-qubit 2x2 unitary u to the target qubit.
// Iterates over pairs (i0, i1) that differ only in the target bit;
// parallelised via parallelOverPairs.
func (q *Qreg) ApplyU(target int, u [2][2]complex128) {
	assert(target >= 0 && target < q.nQubits,
		"ApplyU: target=%d out of [0, %d)", target, q.nQubits)
	nPairs := 1 << (q.nQubits - 1)
	tBit := uint(target)
	q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			lower := i & ((1 << tBit) - 1)
			upper := (i >> tBit) << (tBit + 1)
			i0 := upper | lower
			i1 := i0 | (1 << tBit)
			a0, a1 := amp[i0], amp[i1]
			amp[i0] = u[0][0]*a0 + u[0][1]*a1
			amp[i1] = u[1][0]*a0 + u[1][1]*a1
		}
	})
}

// ApplyH applies the Hadamard gate to the target qubit.
//
//	H = (1/sqrt(2)) [[1,  1], [1, -1]]
func (q *Qreg) ApplyH(target int) {
	inv2 := complex(1.0/math.Sqrt2, 0)
	q.ApplyU(target, [2][2]complex128{
		{inv2, inv2},
		{inv2, -inv2},
	})
}

// ApplyX applies the Pauli-X (NOT) gate.
//
//	X = [[0, 1], [1, 0]]
func (q *Qreg) ApplyX(target int) {
	q.ApplyU(target, [2][2]complex128{
		{0, 1},
		{1, 0},
	})
}

// ApplyY applies the Pauli-Y gate.
//
//	Y = [[0, -i], [i, 0]]
func (q *Qreg) ApplyY(target int) {
	q.ApplyU(target, [2][2]complex128{
		{0, complex(0, -1)},
		{complex(0, 1), 0},
	})
}

// ApplyZ applies the Pauli-Z gate.
//
//	Z = [[1, 0], [0, -1]]
func (q *Qreg) ApplyZ(target int) {
	q.ApplyU(target, [2][2]complex128{
		{1, 0},
		{0, -1},
	})
}
