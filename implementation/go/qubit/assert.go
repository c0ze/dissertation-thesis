// Package qubit implements a sparse-gate quantum-circuit simulator
// parallelised with goroutines. See docs/superpowers/specs/
// 2026-05-21-implementation-go-design.md for the architecture
// rationale and the thesis-claim mapping.
package qubit

import "fmt"

// assert panics with "qubit: " + the formatted message if cond is false.
// Used for programmer-error preconditions (out-of-range qubit indices,
// control == target, etc.). For recoverable errors -- bad nQubits input
// to NewQreg -- the function returns an error directly instead.
func assert(cond bool, format string, args ...interface{}) {
	if !cond {
		panic(fmt.Errorf("qubit: "+format, args...))
	}
}
