package qubit

import (
	"math/rand"
)

// QregMaxQubits is the public construction ceiling. See spec §4 for the
// per-qubit memory progression (n=25 -> 512 MiB amp, n=26 -> 1 GiB amp,
// n=29 -> won't fit on 16 GiB laptop). Diverges from /c's 60: /c's bound
// is shift-overflow defensive, Go hits the OS allocator long before 60.
const QregMaxQubits = 26

// Qreg is a state-vector quantum register parallelised with goroutines
// inside each gate primitive.
//
// Concurrency contract: a *Qreg is NOT safe for concurrent method calls
// from multiple goroutines. The amplitude slice is shared-mutable and
// rand.Rand is not goroutine-safe. Different *Qregs are independent.
// `go test -race` catches intra-gate races but does not certify the type
// as concurrent-user-safe.
type Qreg struct {
	amp     []complex128 // contiguous state vector, len = 1 << nQubits
	nQubits int          // 1 .. QregMaxQubits

	workers int        // dispatch fan-out; default runtime.GOMAXPROCS(0)
	rng     *rand.Rand // measurement sampling
}

// chunkFn is the signature every closure passed to the parallel
// dispatcher satisfies. `amp` is the amplitude slice the closure should
// operate on (snapshotted by the dispatcher at call entry). `[lo, hi)`
// is a half-open work range -- pair-index for parallelOverPairs,
// absolute amp-index for parallelOverIndices.
type chunkFn func(amp []complex128, lo, hi int)
