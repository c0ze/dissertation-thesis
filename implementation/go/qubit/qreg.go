package qubit

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"
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

// NewQreg constructs a Qreg of nQubits qubits, with default
// GOMAXPROCS(0) workers and a time-seeded RNG. Apply opts in order to
// override defaults; opts are infallible (see options.go).
//
// Returns (nil, error) if nQubits is outside [1, QregMaxQubits]. All
// other validation failures (out-of-range qubit index on a gate call,
// etc.) panic via the assert helper -- see spec §4.4 for the policy.
func NewQreg(nQubits int, opts ...Option) (*Qreg, error) {
	if nQubits < 1 || nQubits > QregMaxQubits {
		return nil, fmt.Errorf("qubit: NewQreg: nQubits=%d out of [1, %d]",
			nQubits, QregMaxQubits)
	}
	q := &Qreg{
		amp:     make([]complex128, 1<<nQubits),
		nQubits: nQubits,
		workers: runtime.GOMAXPROCS(0),
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	for _, opt := range opts {
		opt(q)
	}
	return q, nil
}

// NQubits returns the number of qubits this register represents.
func (q *Qreg) NQubits() int { return q.nQubits }

// InitBasis collapses the register to the computational basis state |basis>.
// All other amplitudes are zeroed; amp[basis] is set to 1 + 0i.
// Panics if basis is outside [0, 1<<nQubits).
func (q *Qreg) InitBasis(basis uint64) {
	assert(basis < uint64(len(q.amp)),
		"InitBasis: basis=%d out of [0, %d)", basis, len(q.amp))
	for i := range q.amp {
		q.amp[i] = 0
	}
	q.amp[basis] = complex(1, 0)
}
