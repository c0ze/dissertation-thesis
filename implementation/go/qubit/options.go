package qubit

import "math/rand"

// Option configures a Qreg at construction time. See NewQreg.
type Option func(*Qreg)

// WithSeed pins the RNG used for measurement sampling. Tests use this
// for deterministic measure-based assertions. Production code omits it
// and inherits the default time.Now().UnixNano() seed installed by
// NewQreg.
func WithSeed(seed int64) Option {
	return func(q *Qreg) {
		q.rng = rand.New(rand.NewSource(seed))
	}
}

// WithWorkers overrides the default GOMAXPROCS(0) dispatch fan-out.
// Useful for benchmarking, for pinning a smaller count under
// constrained CI runners, or for forcing workers=1 when narrowing a
// race-detector report. Non-positive values are silently ignored;
// the default stays in effect.
func WithWorkers(n int) Option {
	return func(q *Qreg) {
		if n > 0 {
			q.workers = n
		}
	}
}
