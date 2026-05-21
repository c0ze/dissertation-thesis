package qubit

import (
	"math/rand"
	"testing"
)

func TestWithSeedReplacesRng(t *testing.T) {
	q := &Qreg{rng: rand.New(rand.NewSource(1))}
	WithSeed(42)(q)
	// Two calls with the same seed should be identical.
	r1 := q.rng.Int63()
	WithSeed(42)(q)
	r2 := q.rng.Int63()
	if r1 != r2 {
		t.Errorf("WithSeed: expected reproducible draws, got %d vs %d", r1, r2)
	}
}

func TestWithWorkersOverrides(t *testing.T) {
	q := &Qreg{workers: 1}
	WithWorkers(8)(q)
	if q.workers != 8 {
		t.Errorf("WithWorkers(8): q.workers = %d, want 8", q.workers)
	}
}

func TestWithWorkersIgnoresNonPositive(t *testing.T) {
	q := &Qreg{workers: 4}
	WithWorkers(0)(q)
	if q.workers != 4 {
		t.Errorf("WithWorkers(0) should be no-op; q.workers = %d, want 4", q.workers)
	}
	WithWorkers(-3)(q)
	if q.workers != 4 {
		t.Errorf("WithWorkers(-3) should be no-op; q.workers = %d, want 4", q.workers)
	}
}
