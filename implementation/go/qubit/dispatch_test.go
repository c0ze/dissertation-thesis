package qubit

import (
	"sync/atomic"
	"testing"
)

func TestParallelOverPairsCoversAllIndices(t *testing.T) {
	q, _ := NewQreg(8) // 256 amps, 128 pairs
	nPairs := 1 << (q.nQubits - 1)
	visited := make([]int32, nPairs)
	q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			atomic.AddInt32(&visited[i], 1)
		}
	})
	for i, n := range visited {
		if n != 1 {
			t.Fatalf("pair-index %d visited %d times, want 1", i, n)
		}
	}
}

func TestParallelOverIndicesCoversAllIndices(t *testing.T) {
	q, _ := NewQreg(6)
	n := len(q.amp)
	visited := make([]int32, n)
	q.parallelOverIndices(n, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			atomic.AddInt32(&visited[i], 1)
		}
	})
	for i, c := range visited {
		if c != 1 {
			t.Fatalf("index %d visited %d times, want 1", i, c)
		}
	}
}

func TestDispatcherRespectsWorkersOption(t *testing.T) {
	q, _ := NewQreg(8, WithWorkers(2))
	var maxActive int32
	var active int32
	q.parallelOverPairs(1<<7, func(amp []complex128, lo, hi int) {
		cur := atomic.AddInt32(&active, 1)
		for {
			old := atomic.LoadInt32(&maxActive)
			if cur <= old || atomic.CompareAndSwapInt32(&maxActive, old, cur) {
				break
			}
		}
		atomic.AddInt32(&active, -1)
	})
	if maxActive > 2 {
		t.Errorf("max concurrent workers = %d, want <= 2", maxActive)
	}
}
