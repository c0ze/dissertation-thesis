package qubit

import "sync"

// parallelOverPairs splits [0, nPairs) into chunks across q.workers
// goroutines, invokes fn on each chunk, and joins via wg.Wait.
//
// Callers using pair-index iteration over single- or two-qubit gates
// must pass the *pair* count, not the amp count (typically
// 1 << (nQubits - 1) for single-qubit gates). See §4.1 of the spec
// for the pair-index math.
//
// fn receives the amp slice the dispatcher snapshotted at entry. If a
// later gate swaps q.amp (notably ApplyModularExp, §5.5), the next
// dispatch picks up the new slice; in-flight chunks keep the slice
// they were sent with.
//
// fn must not capture and reuse the slice across calls.
func (q *Qreg) parallelOverPairs(nPairs int, fn chunkFn) {
	workers := q.workers
	if workers > nPairs {
		workers = nPairs
	}
	if workers <= 0 {
		return
	}
	chunkSize := (nPairs + workers - 1) / workers
	amp := q.amp
	var wg sync.WaitGroup
	for c := 0; c < workers; c++ {
		lo := c * chunkSize
		hi := lo + chunkSize
		if hi > nPairs {
			hi = nPairs
		}
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(amp, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// parallelOverIndices splits [0, nIndices) into chunks across
// q.workers goroutines and joins via wg.Wait. Mechanics identical to
// parallelOverPairs; the difference is the caller's promise that fn
// operates on absolute amp-index ranges (used by ApplyModularExp,
// which is a permutation of basis states rather than a pair gate).
func (q *Qreg) parallelOverIndices(nIndices int, fn chunkFn) {
	workers := q.workers
	if workers > nIndices {
		workers = nIndices
	}
	if workers <= 0 {
		return
	}
	chunkSize := (nIndices + workers - 1) / workers
	amp := q.amp
	var wg sync.WaitGroup
	for c := 0; c < workers; c++ {
		lo := c * chunkSize
		hi := lo + chunkSize
		if hi > nIndices {
			hi = nIndices
		}
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(amp, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}
