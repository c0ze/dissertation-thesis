# Design: implementation/go --- goroutine-parallel quantum simulator

**Date:** 2026-05-21
**Status:** Round-7 revision applied (MiB/GB consistency throughout; "~8x cohabitation headroom" replaced with concrete "2 GiB peak vs 16 GiB physical RAM"). Awaiting written spec review.
**Owner:** Arda Karaduman
**Author of this doc:** Claude

## 1. Goal and scope

Build a Go implementation of the sparse-gate quantum-circuit simulator at
`implementation/go/`, covering the same thesis claims as `implementation/c/`
(register construction and operations, all gates, measurement, QFT,
Grover, Shor). Use Go's
goroutines for concurrency instead of MPI --- spawned per call from
`parallelOverPairs`/`parallelOverIndices` and joined via
`sync.WaitGroup`, no persistent pool to manage. Target 25 qubits
comfortably on a 16 GiB laptop. Single-process by design --- no
distributed nodes, no MPI.

This is one of two alternative implementations being added alongside `/c`
to demonstrate that the same thesis claims hold under different
concurrency models. The Python/PyTorch GPU version is a separate
subproject with its own spec.

## 2. Constraints and conventions

| Constraint | Value | Source |
|---|---|---|
| Concurrency model | Per-call goroutines spawned by the parallel dispatcher (`parallelOverPairs`, `parallelOverIndices`) and joined via `sync.WaitGroup`. No persistent pool, no channels. Goroutine launch is microseconds; even the lightest 25-qubit gate is tens of milliseconds of useful work, so the dispatch overhead disappears into the noise. | User decision (brainstorm Q2 + spec round 4) |
| API style | Idiomatic Go. Every operation that mutates a register's state is a method on `*Qreg` (gates, measurement, QFT, Grover, ModularExp, ShorPeriod). Package-level functions are reserved for entry points that *create* their own register (`ShorFactor` is the only one in v1). Apply-prefix used uniformly for state-mutating methods (`ApplyH`, `ApplyQFT`, `ApplyShorPeriod`). Construction takes functional options (`NewQreg(n, WithSeed(42), WithWorkers(8))`) instead of post-construction mutators. | User decision (brainstorm Q3), tightened in spec round 3, options added in round 4 |
| Target scale | 25 qubits comfortably (laptop, 512 MiB state vector; 1 GiB peak during ApplyModularExp) | User decision (brainstorm Q1) |
| Module layout | Library under `qubit/`, binary under `cmd/qubit/`, tests adjacent as `_test.go` | Standard Go convention |
| Validation policy | Programmer errors (out-of-range qubit index, invalid preconditions) `panic` with a formatted message. Construction errors (invalid `nQubits`) return `error` from `NewQreg`. Library code never calls `os.Exit`; the CLI binary in `cmd/qubit` is the only place that translates panics into process exit codes. | User decision (spec round 4); diverges from `/c` deliberately to fit Go convention |
| Sequential fallback | None --- every gate goes through `parallelOverPairs`/`parallelOverIndices` regardless of register size. With per-call goroutines, the dispatch overhead is negligible even on 4-qubit toy tests. | User decision |
| Test framework | Go stdlib `testing` package, one `*_test.go` per source file | Go convention |
| Race detection | `go test -race` required in CI | Defensive |
| Qubit indexing | 0-indexed from LSB; same as `/c` | Inherited from thesis §5 |
| Tensor convention | Big-endian: qubit `n-1` is the most-significant bit | Thesis §5.2 |
| Complex type | Go stdlib `complex128` | Native, no external dep |
| Numeric tolerances | `AmpTol = 1e-10`, `ProbTol = 1e-9` --- match `/c` | Same as `/c` spec §7.5 |

## 3. File layout

```
implementation/go/
├── go.mod, go.sum
├── qubit/                          # package qubit
│   ├── qreg.go                     # Qreg, NewQreg, NQubits, InitBasis, ProbOf, Amplitude, AmplitudesCopy, Norm
│   ├── options.go                  # Option, WithSeed, WithWorkers
│   ├── dispatch.go                 # parallelOverPairs, parallelOverIndices (per-call goroutines + WaitGroup)
│   ├── gates_single.go             # ApplyU + H, X, Y, Z, S, T, Phase, Rx, Ry, Rz
│   ├── gates_controlled.go         # ApplyCU + CNOT, CZ, ControlledPhase, SWAP
│   ├── gates_multi.go              # ApplyMultiControlledZ, ApplyMultiControlledX
│   ├── measure.go                  # MeasureAll, MeasureQubit, SampleDistribution, Clone, Dump
│   ├── qft.go                      # QFT, QFTInverse
│   ├── grover.go                   # Grover + OracleFn callback
│   ├── shor.go                     # ModularExp, ShorPeriod, ShorFactor + result structs
│   ├── standart.go                 # GCD, ModPow, ContinuedFraction, IsPowerOfTwo, Ilog2
│   ├── assert.go                   # assert(cond, format, args...) helper
│   ├── qreg_test.go
│   ├── gates_single_test.go
│   ├── gates_controlled_test.go
│   ├── gates_multi_test.go
│   ├── measure_test.go
│   ├── qft_test.go
│   ├── grover_test.go
│   └── shor_test.go
├── cmd/
│   └── qubit/
│       └── main.go                 # --algo {bell|qft|grover|shor}
├── Makefile                        # thin wrapper: build, test, test-race, demo, bench, fmt, vet, clean
├── README.md
└── assessment.md
```

`build/` (Go's default cache lives outside the source tree, but any
local artifacts from `make demo` go here) is gitignored.

The `standart.go` filename intentionally matches `/c`'s misspelling
so the cross-implementation parallel is visually obvious.

## 4. Data model

```go
// QregMaxQubits is the public construction ceiling. It is sized to
// what NewQreg can reliably succeed on a 16 GiB laptop alongside the
// OS, the Go runtime, and a typical IDE/browser footprint -- not to
// the theoretical bit-shift bound (which would be ~62) nor to the
// "biggest slice make() will accept in isolation" (which would be
// ~30). At 16 bytes per complex128 amplitude:
//
//     n=25 -> 512 MiB amp,   1 GiB ModularExp peak  (thesis target)
//     n=26 -> 1 GiB amp,     2 GiB ModularExp peak  (this ceiling)
//     n=27 -> 2 GiB amp,     4 GiB ModularExp peak
//     n=28 -> 4 GiB amp,     8 GiB ModularExp peak  (cohabiting risk)
//     n=29 -> 8 GiB amp,    16 GiB ModularExp peak  (won't fit)
//
// 26 is one step above the 25-qubit thesis target -- a single qubit
// of headroom for ad-hoc experimentation. 2 GiB peak working set
// against 16 GiB total physical RAM leaves the rest (~14 GiB minus
// the OS, the Go runtime, and whatever the user has open) free for
// cohabitation; even with a heavy IDE and a browser there is room
// to spare. Going higher would advertise a construction that the
// OS allocator can refuse mid-run, surfacing as a runtime panic
// from make() rather than a clean error from NewQreg -- exactly
// the API smell §4.4 is set up to avoid.
//
// (Diverges from /c's QREG_MAX_QUBITS=60. /c uses 60 as a defensive
// shift-overflow bound on size_t; Go hits the OS allocator long
// before that, so surfacing 60 as the public ceiling would be
// fiction. Tests never approach this ceiling either way; the largest
// register the test suite builds is the 16-qubit Shor-21 register.)
const QregMaxQubits = 26

type Qreg struct {
    amp     []complex128   // contiguous state vector, len = 1 << nQubits
    nQubits int            // 1 .. QregMaxQubits

    workers int            // dispatch fan-out; default runtime.GOMAXPROCS(0)
    rng     *rand.Rand     // measurement sampling
}

// chunkFn is the signature every closure passed to the parallel dispatcher
// satisfies. `amp` is the amplitude slice the closure should operate on
// (snapshotted by the dispatcher at call entry so any later q.amp swap
// is invisible to in-flight chunks). `[lo, hi)` is a half-open work
// range -- pair-index for parallelOverPairs, absolute amp-index for
// parallelOverIndices. The dispatcher promises chunks never overlap;
// the closure is iteration-style-agnostic.
type chunkFn func(amp []complex128, lo, hi int)
```

The struct is intentionally minimal. All fields are unexported. Same-package code (every file under `qubit/`) reads `q.amp` and `q.nQubits` directly; external callers go through accessors (§6.1). There is no persistent worker pool, no channel handles, no `WaitGroup` living on the struct, no `sync.Once` guarding teardown -- the dispatcher creates and joins its goroutines per call (§4.3), so the only lifetime concern is the amplitude slice itself.

**Concurrency contract.** A `*Qreg` is **not** safe for concurrent
method calls from multiple goroutines. The amplitude slice is
shared-mutable and `rand.Rand` is itself not goroutine-safe; the
dispatcher's internal fan-out is already serialised by the caller's
`wg.Wait()`, so within a single gate the workers are race-free, but
two simultaneous gate calls on the same `*Qreg` would race on
`q.amp`. Callers that need parallelism across registers create
independent `*Qreg`s; gates on different registers are fully
independent. `go test -race` is required in CI to catch *intra-gate*
races (a gate primitive accidentally writing to overlapping cells, a
new option mutating shared state without a lock); it does **not**
imply the type itself is concurrent-user-safe.

### 4.1 Pair-index iteration

Every gate processes amplitudes in pairs. There are `1 << (NQubits-1)`
pairs total. Each pair $(i_0, i_1)$ differs in exactly one bit (the
target's bit for single-qubit gates, or two bits for two-qubit gates).

We chunk over pair-index $i \in [0, 1 << (NQubits-1))$, not amp-index.
For a single-qubit gate on `target`, each pair-index $i$ maps to:

```go
lower := i & ((1 << target) - 1)
upper := (i >> target) << (target + 1)
i0    := upper | lower
i1    := i0 | (1 << target)
```

This insertion-of-a-zero-bit at `target`'s position means workers can
chunk over $i$ with any boundary they like --- pair endpoints never
straddle a chunk. Trivial parallelism, no alignment concerns.

### 4.2 Construction

`func NewQreg(nQubits int, opts ...Option) (*Qreg, error)`:

1. Validate `1 <= nQubits <= QregMaxQubits`; return `(nil, error)` if not.
   This is a *recoverable* input error (the caller picked the value),
   so it goes through `error` rather than `panic`.
2. Allocate `amp := make([]complex128, 1<<nQubits)` (Go zero-initialises).
3. Initialise defaults:
   * `workers = runtime.GOMAXPROCS(0)`
   * `rng     = rand.New(rand.NewSource(time.Now().UnixNano()))`
4. Apply each option in order: `for _, opt := range opts { opt(q) }`.
   Options may override `workers` and/or `rng`. See §6.1a for the
   `WithSeed` and `WithWorkers` definitions.
5. Return `(&Qreg{...}, nil)`.

```go
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
```

**No `Destroy()`.** Earlier drafts kept a persistent worker pool that
required teardown. The present design spawns goroutines per call
inside the dispatcher (§4.3) and joins them all with `sync.WaitGroup`
before returning, so no resources outlive any single gate
invocation. The garbage collector reclaims the `Qreg` itself
whenever the caller drops the last reference -- no finalizer, no
manual cleanup, no destroy-then-use undefined-behaviour footgun, no
`defer q.Destroy()` boilerplate in every test. This deliberately
diverges from `/c`'s `qreg_destroy` requirement; in Go, lifetime
management goes through the GC unless there is a concrete external
resource (file, socket, mmap) to release. A simulator's state vector
is just a slice and has none.

The trade-off the persistent pool was meant to avoid -- per-gate
goroutine launch cost -- only matters when gate work is itself
microseconds. At the 25-qubit target, even a single `ApplyH` iterates
over `2^24 = ~16M` pair-indices (hundreds of milliseconds of
arithmetic); launching `GOMAXPROCS` goroutines once per call adds a
few microseconds. Three orders of magnitude apart, so the pool
optimises nothing in the regime that matters. If a future benchmark
ever flips the cost picture, a package-level pool can be reintroduced
without changing the public API -- the dispatcher is purely internal.

### 4.3 parallelOverPairs

```go
func (q *Qreg) parallelOverPairs(nPairs int, fn chunkFn) {
    workers := q.workers
    if workers > nPairs { workers = nPairs }
    if workers <= 0 { return }   // degenerate; nothing to do
    chunkSize := (nPairs + workers - 1) / workers
    // Snapshot q.amp at dispatch entry. ModularExp (§5.5) is the only
    // method that swaps q.amp wholesale, and it does so *after* its
    // own parallelOverIndices Wait returns -- so the snapshot is
    // defensive rather than load-bearing, but it makes the contract
    // ("workers operate on the slice the dispatcher saw at entry")
    // explicit and removes the question entirely.
    amp := q.amp
    var wg sync.WaitGroup
    for c := 0; c < workers; c++ {
        lo := c * chunkSize
        hi := lo + chunkSize
        if hi > nPairs { hi = nPairs }
        if lo >= hi { break }
        wg.Add(1)
        go func(lo, hi int) {
            defer wg.Done()
            fn(amp, lo, hi)
        }(lo, hi)
    }
    wg.Wait()
}
```

The dispatcher spawns at most `workers` goroutines per call and joins
them all before returning. No channels, no persistent state.

If `fn` panics, the Go runtime prints the panic and stack trace and
terminates the process -- exactly the fail-fast behaviour the spec
wants for programmer-error paths. The persistent-pool design needed a
defer/recover/`os.Exit` dance just to prevent the dispatcher from
deadlocking on a missed completion signal; the per-call design needs
none of that. `wg.Done()` is in a `defer` so even a panicking worker
decrements the counter, and Go's default goroutine-panic behaviour
crashes the program cleanly with the stack trace already printed.

#### parallelOverIndices --- for permutation gates

Pair-index iteration is correct for any gate that processes pairs of
amplitudes (every single-qubit gate, every controlled gate). It is
**not** correct for gates that move every amplitude individually (e.g.,
Shor's ModularExp, which is a permutation of the full basis). Those
gates need a sibling helper that chunks over the global amp-index
range `[0, len(amp))`:

```go
func (q *Qreg) parallelOverIndices(nIndices int, fn chunkFn) {
    workers := q.workers
    if workers > nIndices { workers = nIndices }
    if workers <= 0 { return }
    chunkSize := (nIndices + workers - 1) / workers
    amp := q.amp
    var wg sync.WaitGroup
    for c := 0; c < workers; c++ {
        lo := c * chunkSize
        hi := lo + chunkSize
        if hi > nIndices { hi = nIndices }
        if lo >= hi { break }
        wg.Add(1)
        go func(lo, hi int) {
            defer wg.Done()
            fn(amp, lo, hi)
        }(lo, hi)
    }
    wg.Wait()
}
```

The two helpers are mechanically identical; they differ only in what
contract the caller's closure satisfies (pair-index vs. amp-index).
Naming is enough to keep the two iteration styles separate at every
call site.

### 4.4 Validation and bounds

`QregMaxQubits = 26` -- rationale at the constant's definition in
§4. The ceiling is sized to what `NewQreg` can reliably succeed on
a 16 GiB laptop alongside the OS and Go runtime: 1 GiB amp slice,
2 GiB ModularExp peak working set, leaving the remaining ~14 GiB
of physical RAM (minus whatever the OS and the user's other tools
occupy) free for cohabitation. Diverges deliberately from `/c`'s
60 (which is a shift-overflow bound, not an allocation bound).
Validation falls into two layers, and they deliberately use
different mechanisms:

* **Construction errors** -- bad `nQubits` argument to `NewQreg` --
  return `error`. The caller picked the value, so they get a
  recoverable result they can branch on.
  ```go
  q, err := qubit.NewQreg(32)
  if err != nil { /* nQubits exceeds QregMaxQubits */ }
  ```
  Options themselves are infallible in v1 -- `type Option func(*Qreg)`
  with no error return. `WithWorkers(0)` is silently ignored (the
  default stays in effect) rather than failing the construction, so
  there is no "invalid option" failure mode to surface. If a future
  option grows a genuine validation step, the cleanest path is to
  change the type to `func(*Qreg) error` and have `NewQreg` propagate;
  no caller would break, because errors from `NewQreg` already need
  branching.

* **Programmer errors** -- out-of-range qubit indices passed into a
  gate method, `control == target`, register-overlap violations in
  `ApplyModularExp` -- `panic` with a formatted message. These are
  bugs in the caller's code, not user-facing failures, and the right
  response is "fix the bug", not "handle the error". A library that
  silently degrades when its preconditions are violated produces
  worse code in the long run.

A single helper lives in `assert.go`:

```go
// assert panics with "qubit: " + the formatted message if cond is false.
// Used for programmer-error preconditions (out-of-range qubit indices,
// control == target, etc.). For recoverable errors -- bad input to
// NewQreg, malformed Option values -- the function returns an error
// directly instead.
func assert(cond bool, format string, args ...interface{}) {
    if !cond {
        panic(fmt.Errorf("qubit: "+format, args...))
    }
}
```

The library itself **never** calls `os.Exit`. That call lives in
exactly one place: the CLI binary in `cmd/qubit/main.go`, which
recovers from a panic, prints the panic value, and exits non-zero
(§8.2). Tests that exercise the panic path use the standard `recover`
pattern; see the negative-case entries in §7.2.

A future soft-error variant (`Try*` methods returning errors on every
gate call) is out of v1 scope. The thesis-claim coverage does not
need it, and adding it would double the method count without serving
any actual user.

## 5. Gate model

### 5.1 Single-qubit gates

`(*Qreg).ApplyU(target int, u [2][2]complex128)` is the workhorse;
named gates (`ApplyH`, `ApplyX`, etc.) construct their 2x2 and call
through. ApplyU:

```go
func (q *Qreg) ApplyU(target int, u [2][2]complex128) {
    assert(target >= 0 && target < q.nQubits,
           "ApplyU: target=%d out of [0, %d)", target, q.nQubits)
    nPairs := 1 << (q.nQubits - 1)
    tBit := uint(target)
    q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
        for i := lo; i < hi; i++ {
            lower := i & ((1 << tBit) - 1)
            upper := (i >> tBit) << (tBit + 1)
            i0    := upper | lower
            i1    := i0 | (1 << tBit)
            a0, a1 := amp[i0], amp[i1]
            amp[i0] = u[0][0]*a0 + u[0][1]*a1
            amp[i1] = u[1][0]*a0 + u[1][1]*a1
        }
    })
}
```

Specialisations: ApplyH, ApplyX, ApplyY, ApplyZ, ApplyS, ApplyT,
ApplyPhase (theta), ApplyRx (theta), ApplyRy (theta), ApplyRz (theta).
All trivial wrappers around ApplyU with the appropriate 2x2.

### 5.2 Controlled gates

`(*Qreg).ApplyCU(control, target int, u [2][2]complex128)`. Same
chunked pair-iteration but the closure checks the control bit. Because
the state vector is shared memory, the four locality cases that `/c`
needs collapse into a single uniform loop:

```go
func (q *Qreg) ApplyCU(control, target int, u [2][2]complex128) {
    assert(control >= 0 && control < q.nQubits,
           "ApplyCU: control=%d out of [0, %d)", control, q.nQubits)
    assert(target  >= 0 && target  < q.nQubits,
           "ApplyCU: target=%d out of [0, %d)",  target,  q.nQubits)
    assert(control != target,
           "ApplyCU: control == target == %d", control)
    nPairs := 1 << (q.nQubits - 1)
    cMask  := 1 << uint(control)
    tBit   := uint(target)
    q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
        for i := lo; i < hi; i++ {
            lower := i & ((1 << tBit) - 1)
            upper := (i >> tBit) << (tBit + 1)
            i0    := upper | lower
            if i0 & cMask == 0 { continue }   // control bit must be 1
            i1    := i0 | (1 << tBit)
            a0, a1 := amp[i0], amp[i1]
            amp[i0] = u[0][0]*a0 + u[0][1]*a1
            amp[i1] = u[1][0]*a0 + u[1][1]*a1
        }
    })
}
```

ApplyCNOT, ApplyCZ, ApplyControlledPhase wrap ApplyCU with the standard
2x2s. ApplySWAP decomposes into three CNOTs (same as `/c`).

### 5.3 Multi-controlled

`(*Qreg).ApplyMultiControlledZ(n int)` phase-flips the single all-ones
amplitude across the first `n` qubits. Implementation walks the local
slice and negates the amplitude(s) where the lower `n` bits are all 1.
Parallel via parallelOverPairs is overkill for a single-amplitude flip
but kept uniform for simplicity.

`(*Qreg).ApplyMultiControlledX(controls []int, target int)` generalised
Toffoli; parallel iteration over pair-index, gate amp swap where every
control bit is 1 and target bit is 0.

### 5.4 Measurement

* `ProbOf(basis uint64) float64`: direct lookup; no reduction.
* `Norm() float64`: parallel reduction over `Amp`, sum of |a|^2.
* `MeasureQubit(target int) int`: parallel reduce for $p_0$, sample,
  parallel project + renormalise.
* `MeasureAll() uint64`: parallel partial sum over chunks, CPU prefix
  scan over chunks, sample uniform $u$, find chosen chunk + offset,
  collapse.
* `SampleDistribution(out []uint64, shots int)`: Clone-and-MeasureAll
  per shot; same shape as `/c`.
* `Clone() *Qreg`, `Dump(w io.Writer)` for diagnostics.

### 5.5 Shor's modular exponentiation

ModularExp is **not** a pair-update gate. It is a permutation of
computational basis states: the map
$(x, y) \to (x, (a^x y) \bmod N)$ for $y < N$ and $(x, y) \to (x, y)$
for $y \ge N$ is a bijection on $\{0, \ldots, 2^{n_\text{total}}-1\}$.
That means each output index gets contributions from exactly one input
index, so workers writing to disjoint output cells **never collide**.

`(*Qreg).ApplyModularExp(countingStart, t, targetStart, n int, a, N uint64)`:

1. Allocate `newAmp := make([]complex128, len(q.amp))` (Go zeroes it).
2. Use the **`parallelOverIndices`** helper (see §4.3) to iterate
   over input indices in `[0, len(q.amp))`:

    ```go
    q.parallelOverIndices(len(q.amp), func(amp []complex128, lo, hi int) {
        for i := lo; i < hi; i++ {
            if amp[i] == 0 { continue }
            x := (uint64(i) >> countingStart) & ((1 << uint(t)) - 1)
            y := (uint64(i) >> targetStart)   & ((1 << uint(n)) - 1)
            var yNew uint64
            if y < N {
                // Use MulMod, NOT a plain `y * ModPow(...) % N`. For N
                // close to 2^32 the product overflows uint64 before the
                // modulus reduces it. MulMod is double-and-add over
                // addMod (see standart.go §6.9) and is safe for any
                // N < 2^64 -- no implicit dependence on QregMaxQubits.
                yNew = MulMod(y, ModPow(a, x, N), N)
            } else {
                yNew = y           // reversibility pass-through, spec §5.5 of /c
            }
            // Reassemble the destination global index.
            outer := uint64(i) &^ (
                (((uint64(1) << uint(t)) - 1) << countingStart) |
                (((uint64(1) << uint(n)) - 1) << targetStart))
            iNew := outer |
                    (x    << countingStart) |
                    (yNew << targetStart)
            newAmp[iNew] = amp[i]    // safe: permutation, no contention
        }
    })
    ```

3. `q.amp = newAmp`. The parallelOverIndices call has already
   `wg.Wait`ed for every worker, so no goroutine is still holding a
   reference to the old slice when the assignment happens; the old
   slice's backing array becomes garbage and is reclaimed on the next
   GC cycle.

The shared-memory shortcut means `redistribute_pairs` from `/c`
disappears entirely. **Peak memory is exactly `2 * len(amp)` complex
doubles** (old + new state vector) regardless of worker count --- no
per-worker scratch. At 25 qubits that is 1 GiB total, well within the
16 GiB laptop target.

Pre-conditions, asserted at the top of ApplyModularExp via the
`assert(...)` helper (§4.4):

* `t >= 1`, `n >= 1`, counting and target register ranges non-overlapping
* `countingStart + t <= q.nQubits`, `targetStart + n <= q.nQubits`
* `N >= 2`
* `N <= 1 << n` (otherwise `(a^x y) mod N` could exceed the target register)
* `GCD(a, N) == 1`

(The `q != nil` precondition from earlier drafts is automatic in Go:
calling a method on a nil `*Qreg` panics with a nil-pointer
dereference before the method body runs.)

## 6. Module-by-module surface

### 6.1 qubit.Qreg, qubit.NewQreg, qubit.QregMaxQubits

All declared in `qreg.go` per §4. The struct's fields are
**unexported** (§4); external callers go through the accessors below.
Same-package internals (gates, dispatcher, measurement, QFT, Grover,
Shor) read `q.amp` and `q.nQubits` directly.

* `func NewQreg(nQubits int, opts ...Option) (*Qreg, error)` --- §4.2
* `(*Qreg).NQubits() int`              --- returns the qubit count
* `(*Qreg).InitBasis(basis uint64)`    --- collapse to a basis state
* `(*Qreg).Norm() float64`             --- sum of |amp[i]|^2
* `(*Qreg).ProbOf(basis uint64) float64` --- |amp[basis]|^2, bounds-checked
* `(*Qreg).Amplitude(i uint64) complex128` --- single bounds-checked read
* `(*Qreg).AmplitudesCopy() []complex128`  --- defensive copy for diagnostics

There is intentionally **no** `Amplitudes() []complex128` returning
the live slice: that would hand external callers a mutable reference
to the simulator's internal state, defeat the unexported-field
discipline, and create race-detector noise if the caller reads while
another goroutine triggers a gate. Tests and diagnostics that need
the full vector use `AmplitudesCopy()`; production code reads
specific indices via `Amplitude(i)` or `ProbOf(basis)`.

There is no `Destroy()`. The Qreg holds no resources outside the GC
(§4.2); the caller simply drops the last reference when done.

### 6.1a Options (options.go)

```go
// Option configures a Qreg at construction time.
type Option func(*Qreg)

// WithSeed pins the RNG used for measurement sampling. Tests use
// this for deterministic measure-based assertions. Production code
// omits it and inherits the default time.Now().UnixNano() seed.
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
```

Options are applied in order after `NewQreg` installs defaults
(workers from GOMAXPROCS, RNG from time.Now), so the caller's value
wins. This replaces the earlier `(*Qreg).SeedRNG(int64)` mutator,
which leaked test-only behaviour into the production API surface.

### 6.2 Single-qubit gates (gates_single.go)

* `(*Qreg).ApplyU(target int, u [2][2]complex128)`
* `(*Qreg).ApplyH(target int)`, `ApplyX`, `ApplyY`, `ApplyZ`, `ApplyS`, `ApplyT`
* `(*Qreg).ApplyPhase(target int, theta float64)`
* `(*Qreg).ApplyRx(target int, theta float64)`, `ApplyRy`, `ApplyRz`

### 6.3 Two-qubit gates (gates_controlled.go)

* `(*Qreg).ApplyCU(control, target int, u [2][2]complex128)`
* `(*Qreg).ApplyCNOT(control, target int)`
* `(*Qreg).ApplyCZ(control, target int)`
* `(*Qreg).ApplyControlledPhase(control, target int, theta float64)`
* `(*Qreg).ApplySWAP(a, b int)`

### 6.4 Multi-controlled (gates_multi.go)

* `(*Qreg).ApplyMultiControlledZ(n int)`
* `(*Qreg).ApplyMultiControlledX(controls []int, target int)`

### 6.5 Measurement and utilities (measure.go)

* `(*Qreg).MeasureAll() uint64`
* `(*Qreg).MeasureQubit(target int) int`
* `(*Qreg).SampleDistribution(out []uint64, shots int)`
* `(*Qreg).Clone() *Qreg`
* `(*Qreg).Dump(w io.Writer)`

### 6.6 QFT (qft.go)

```go
func (q *Qreg) ApplyQFT       (start, n int)
func (q *Qreg) ApplyQFTInverse(start, n int)
```

Methods (consistent with the gates) since both mutate q's amplitude
slice and conceptually act on the register. Both include the final
bit-reversal swaps so output amplitudes are in natural binary order,
same convention as `/c`. An earlier draft of this spec listed them as
package functions taking `q *Qreg`; that contradicted the §2
constraint, see the round-3 note in the constraints table.

### 6.7 Grover (grover.go)

```go
type OracleFn func(q *Qreg, user interface{})

func (q *Qreg) ApplyGrover(nQubits int, oracle OracleFn, user interface{},
                           iterations int)
```

### 6.8 Shor (shor.go)

```go
type ShorPeriodResult struct {
    R          uint64   // recovered period (0 on failure)
    MeasuredC  uint64   // raw counting-register outcome
}

type ShorFactorResult struct {
    P, Q     uint64    // non-trivial factors of N (0 on failure)
    Attempts int       // number of period-finding rounds used
}

// Methods: both mutate q's amplitude slice.
func (q *Qreg) ApplyModularExp(countingStart, t, targetStart, n int,
                               a, N uint64)

func (q *Qreg) ApplyShorPeriod(countingStart, t, targetStart, n int,
                               a, N uint64) ShorPeriodResult

// Package function: allocates its own Qreg (or several across
// attempts), so it has no "current register" to be a method on.
func ShorFactor(N uint64, maxAttempts int) ShorFactorResult
```

### 6.9 standart.go (helpers)

* `GCD(a, b uint64) uint64`
* `addMod(a, b, mod uint64) uint64` (package-private) --- overflow-safe
  `(a+b) mod mod`. Plain `(a+b) % mod` overflows when `a+b >= 2^64`,
  which can happen for any `mod` above `2^63`. Implementation avoids
  that by subtracting from the modulus instead of adding past it:

  ```go
  // Precondition: a < mod, b < mod (callers in MulMod maintain this).
  func addMod(a, b, mod uint64) uint64 {
      // mod - b is well-defined and < mod since b < mod.
      // If a >= mod - b, then a + b >= mod, so wrap by subtracting.
      if a >= mod - b {
          return a - (mod - b)
      }
      return a + b
  }
  ```

  Works for any `mod` strictly less than `2^64` --- the entire useful
  range. No reliance on QregMaxQubits-derived bounds.

* `MulMod(a, b, mod uint64) uint64` --- overflow-safe `(a*b) mod mod`
  via double-and-add over `addMod`. Required because Go has no native
  `__uint128_t` (the trick `/c`'s ModPow used). Reference body:

  ```go
  func MulMod(a, b, mod uint64) uint64 {
      if mod == 0 { return 0 }       // defensive; gate primitives never call with mod=0
      var result uint64
      a %= mod
      for b > 0 {
          if b & 1 == 1 {
              result = addMod(result, a, mod)
          }
          a = addMod(a, a, mod)      // doubling step
          b >>= 1
      }
      return result
  }
  ```

  Loop invariant: `result < mod` and `a < mod` after every iteration
  (`addMod` preserves both). The body therefore never overflows
  regardless of `mod`'s magnitude --- it is safe for any
  `mod < 2^64`, which is the entire representable range. The earlier
  draft relied on a "mod < 2^62 with QregMaxQubits headroom" argument
  that was both fragile (implicit dependence on n_qubits) and wrong
  near the upper end; `addMod` removes the assumption entirely.

* `ModPow(base, exp, mod uint64) uint64` --- square-and-multiply,
  internally using `MulMod` for every multiplication so the
  intermediate never overflows. Public callers see the same
  signature as `/c`'s `mod_pow`.
* `ContinuedFraction(x float64, maxDenom uint64) (num, den uint64)`
* `IsPowerOfTwo(x int) bool`
* `Ilog2(x uint32) int`

## 7. Testing

### 7.1 Framework and conventions

Standard `testing` package. One `_test.go` per source file. Test names
follow Go convention (`TestApplyHTwiceIsIdentity`). Tolerances
`AmpTol = 1e-10` and `ProbTol = 1e-9` defined in
`testdata_test.go` (or simply inline per-test).

Tests use `t.Helper()` and the helper `assertAmpNear(t, expected,
actual)` to mirror `/c`'s `ASSERT_NEAR_AMP` macro.

### 7.2 Coverage matrix

Same shape as the `/c` matrix; specific test count target ~75 cases
across the suite. Highlights:

| File | Cases |
|---|---|
| `qreg_test.go` | NewQreg accept (1..QregMaxQubits, but actual cases stop at 16 -- no point allocating 16 GiB in CI); NewQreg returns error on 0, QregMaxQubits+1, -1; WithSeed reproducibility; WithWorkers caps fan-out; InitBasis correct; Norm invariant; ProbOf basis; Amplitude in-range vs out-of-range (out-of-range panics, recovered in test); AmplitudesCopy is independent of subsequent gates |
| `gates_single_test.go` | H twice = I; S^2 = Z; T^4 = Z; R 2pi = I; Y on \|0\>; Z on \|0\> = I; Z on \|1\> negates (via H sandwich) |
| `gates_controlled_test.go` | Bell state from H+CNOT; CZ phase visible via H-sandwich; SWAP exchanges; CU general (Bell with custom 2x2) |
| `gates_multi_test.go` | MCZ flips only all-ones (verified via H^n MCZ H^n analytical probability check); MCX as Toffoli |
| `measure_test.go` | Deterministic measure on basis state; collapse for \|+\> on a qubit; SampleDistribution counts in {0,1}; Clone independent |
| `qft_test.go` | 1-qubit QFT = H; uniform from \|0\>; round-trip on six bases; period detection on periodic input |
| `grover_test.go` | 1 marked in N=16 \>= 0.95 after 3 iters; 4 marked total prob = 1 after 1 iter; over-iteration drops prob; zero-iter uniform; formula match; all-marked uniform |
| `shor_test.go` | ModularExp pass-through y>=N; ModularExp in-ring (orbit table for a=2 mod 5); period of a=7 mod 15; period of a=4 mod 15; factor 15 (3 reps); period of a=2 mod 21 gated by RUN_SHOR_21 |

### 7.3 Race detection

`make test-race` runs `go test -race ./...`. Required for CI: a race
in the dispatcher or in a gate closure (workers writing to overlapping
amp cells, a future option mutating shared state without
synchronisation) would silently corrupt results in production. The
race detector catches *intra-gate* contention; it does not certify
that callers can share a `*Qreg` across goroutines -- see the
"Concurrency contract" note in §4. Tests that intentionally exercise
the worker fan-out (e.g. 16-qubit ApplyH with `WithWorkers(8)`) are
the highest-signal cases for the race detector.

### 7.4 Determinism

Tests that depend on `rng` (measurement-driven tests) pin the seed at
construction via the functional option:

```go
q, err := qubit.NewQreg(n, qubit.WithSeed(42))
if err != nil { t.Fatal(err) }
```

The seed is fixed once at construction; the production API has no
post-construction mutator. Tests that want to vary the seed across
sub-tests construct a fresh Qreg per sub-test (cheap: it is just a
`make([]complex128, ...)`).

## 8. Build, demo, CI

### 8.1 Makefile (thin wrapper over `go`)

```
make             go build ./...
make test        go test ./...                          (parallel)
make test-race   go test -race ./...
make bench       go test -bench=. -benchmem ./qubit/... (manual only)
make demo ALGO=qft
                 cd cmd/qubit && go run . --algo=$(ALGO)
make fmt         gofmt -w .
make vet         go vet ./...
make clean       go clean ./...
```

`make test-large` is intentionally absent: there is no NP knob in Go.
Shor-21 is gated by `RUN_SHOR_21=1` env var inherited from the user's
shell (no MPI forwarding needed).

### 8.2 Demo binary (cmd/qubit/main.go)

`--algo {bell|qft|grover|shor}`. Each demo prints a couple of
probabilities or factors. Mirrors `/c`'s qubit.c. `main` wraps its
body in a `recover` so a panic raised on the **main goroutine** --
precondition violations checked before dispatch, a bad CLI flag value
caught by an assert, etc. -- exits the CLI cleanly with a non-zero
status:

```go
func main() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Fprintln(os.Stderr, r)
            os.Exit(1)
        }
    }()
    run()   // parse flags, dispatch on --algo, print results
}
```

Panics raised inside dispatcher-spawned goroutines (the gate closures
themselves) are *not* recoverable from `main` -- Go's default
goroutine-panic behaviour prints the panic value and stack trace and
terminates the whole process, also exiting non-zero. Either path
satisfies "loud failure, non-zero exit". `main`'s `recover` is just
for the synchronous-precondition-failure case where we get a tidy
single-line stderr instead of a goroutine stack trace.

This is the **only** site in the codebase that calls `os.Exit`; the
library itself never does (§4.4).

### 8.3 CI integration

Add to `.github/workflows/ci.yml`:

```yaml
go-tests:
  name: implementation/go -- make test + test-race
  runs-on: ubuntu-latest
  timeout-minutes: 10
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-go@v5
      with:
        go-version: '1.22'
    - run: make test
      working-directory: implementation/go
    - run: make test-race
      working-directory: implementation/go
    - run: make vet
      working-directory: implementation/go
```

Plus a separate `go-tests-shor-21` job that exports `RUN_SHOR_21=1`
before invoking `make test` --- mirrors the `/c` `c-tests-large`
pattern.

## 9. Algorithm scope

### 9.1 In v1

All entries in `/c`'s spec §9.1: register construction and state
management (no `Destroy` -- see §4.2), every single- and two-qubit
gate, multi-controlled-Z/X, measurement (full/qubit/sampling), QFT
+ inverse, Grover with oracle callback, Shor (ModularExp,
ShorPeriod, ShorFactor), the qubit.c-equivalent demo.

### 9.2 Explicitly out

* Distributed execution across machines (would re-introduce MPI's
  problem space; outside the goroutine-concurrency premise).
* Density matrices / mixed states.
* Noise models.
* GPU offload (the Python/PyTorch sibling covers that).
* CGO wrappers around `/c`'s code (defeats the point of a pure-Go
  alternative).

### 9.3 Performance targets

**25 qubits is the intended operating point.** The performance
numbers below are calibrated for that size. `QregMaxQubits = 26`
exists to give a single qubit of ad-hoc headroom (§4); it is
**not** a routinely benchmarked configuration, and timings at 26 are
allowed to be ~2x worse than at 25 without that counting as a
regression.

At the 25-qubit operating point:

* Shor-period round in <2 s on Apple Silicon / comparable.
* 20-qubit Grover with one marked item in <100 ms.
* QFT on 25 qubits with full bit-reversal in <500 ms.

Not pass/fail gates; sanity-check numbers for the README. The Go
implementation is allowed to be ~2x slower than `/c` at NP=1 since it
trades raw throughput for goroutine-friendly code organisation. The
26-qubit ceiling exists for headroom, not benchmarking; the test
suite never builds a register above 16 qubits.

## 10. Coverage tracking

`implementation/go/assessment.md` mirrors `/c`'s assessment matrix: each
thesis claim mapped to a Go file:line. The matrix is updated at the end
of each implementation phase.

## 11. Out of scope for this design

* The plan file itself (writing-plans skill produces it after spec
  approval).
* The Python/PyTorch sibling (separate spec / plan / implementation
  cycle).
* Co-changes to the LaTeX thesis: this implementation is a parallel
  realisation, not a thesis-API extension. No §12 ref_guide updates
  required (the thesis API is language-agnostic; Go just renames
  consistently).

## 12. Approval

Sections §1--§4 of this spec were each presented in brainstorming and
approved at v1. Subsequent rounds revised the written form:

* Round 1: surface the ModularExp-as-permutation shortcut and the
  shared-memory observation that drops `redistribute_pairs`.
* Round 2: fix workers-stranded-on-old-Amp (snapshot per dispatch),
  fix MulMod overflow near 2^32, drop the dead finalizer claim.
* Round 3: `addMod` helper closes the residual overflow window in
  MulMod, API unified on methods (no mixed method/package-fn pairs
  for state-mutating ops), worker goroutine made panic-safe.
* Round 4: replace `os.Exit` fail-fast with `panic` for programmer
  errors and `error` returns for construction errors; drop the
  persistent worker pool and `Destroy()` in favour of per-call
  goroutines + `sync.WaitGroup`; unexport `amp` and expose
  `Amplitude`/`AmplitudesCopy`/`NQubits` accessors; replace the
  `SeedRNG` test-mutator with functional options `WithSeed` /
  `WithWorkers` at `NewQreg`. Semantic parity with `/c` preserved;
  the changes are purely API/lifecycle/error-handling.
* Round 5: `QregMaxQubits` dropped from 60 to 30 so the documented
  ceiling matches what `make([]complex128, 1<<n)` can actually
  allocate on the laptop target; option-error policy contradiction
  resolved (options stay infallible in v1, the type remains
  `func(*Qreg)`, `NewQreg`'s `error` return is reserved for the
  `nQubits` check); explicit "Qreg is not safe for concurrent method
  calls" contract added to §4; §7.3 wording updated from "race in
  the worker pool" to "race in the dispatcher or gate closures";
  §8.2 CLI recover narrowed to "main-goroutine panics only --
  worker-goroutine panics crash via Go default and also exit
  non-zero".
* Round 6: `QregMaxQubits` further dropped 30 -> 26. Round 5's 30
  sized the ceiling to "biggest slice make() can allocate in
  isolation," but on a 16 GiB laptop cohabiting with the OS, the Go
  runtime, and an IDE/browser, `NewQreg(30)` would still OOM at
  runtime rather than return a clean error. 26 sizes the ceiling
  to what `NewQreg` can reliably succeed on the laptop target:
  1 GiB amp slice, 2 GiB ModularExp peak working set, leaving ~14
  GiB of physical RAM for the OS and the user's other tools. §9.3
  performance targets reworded to make "25 = intended operating
  point" (where the perf numbers apply) and "26 = ad-hoc headroom,
  not routinely benchmarked" explicit. Wording cleanup: "qreg
  lifecycle" (which implied /c-style create+destroy parity)
  replaced with "register construction and operations" / "register
  construction and state management" in §1 and §9.1; the §4.2 note
  that Go intentionally has no `Destroy` is now cross-referenced.
* Round 7 (current): MiB/GB unit consistency (§2 target-scale row
  uses MiB to match the §4 memory table; §1 "16 GB laptop" -> "16
  GiB laptop"). "~8x cohabitation headroom" -- which left it
  ambiguous what the multiplier was over -- replaced with the
  concrete "2 GiB peak vs 16 GiB total physical RAM, leaving ~14
  GiB for OS/runtime/tools" both at the `QregMaxQubits` constant
  comment and in the §4.4 mirror sentence.

Awaiting user re-approval before transitioning to the implementation
plan via the writing-plans skill.
