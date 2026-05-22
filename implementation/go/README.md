# implementation/go -- goroutine-parallel quantum simulator

A pure-Go state-vector quantum-circuit simulator. Covers the same
thesis claims as `implementation/c/` (every gate, measurement, QFT,
Grover, Shor) but parallelised with goroutines instead of MPI, with an
idiomatic-Go API: panic for programmer errors, error for construction,
functional options, no manual destroy.

## Quickstart

```bash
cd implementation/go
make            # go build ./...
make test       # go test ./...
make test-race  # go test -race ./...
make demo ALGO=bell  # run the Bell-state demo
```

## API at a glance

```go
import "github.com/c0ze/dissertation-thesis/implementation/go/qubit"

q, err := qubit.NewQreg(4, qubit.WithSeed(42), qubit.WithWorkers(4))
if err != nil { /* nQubits out of [1, QregMaxQubits] */ }
// NewQreg initialises to |0...0>, so you can apply gates immediately.
// Call q.InitBasis(b) to reset / re-initialise from a different basis.
q.ApplyH(0)
q.ApplyCNOT(0, 1)
fmt.Println("P(00) =", q.ProbOf(0))
```

Every state-mutating operation is a method on `*Qreg`. The amplitude
slice is intentionally unexported; use `Amplitude(i)`, `ProbOf(basis)`,
or `AmplitudesCopy()` to inspect. External Grover oracles should use
`q.FlipPhase(basis)` to mark amplitudes — that's the exported helper
that replaces direct `q.amp[mark] = -q.amp[mark]` (which only works
inside `package qubit`).

```go
oracle := func(q *qubit.Qreg, user any) {
    q.FlipPhase(user.(uint64))   // mark the basis state in `user`
}
q.ApplyGrover(n, oracle, uint64(mark), iterations)
```

## Design notes

* **Concurrency model.** Per-call goroutines inside the parallel
  dispatcher, joined with `sync.WaitGroup`. No persistent worker pool
  and no `Destroy()`. See spec §4.2 for the rationale.
* **Validation.** Out-of-range qubit indices and other programmer errors
  panic with a "qubit: ..." message. `NewQreg` returns `error` for bad
  `nQubits`. The library never calls `os.Exit`.
* **Concurrency safety.** A `*Qreg` is not safe for concurrent method
  calls; different `*Qreg`s are independent.
* **Ceiling.** `QregMaxQubits = 26` (1 GiB amp slice, 2 GiB ModularExp
  peak). Diverges from `/c`'s 60 (which is a shift-overflow bound, not
  an allocation bound).
* **`ShorFactor` ceiling.** With `QregMaxQubits = 26` and the
  `3 * ceil(log2 N) + 1` shortcut layout, the top-level odd-N path
  supports N up to roughly 8 bits (Shor-15, Shor-21 in particular).
  Larger odd N is rejected upfront with `{0, 0, 0}`. Even N is
  short-circuited classically regardless of size. Non-positive
  `maxAttempts` returns `{0, 0, 0}` without allocating a Qreg.

## Tests

```
make test           # all tests, default workers
make test-race      # all tests with the race detector
RUN_SHOR_21=1 make test
                    # also runs the 16-qubit Shor-21 period test
                    # (~6-20 ms on Apple Silicon)
```

The test suite never builds a register above 16 qubits, so `make test`
completes in well under a minute on commodity hardware.

## Files

| File | Responsibility |
|---|---|
| `qubit/qreg.go` | Qreg struct, NewQreg, accessors |
| `qubit/options.go` | WithSeed, WithWorkers functional options |
| `qubit/dispatch.go` | parallelOverPairs / parallelOverIndices |
| `qubit/assert.go` | panic helper for programmer errors |
| `qubit/standart.go` | GCD, addMod, MulMod, ModPow, ContinuedFraction |
| `qubit/gates_single.go` | ApplyU + H, X, Y, Z, S, T, Phase, Rx, Ry, Rz |
| `qubit/gates_controlled.go` | ApplyCU + CNOT, CZ, ControlledPhase, SWAP |
| `qubit/gates_multi.go` | ApplyMultiControlledZ, ApplyMultiControlledX |
| `qubit/measure.go` | MeasureAll, MeasureQubit, SampleDistribution, Clone, Dump |
| `qubit/qft.go` | ApplyQFT, ApplyQFTInverse |
| `qubit/grover.go` | OracleFn, ApplyGrover |
| `qubit/shor.go` | ApplyModularExp, ApplyShorPeriod, ShorFactor |
| `cmd/qubit/main.go` | --algo {bell|qft|grover|shor} demo |

The `standart.go` filename mirrors `/c`'s misspelling deliberately so
the cross-implementation parallel is visually obvious.

## Spec and plan

* Spec: `docs/superpowers/specs/2026-05-21-implementation-go-design.md`
* Plan: `docs/superpowers/plans/2026-05-22-implementation-go.md`
