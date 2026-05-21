# implementation/go -- coverage of thesis claims

Updated end of each implementation phase. File:line references point at
the canonical implementation site for each claim.

## §8 (sparse-gate strategy, 2026 thesis)

| Claim | Status | Location |
|---|---|---|
| In-place single-qubit gate, O(2^n) | ✓ | `qubit/gates_single.go::ApplyU` |
| Shared-memory state vector | ✓ | `qubit/qreg.go::Qreg.amp` |
| Parallel pair-iteration | ✓ | `qubit/dispatch.go::parallelOverPairs` |
| Controlled gate uniform loop | ✓ | `qubit/gates_controlled.go::ApplyCU` |
| Modular_exp as in-place permutation | ✓ | `qubit/shor.go::ApplyModularExp` |
| qreg API per §12 | ✓ | `qubit/qreg.go` + accessors |

## §9 (2026 QFT)

| Claim | Status | Location |
|---|---|---|
| QFT forward + inverse | ✓ | `qubit/qft.go::ApplyQFT` / `ApplyQFTInverse` |
| Includes bit-reversal swaps | ✓ | `qubit/qft.go::ApplyQFT` final swap loop |
| Period detection on known periodic input | ✓ (tested) | `qubit/qft_test.go::TestQFTDetectsPeriod` |

## §10 (2026 Grover)

| Claim | Status | Location |
|---|---|---|
| Phase-oracle callback API | ✓ | `qubit/grover.go::OracleFn` |
| H^n -> oracle/diffusion loop | ✓ | `qubit/grover.go::ApplyGrover` |
| Optimum-stop tested | ✓ | `qubit/grover_test.go::TestGroverOverIterationDropsAccuracy` |
| Multiple marked items | ✓ | `qubit/grover_test.go::TestGrover4MarkedIn16` |

## §11 (2026 Shor)

| Claim | Status | Location |
|---|---|---|
| ApplyModularExp with y>=N pass-through | ✓ | `qubit/shor.go::ApplyModularExp` |
| ApplyShorPeriod (period finding) | ✓ | `qubit/shor.go::ApplyShorPeriod` |
| ShorFactor (end-to-end) | ✓ | `qubit/shor.go::ShorFactor` |
| Continued-fraction post-processing | ✓ | `qubit/standart.go::ContinuedFraction` |
| Factor N=15 reliably | ✓ (tested) | `qubit/shor_test.go::TestShorFactor15` |

## §12 (qreg API)

Every entry in spec §6 is implemented in `qubit/qreg.go` + adjacent gate
files. The Go-specific deviations vs. thesis §12:

* No `Destroy()` -- per-call goroutines + WaitGroup means no resources
  outlive a gate call; the GC reclaims the Qreg when the caller drops
  the last reference.
* Amplitude slice unexported; accessors are `Amplitude(i)`,
  `AmplitudesCopy()`, `ProbOf(basis)`, `Norm()`.
* Construction via `NewQreg(nQubits, WithSeed(...), WithWorkers(...))`
  with functional options instead of post-construction mutators.
* Programmer-error panics use Go panic with "qubit: ..." prefix;
  construction errors return `error`. Library never calls `os.Exit`.

## Out of scope for v1

* Distributed execution across machines.
* Density matrices / mixed states.
* Noise models.
* GPU offload (the Python/PyTorch sibling covers that).
* CGO wrappers around `/c`'s code.

## Test matrix

All test binaries pass via `make test` and `make test-race`. The
Shor-21 period test (16-qubit, a=2 mod 21, expected r divides 6) is
gated behind `RUN_SHOR_21=1` to keep the default `make test` loop fast.
The test calls `ApplyShorPeriod` directly with fixed base a=2 (no
random base-selection); the only stochasticity is the inherent
QFT-readout measurement.
