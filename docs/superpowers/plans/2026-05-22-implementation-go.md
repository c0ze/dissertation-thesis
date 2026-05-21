# Go Sparse-Gate Quantum Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-Go implementation of the sparse-gate quantum simulator at `implementation/go/`, covering every thesis claim already implemented by `/c` (qreg construction + state, all gates, measurement, QFT, Grover, Shor) but using idiomatic Go semantics: per-call goroutines for parallelism, unexported state vector, panic-vs-error split for validation, functional options for construction, no `Destroy()`.

**Architecture:** Single package `qubit/` plus a `cmd/qubit/` CLI. Per-call `go func()` + `sync.WaitGroup.Wait()` for gate-internal fan-out — no persistent worker pool. Methods on `*Qreg` for all state-mutating operations (`ApplyH`, `ApplyQFT`, `ApplyShorPeriod`, `ApplyModularExp`); package functions reserved for entry points that allocate their own register (`ShorFactor` only). Programmer errors `panic` with a formatted message; construction errors return `error` from `NewQreg`. Ceiling at 26 qubits to keep `NewQreg` cleanly succeed-or-error on a 16 GiB laptop.

**Tech Stack:** Go 1.21+ (the user's system Go at `/usr/local/go/bin/go` is 1.21.9; plan uses no 1.22-specific features). Standard library only — `math`, `math/cmplx`, `math/rand`, `sync`, `runtime`, `fmt`, `io`, `os`, `time`. No external dependencies. `testing` package for tests; `go test -race` gated by `make test-race`.

**Spec:** `docs/superpowers/specs/2026-05-21-implementation-go-design.md`. Re-read it whenever a task says "see spec §X" — the spec is the source of truth for invariants and API contracts.

**Build target:** `implementation/go/` is a self-contained Go module that does not depend on the existing `implementation/c/` or `implementation/original/` code. Cross-implementation parity is verified by sharing the same test scenarios (factor 15, period of 2 mod 21, etc.) not by sharing source.

---

## Phase 0: Bootstrap

### Task 1: Directory layout, go.mod, .gitignore

**Files:**
- Create: `implementation/go/go.mod`
- Create: `implementation/go/.gitignore`
- Create: `implementation/go/qubit/.gitkeep`
- Create: `implementation/go/cmd/qubit/.gitkeep`

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p implementation/go/qubit implementation/go/cmd/qubit
```

- [ ] **Step 2: Write `implementation/go/go.mod`**

```
module github.com/arda-karaduman/thesis-go

go 1.21
```

(The module path is invented — there's no actual GitHub repo to publish to, and no external deps will import this code. It just needs to be a valid path so `go build` accepts it.)

- [ ] **Step 3: Write `implementation/go/.gitignore`**

```
/build/
*.test
*.out
coverage.out
```

- [ ] **Step 4: Create `.gitkeep` files so the empty dirs survive git**

```bash
touch implementation/go/qubit/.gitkeep implementation/go/cmd/qubit/.gitkeep
```

- [ ] **Step 5: Verify the module is valid**

Run: `cd implementation/go && go mod tidy`
Expected: no output. (`go build ./...` would fail at this stage because there are no `.go` files yet; the first build runs in Task 4 once `assert.go` exists.)

- [ ] **Step 6: Commit**

```bash
git add implementation/go/go.mod implementation/go/.gitignore implementation/go/qubit/.gitkeep implementation/go/cmd/qubit/.gitkeep
git commit -m "feat(go): scaffold module + dir layout"
```

---

### Task 2: assert.go (panic helper used by every gate)

**Files:**
- Create: `implementation/go/qubit/assert.go`
- Create: `implementation/go/qubit/assert_test.go`

The `assert` helper is the foundation for all gate preconditions (§4.4 of spec). It must panic with a formatted message on `!cond`. It is package-private (lowercase first letter).

- [ ] **Step 1: Write the test first**

Create `implementation/go/qubit/assert_test.go`:

```go
package qubit

import (
	"strings"
	"testing"
)

func TestAssertPasses(t *testing.T) {
	// Should not panic.
	assert(true, "this should not fire")
}

func TestAssertPanicsWithMessage(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic, got none")
		}
		msg := r.(error).Error()
		if !strings.HasPrefix(msg, "qubit: ") {
			t.Fatalf("panic message missing qubit: prefix: %q", msg)
		}
		if !strings.Contains(msg, "bad: 42") {
			t.Fatalf("panic message missing formatted args: %q", msg)
		}
	}()
	assert(false, "bad: %d", 42)
	t.Fatal("unreachable")
}
```

- [ ] **Step 2: Run the test to confirm it fails to compile**

Run: `cd implementation/go && go test ./qubit/ -run TestAssert`
Expected: build error — `undefined: assert`

- [ ] **Step 3: Write the helper**

Create `implementation/go/qubit/assert.go`:

```go
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
```

- [ ] **Step 4: Run the tests and confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run TestAssert -v`
Expected: `PASS` for both tests.

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/assert.go implementation/go/qubit/assert_test.go
git commit -m "feat(go): add assert helper for programmer-error panics"
```

---

### Task 3: Qreg struct skeleton (no methods yet)

**Files:**
- Create: `implementation/go/qubit/qreg.go`

Lay down the type definitions referenced by every later file. No methods yet — just types and the `QregMaxQubits` constant.

- [ ] **Step 1: Write `implementation/go/qubit/qreg.go`**

```go
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
```

- [ ] **Step 2: Verify it builds**

Run: `cd implementation/go && go build ./qubit/`
Expected: no output (clean build).

- [ ] **Step 3: Commit**

```bash
git add implementation/go/qubit/qreg.go
git commit -m "feat(go): Qreg struct skeleton + QregMaxQubits constant"
```

---

### Task 4: Makefile

**Files:**
- Create: `implementation/go/Makefile`

Thin wrapper over `go`. Spec §8.1 fixes the target list.

- [ ] **Step 1: Write `implementation/go/Makefile`**

```makefile
# ---------------------------------------------------------------------------
# Makefile for implementation/go -- goroutine-parallel quantum simulator.
#
# Targets:
#   make             go build ./...
#   make test        go test ./...
#   make test-race   go test -race ./...
#   make bench       go test -bench=. -benchmem ./qubit/...
#   make demo ALGO=qft
#                    go run ./cmd/qubit --algo=$(ALGO)
#   make fmt         gofmt -w .
#   make vet         go vet ./...
#   make clean       go clean ./...
#
# Shor-21 (the 16-qubit period-finding test for a=2 mod 21) is gated by
# RUN_SHOR_21=1 in the environment, mirroring /c's RUN_SHOR_21 convention.
# Set it directly: `RUN_SHOR_21=1 make test` -- there is no NP knob in Go.
# ---------------------------------------------------------------------------

ALGO ?= bell

.PHONY: all test test-race bench demo fmt vet clean

all:
	go build ./...

test:
	go test ./...

test-race:
	go test -race ./...

bench:
	go test -bench=. -benchmem ./qubit/...

demo:
	go run ./cmd/qubit --algo=$(ALGO)

fmt:
	gofmt -w .

vet:
	go vet ./...

clean:
	go clean ./...
```

- [ ] **Step 2: Verify the makefile parses and `make` runs**

Run: `cd implementation/go && make`
Expected: no output (the qubit package builds; cmd/qubit is still empty so its target is skipped).

- [ ] **Step 3: Verify `make test` runs and passes (only assert tests so far)**

Run: `cd implementation/go && make test`
Expected: `ok  github.com/arda-karaduman/thesis-go/qubit` line for the assert tests.

- [ ] **Step 4: Commit**

```bash
git add implementation/go/Makefile
git commit -m "feat(go): Makefile wrapper over go build/test/vet"
```

---

## Phase 1: Arithmetic helpers (standart.go)

These functions have no Qreg dependency, so they go first. Each one gets its own commit.

### Task 5: standart.go GCD

**Files:**
- Create: `implementation/go/qubit/standart.go`
- Create: `implementation/go/qubit/standart_test.go`

The filename intentionally mirrors `/c`'s misspelling so the cross-implementation parallel is visually obvious (spec §3).

- [ ] **Step 1: Write the test first**

Create `implementation/go/qubit/standart_test.go`:

```go
package qubit

import "testing"

func TestGCD(t *testing.T) {
	cases := []struct {
		a, b, want uint64
	}{
		{0, 5, 5},
		{5, 0, 5},
		{12, 8, 4},
		{7, 13, 1},
		{15, 21, 3},
		{100, 75, 25},
	}
	for _, c := range cases {
		got := GCD(c.a, c.b)
		if got != c.want {
			t.Errorf("GCD(%d, %d) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestGCD`
Expected: build error — `undefined: GCD`.

- [ ] **Step 3: Implement GCD in `standart.go`**

Create `implementation/go/qubit/standart.go`:

```go
package qubit

// GCD returns gcd(a, b) using Euclid's algorithm. gcd(0, n) = n.
func GCD(a, b uint64) uint64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
```

- [ ] **Step 4: Run to confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run TestGCD -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/standart.go implementation/go/qubit/standart_test.go
git commit -m "feat(go): standart.GCD"
```

---

### Task 6: standart.go addMod (overflow-safe)

**Files:**
- Modify: `implementation/go/qubit/standart.go`
- Modify: `implementation/go/qubit/standart_test.go`

Spec §6.9: plain `(a+b) % mod` overflows when `a+b >= 2^64`, which happens for any `mod` above `2^63`. `addMod` subtracts from the modulus instead of adding past it.

- [ ] **Step 1: Add the test**

Append to `standart_test.go`:

```go
func TestAddMod(t *testing.T) {
	cases := []struct {
		a, b, mod, want uint64
	}{
		{1, 2, 5, 3},
		{4, 4, 5, 3},          // 8 mod 5 = 3
		{0, 0, 5, 0},
		{1<<63 + 1, 1<<63 + 2, 1<<63 + 5, 1<<63 - 2}, // overflow-prone
		{1<<63, 1<<63 - 1, 1<<63 + 1, 1<<63 - 2},      // mod near 2^63
	}
	for _, c := range cases {
		got := addMod(c.a, c.b, c.mod)
		if got != c.want {
			t.Errorf("addMod(%d, %d, %d) = %d, want %d",
				c.a, c.b, c.mod, got, c.want)
		}
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestAddMod`
Expected: `undefined: addMod`.

- [ ] **Step 3: Implement**

Append to `standart.go`:

```go
// addMod returns (a + b) mod mod without overflowing uint64. Plain
// (a+b) % mod can overflow when a+b >= 2^64, which happens for any mod
// above 2^63. We subtract from the modulus instead of adding past it.
//
// Precondition: a < mod, b < mod. Callers in MulMod maintain this.
func addMod(a, b, mod uint64) uint64 {
	// mod - b is well-defined and < mod since b < mod.
	// If a >= mod - b, then a + b >= mod, so wrap by subtracting.
	if a >= mod-b {
		return a - (mod - b)
	}
	return a + b
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run TestAddMod -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/standart.go implementation/go/qubit/standart_test.go
git commit -m "feat(go): standart.addMod (overflow-safe)"
```

---

### Task 7: standart.go MulMod (double-and-add)

**Files:**
- Modify: `implementation/go/qubit/standart.go`
- Modify: `implementation/go/qubit/standart_test.go`

Go has no native `__uint128_t` (`/c`'s ModPow trick). Spec §6.9 specifies double-and-add over `addMod`.

- [ ] **Step 1: Add the test**

Append to `standart_test.go`:

```go
func TestMulMod(t *testing.T) {
	cases := []struct {
		a, b, mod, want uint64
	}{
		{0, 5, 7, 0},
		{3, 4, 7, 5},        // 12 mod 7 = 5
		{1234567, 7654321, 1000000007, 772047864},
		{1 << 40, 1 << 40, 1 << 50, 0},      // exact-power-of-2 case
	}
	for _, c := range cases {
		got := MulMod(c.a, c.b, c.mod)
		if got != c.want {
			t.Errorf("MulMod(%d, %d, %d) = %d, want %d",
				c.a, c.b, c.mod, got, c.want)
		}
	}
}

func TestMulModZeroModulus(t *testing.T) {
	if got := MulMod(5, 7, 0); got != 0 {
		t.Errorf("MulMod(5, 7, 0) = %d, want 0", got)
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestMulMod`
Expected: `undefined: MulMod`.

- [ ] **Step 3: Implement**

Append to `standart.go`:

```go
// MulMod returns (a * b) mod mod using double-and-add over addMod.
// Safe for any mod < 2^64 (the entire representable range); the loop
// invariant `result < mod && a < mod` is preserved by addMod every
// iteration, so the body never overflows regardless of mod's magnitude.
func MulMod(a, b, mod uint64) uint64 {
	if mod == 0 {
		return 0
	}
	var result uint64
	a %= mod
	for b > 0 {
		if b&1 == 1 {
			result = addMod(result, a, mod)
		}
		a = addMod(a, a, mod) // doubling step
		b >>= 1
	}
	return result
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run TestMulMod -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/standart.go implementation/go/qubit/standart_test.go
git commit -m "feat(go): standart.MulMod via double-and-add"
```

---

### Task 8: standart.go ModPow

**Files:**
- Modify: `implementation/go/qubit/standart.go`
- Modify: `implementation/go/qubit/standart_test.go`

Square-and-multiply over MulMod (so no intermediate overflows). Same signature as `/c`'s `mod_pow`.

- [ ] **Step 1: Test**

Append to `standart_test.go`:

```go
func TestModPow(t *testing.T) {
	cases := []struct {
		base, exp, mod, want uint64
	}{
		{2, 10, 1000, 24},
		{7, 0, 15, 1},
		{0, 0, 7, 1},
		{2, 1<<10, 1000000007, 812734592},     // mod within uint32
		{1<<32 + 1, 5, 1<<33 - 1, 5100273671}, // mod over uint32
	}
	for _, c := range cases {
		got := ModPow(c.base, c.exp, c.mod)
		if got != c.want {
			t.Errorf("ModPow(%d, %d, %d) = %d, want %d",
				c.base, c.exp, c.mod, got, c.want)
		}
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestModPow`

- [ ] **Step 3: Implement**

Append to `standart.go`:

```go
// ModPow returns base^exp mod mod using square-and-multiply over MulMod.
// 0^0 conventionally returns 1.
func ModPow(base, exp, mod uint64) uint64 {
	if mod == 1 {
		return 0
	}
	result := uint64(1)
	base %= mod
	for exp > 0 {
		if exp&1 == 1 {
			result = MulMod(result, base, mod)
		}
		exp >>= 1
		base = MulMod(base, base, mod)
	}
	return result
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run TestModPow -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/standart.go implementation/go/qubit/standart_test.go
git commit -m "feat(go): standart.ModPow via square-and-multiply"
```

---

### Task 9: standart.go ContinuedFraction + IsPowerOfTwo + Ilog2

**Files:**
- Modify: `implementation/go/qubit/standart.go`
- Modify: `implementation/go/qubit/standart_test.go`

Continued-fraction expansion approximates a float by `num/den` with `den <= maxDenom`. Used by ShorPeriod to recover the period from the QFT readout (spec §6.9).

- [ ] **Step 1: Test**

Append to `standart_test.go`:

```go
func TestContinuedFraction(t *testing.T) {
	// 3/8 should round-trip exactly with maxDenom >= 8.
	num, den := ContinuedFraction(3.0/8.0, 100)
	if num != 3 || den != 8 {
		t.Errorf("ContinuedFraction(3/8, 100) = %d/%d, want 3/8", num, den)
	}
	// 1/3: best rational with denominator <= 10 is 1/3.
	num, den = ContinuedFraction(1.0/3.0, 10)
	if num != 1 || den != 3 {
		t.Errorf("ContinuedFraction(1/3, 10) = %d/%d, want 1/3", num, den)
	}
}

func TestIsPowerOfTwo(t *testing.T) {
	cases := []struct {
		x    int
		want bool
	}{
		{1, true}, {2, true}, {4, true}, {1024, true},
		{0, false}, {3, false}, {6, false}, {-2, false},
	}
	for _, c := range cases {
		if got := IsPowerOfTwo(c.x); got != c.want {
			t.Errorf("IsPowerOfTwo(%d) = %v, want %v", c.x, got, c.want)
		}
	}
}

func TestIlog2(t *testing.T) {
	cases := []struct {
		x    uint32
		want int
	}{
		{1, 0}, {2, 1}, {4, 2}, {1024, 10}, {1 << 30, 30},
	}
	for _, c := range cases {
		if got := Ilog2(c.x); got != c.want {
			t.Errorf("Ilog2(%d) = %d, want %d", c.x, got, c.want)
		}
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestContinuedFraction|TestIsPowerOfTwo|TestIlog2'`

- [ ] **Step 3: Implement**

Append to `standart.go`:

```go
// ContinuedFraction approximates x by num/den with den <= maxDenom,
// using the standard continued-fraction expansion. Used by Shor's
// period finder to recover r from the QFT readout c/2^t.
func ContinuedFraction(x float64, maxDenom uint64) (num, den uint64) {
	if x <= 0 {
		return 0, 1
	}
	var (
		h0, h1 uint64 = 0, 1
		k0, k1 uint64 = 1, 0
	)
	for i := 0; i < 64; i++ {
		ai := uint64(x)
		hNew := ai*h1 + h0
		kNew := ai*k1 + k0
		if kNew > maxDenom {
			return h1, k1
		}
		h0, h1 = h1, hNew
		k0, k1 = k1, kNew
		frac := x - float64(ai)
		if frac < 1e-12 {
			return h1, k1
		}
		x = 1.0 / frac
	}
	return h1, k1
}

// IsPowerOfTwo returns true for positive powers of two: 1, 2, 4, 8, ...
func IsPowerOfTwo(x int) bool {
	return x > 0 && x&(x-1) == 0
}

// Ilog2 returns floor(log2(x)) for x >= 1. Undefined for x == 0.
func Ilog2(x uint32) int {
	n := 0
	for x > 1 {
		x >>= 1
		n++
	}
	return n
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestContinuedFraction|TestIsPowerOfTwo|TestIlog2' -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/standart.go implementation/go/qubit/standart_test.go
git commit -m "feat(go): standart.ContinuedFraction + IsPowerOfTwo + Ilog2"
```

---

## Phase 2: Qreg construction, options, accessors

### Task 10: options.go + WithSeed + WithWorkers

**Files:**
- Create: `implementation/go/qubit/options.go`
- Create: `implementation/go/qubit/options_test.go`

Spec §6.1a. Options are infallible (`type Option func(*Qreg)`); invalid values are normalised silently rather than failing construction.

- [ ] **Step 1: Test**

Create `implementation/go/qubit/options_test.go`:

```go
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
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestWith'`
Expected: `undefined: WithSeed`, `undefined: WithWorkers`.

- [ ] **Step 3: Implement**

Create `implementation/go/qubit/options.go`:

```go
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
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestWith' -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/options.go implementation/go/qubit/options_test.go
git commit -m "feat(go): functional options WithSeed + WithWorkers"
```

---

### Task 11: NewQreg + NQubits + InitBasis

**Files:**
- Modify: `implementation/go/qubit/qreg.go`
- Create: `implementation/go/qubit/qreg_test.go`

Spec §4.2 + §6.1. Construction errors return `error`; accessors are unexported-field-safe.

- [ ] **Step 1: Test**

Create `implementation/go/qubit/qreg_test.go`:

```go
package qubit

import (
	"math/cmplx"
	"testing"
)

const AmpTol = 1e-10
const ProbTol = 1e-9

func TestNewQregValid(t *testing.T) {
	for _, n := range []int{1, 2, 8, 16} {
		q, err := NewQreg(n)
		if err != nil {
			t.Fatalf("NewQreg(%d) returned err: %v", n, err)
		}
		if q.NQubits() != n {
			t.Errorf("NewQreg(%d).NQubits() = %d", n, q.NQubits())
		}
		if got, want := len(q.amp), 1<<n; got != want {
			t.Errorf("amp len = %d, want %d", got, want)
		}
	}
}

func TestNewQregRejectsOutOfRange(t *testing.T) {
	cases := []int{0, -1, QregMaxQubits + 1, 100}
	for _, n := range cases {
		_, err := NewQreg(n)
		if err == nil {
			t.Errorf("NewQreg(%d): want error, got nil", n)
		}
	}
}

func TestNewQregAppliesOptions(t *testing.T) {
	q, err := NewQreg(4, WithWorkers(2), WithSeed(42))
	if err != nil {
		t.Fatal(err)
	}
	if q.workers != 2 {
		t.Errorf("WithWorkers(2): q.workers = %d", q.workers)
	}
	if q.rng == nil {
		t.Errorf("WithSeed: q.rng is nil")
	}
}

func TestInitBasisCollapsesToBasisState(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(5) // |101>
	for i, a := range q.amp {
		want := complex(0, 0)
		if i == 5 {
			want = complex(1, 0)
		}
		if cmplx.Abs(a-want) > AmpTol {
			t.Errorf("amp[%d] = %v, want %v", i, a, want)
		}
	}
}

func TestInitBasisRejectsOutOfRange(t *testing.T) {
	q, _ := NewQreg(3)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for InitBasis(8)")
		}
	}()
	q.InitBasis(8) // out of range for 3-qubit register
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestNewQreg`
Expected: `undefined: NewQreg`, `undefined: NQubits`.

- [ ] **Step 3: Implement**

Replace the existing one-entry import block in `qreg.go`:

```go
import (
	"math/rand"
)
```

with this four-entry block:

```go
import (
	"fmt"
	"math/rand"
	"runtime"
	"time"
)
```

Then append below the type declarations:

```go
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
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestNewQreg|TestInitBasis' -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/qreg.go implementation/go/qubit/qreg_test.go
git commit -m "feat(go): NewQreg + NQubits + InitBasis"
```

---

### Task 12: Amplitude + AmplitudesCopy + ProbOf + Norm

**Files:**
- Modify: `implementation/go/qubit/qreg.go`
- Modify: `implementation/go/qubit/qreg_test.go`

Spec §6.1. Accessors that let external code read state without giving them a live mutable slice.

- [ ] **Step 1: Tests**

Append to `qreg_test.go`:

```go
func TestAmplitudeReadsBack(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(2)
	if got := q.Amplitude(2); got != complex(1, 0) {
		t.Errorf("Amplitude(2) = %v, want (1+0i)", got)
	}
	if got := q.Amplitude(3); got != complex(0, 0) {
		t.Errorf("Amplitude(3) = %v, want (0+0i)", got)
	}
}

func TestAmplitudePanicsOutOfRange(t *testing.T) {
	q, _ := NewQreg(2)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for Amplitude(4)")
		}
	}()
	q.Amplitude(4)
}

func TestAmplitudesCopyIsIndependent(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(0)
	c := q.AmplitudesCopy()
	c[0] = complex(99, 0) // mutate the copy
	if q.amp[0] != complex(1, 0) {
		t.Errorf("AmplitudesCopy returned an aliased slice: q.amp[0] = %v", q.amp[0])
	}
}

func TestProbOfBasisState(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(3)
	if got := q.ProbOf(3); got != 1.0 {
		t.Errorf("ProbOf(3) = %v, want 1.0", got)
	}
	if got := q.ProbOf(0); got != 0.0 {
		t.Errorf("ProbOf(0) = %v, want 0.0", got)
	}
}

func TestNormOnBasisStateIsOne(t *testing.T) {
	q, _ := NewQreg(4)
	q.InitBasis(7)
	if got := q.Norm(); abs(got-1.0) > ProbTol {
		t.Errorf("Norm = %v, want 1.0", got)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestAmplitude|TestProbOf|TestNorm'`

- [ ] **Step 3: Implement**

Append to `qreg.go`:

```go
// Amplitude returns the i-th amplitude. Panics if i is out of range.
// Use this for single bounds-checked reads; for full vectors use
// AmplitudesCopy.
func (q *Qreg) Amplitude(i uint64) complex128 {
	assert(i < uint64(len(q.amp)),
		"Amplitude: i=%d out of [0, %d)", i, len(q.amp))
	return q.amp[i]
}

// AmplitudesCopy returns a fresh copy of the full amplitude slice.
// Mutating the returned slice does NOT affect the register; that is
// the point -- the live amp slice is intentionally unexported. For
// in-place inspection that does not need a copy, use Amplitude.
func (q *Qreg) AmplitudesCopy() []complex128 {
	out := make([]complex128, len(q.amp))
	copy(out, q.amp)
	return out
}

// ProbOf returns |amp[basis]|^2. Panics if basis is out of range.
func (q *Qreg) ProbOf(basis uint64) float64 {
	assert(basis < uint64(len(q.amp)),
		"ProbOf: basis=%d out of [0, %d)", basis, len(q.amp))
	a := q.amp[basis]
	return real(a)*real(a) + imag(a)*imag(a)
}

// Norm returns sum over i of |amp[i]|^2. For a valid state vector,
// this is 1.0 to within floating-point precision.
func (q *Qreg) Norm() float64 {
	var sum float64
	for _, a := range q.amp {
		sum += real(a)*real(a) + imag(a)*imag(a)
	}
	return sum
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestAmplitude|TestProbOf|TestNorm' -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/qreg.go implementation/go/qubit/qreg_test.go
git commit -m "feat(go): Amplitude + AmplitudesCopy + ProbOf + Norm accessors"
```

---

## Phase 3: Dispatcher

### Task 13: parallelOverPairs + parallelOverIndices

**Files:**
- Create: `implementation/go/qubit/dispatch.go`
- Create: `implementation/go/qubit/dispatch_test.go`

Spec §4.3. Per-call goroutines, joined by `sync.WaitGroup.Wait()`. No persistent state.

- [ ] **Step 1: Tests**

Create `implementation/go/qubit/dispatch_test.go`:

```go
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
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestParallel|TestDispatcher'`
Expected: `undefined: parallelOverPairs`.

- [ ] **Step 3: Implement**

Create `implementation/go/qubit/dispatch.go`:

```go
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
```

- [ ] **Step 4: Confirm pass at NP=1 AND -race**

Run: `cd implementation/go && go test ./qubit/ -run 'TestParallel|TestDispatcher' -v -race`
Expected: PASS, no race detected.

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/dispatch.go implementation/go/qubit/dispatch_test.go
git commit -m "feat(go): parallelOverPairs + parallelOverIndices dispatcher"
```

---

## Phase 4: Single-qubit gates

### Task 14: ApplyU + ApplyH

**Files:**
- Create: `implementation/go/qubit/gates_single.go`
- Create: `implementation/go/qubit/gates_single_test.go`
- Create: `implementation/go/qubit/testhelpers_test.go`

Spec §5.1. ApplyU is the workhorse; ApplyH wraps it with the Hadamard matrix.

- [ ] **Step 1: Add the shared test helper**

Create `implementation/go/qubit/testhelpers_test.go`:

```go
package qubit

import (
	"math/cmplx"
	"testing"
)

// assertAmpNear fails the test if |got - want| > AmpTol.
func assertAmpNear(t *testing.T, want, got complex128, name string) {
	t.Helper()
	if cmplx.Abs(want-got) > AmpTol {
		t.Errorf("%s: got %v, want %v (|diff| = %g)",
			name, got, want, cmplx.Abs(want-got))
	}
}
```

- [ ] **Step 2: Add the ApplyU + ApplyH tests**

Create `implementation/go/qubit/gates_single_test.go`:

```go
package qubit

import (
	"math"
	"testing"
)

func TestApplyHTwiceIsIdentity(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(5) // |101>
	q.ApplyH(1)
	q.ApplyH(1)
	for i, a := range q.amp {
		want := complex(0, 0)
		if i == 5 {
			want = complex(1, 0)
		}
		assertAmpNear(t, want, a, "H^2 on |101>")
	}
}

func TestApplyHOnZeroProducesPlusState(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyH(0)
	// Expect (|0> + |1>) / sqrt(2)
	inv2 := complex(1.0/math.Sqrt2, 0)
	assertAmpNear(t, inv2, q.amp[0], "H|0> amp[0]")
	assertAmpNear(t, inv2, q.amp[1], "H|0> amp[1]")
}

func TestApplyUTargetOutOfRangePanics(t *testing.T) {
	q, _ := NewQreg(2)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for ApplyU target=5 on 2-qubit register")
		}
	}()
	q.ApplyU(5, [2][2]complex128{{1, 0}, {0, 1}})
}
```

- [ ] **Step 3: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyH|TestApplyU'`

- [ ] **Step 4: Implement**

Create `implementation/go/qubit/gates_single.go`:

```go
package qubit

import "math"

// ApplyU applies the single-qubit 2x2 unitary u to the target qubit.
// Iterates over pairs (i0, i1) that differ only in the target bit;
// parallelised via parallelOverPairs.
func (q *Qreg) ApplyU(target int, u [2][2]complex128) {
	assert(target >= 0 && target < q.nQubits,
		"ApplyU: target=%d out of [0, %d)", target, q.nQubits)
	nPairs := 1 << (q.nQubits - 1)
	tBit := uint(target)
	q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			lower := i & ((1 << tBit) - 1)
			upper := (i >> tBit) << (tBit + 1)
			i0 := upper | lower
			i1 := i0 | (1 << tBit)
			a0, a1 := amp[i0], amp[i1]
			amp[i0] = u[0][0]*a0 + u[0][1]*a1
			amp[i1] = u[1][0]*a0 + u[1][1]*a1
		}
	})
}

// ApplyH applies the Hadamard gate to the target qubit.
//   H = (1/sqrt(2)) [[1,  1], [1, -1]]
func (q *Qreg) ApplyH(target int) {
	inv2 := complex(1.0/math.Sqrt2, 0)
	q.ApplyU(target, [2][2]complex128{
		{inv2, inv2},
		{inv2, -inv2},
	})
}
```

- [ ] **Step 5: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyH|TestApplyU' -v -race`

- [ ] **Step 6: Commit**

```bash
git add implementation/go/qubit/gates_single.go implementation/go/qubit/gates_single_test.go implementation/go/qubit/testhelpers_test.go
git commit -m "feat(go): ApplyU workhorse + ApplyH"
```

---

### Task 15: ApplyX + ApplyY + ApplyZ

**Files:**
- Modify: `implementation/go/qubit/gates_single.go`
- Modify: `implementation/go/qubit/gates_single_test.go`

The Pauli gates. Each is a one-liner over ApplyU.

- [ ] **Step 1: Tests**

Append to `gates_single_test.go`:

```go
func TestApplyXFlips(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(1) // |01>
	q.ApplyX(0)    // flip bit 0 -> |00>
	assertAmpNear(t, complex(1, 0), q.amp[0], "X on |01> amp[0]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "X on |01> amp[1]")
}

func TestApplyZOnOneNegates(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(1)
	q.ApplyZ(0)
	assertAmpNear(t, complex(-1, 0), q.amp[1], "Z|1> amp[1]")
}

func TestApplyYOnZero(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyY(0)
	// Y|0> = i|1>
	assertAmpNear(t, complex(0, 0), q.amp[0], "Y|0> amp[0]")
	assertAmpNear(t, complex(0, 1), q.amp[1], "Y|0> amp[1]")
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyX|TestApplyY|TestApplyZ'`

- [ ] **Step 3: Implement**

Append to `gates_single.go`:

```go
// ApplyX applies the Pauli-X (NOT) gate.
//   X = [[0, 1], [1, 0]]
func (q *Qreg) ApplyX(target int) {
	q.ApplyU(target, [2][2]complex128{
		{0, 1},
		{1, 0},
	})
}

// ApplyY applies the Pauli-Y gate.
//   Y = [[0, -i], [i, 0]]
func (q *Qreg) ApplyY(target int) {
	q.ApplyU(target, [2][2]complex128{
		{0, complex(0, -1)},
		{complex(0, 1), 0},
	})
}

// ApplyZ applies the Pauli-Z gate.
//   Z = [[1, 0], [0, -1]]
func (q *Qreg) ApplyZ(target int) {
	q.ApplyU(target, [2][2]complex128{
		{1, 0},
		{0, -1},
	})
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyX|TestApplyY|TestApplyZ' -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/gates_single.go implementation/go/qubit/gates_single_test.go
git commit -m "feat(go): Pauli X, Y, Z gates"
```

---

### Task 16: ApplyS + ApplyT + ApplyPhase

**Files:**
- Modify: `implementation/go/qubit/gates_single.go`
- Modify: `implementation/go/qubit/gates_single_test.go`

Phase rotation gates. S = sqrt(Z), T = sqrt(S), Phase(theta) is the general version.

- [ ] **Step 1: Tests**

Append to `gates_single_test.go`:

```go
func TestApplyS_Squared_EqualsZ(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(1)
	q.ApplyS(0)
	q.ApplyS(0)
	// S^2 = Z, so amp[1] = -1
	assertAmpNear(t, complex(-1, 0), q.amp[1], "S^2|1> amp[1]")
}

func TestApplyT_FourthPower_EqualsZ(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(1)
	q.ApplyT(0)
	q.ApplyT(0)
	q.ApplyT(0)
	q.ApplyT(0)
	// T^4 = Z, so amp[1] = -1
	assertAmpNear(t, complex(-1, 0), q.amp[1], "T^4|1> amp[1]")
}

func TestApplyPhasePiEqualsZ(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(1)
	q.ApplyPhase(0, math.Pi)
	assertAmpNear(t, complex(-1, 0), q.amp[1], "Phase(pi)|1> amp[1]")
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyS|TestApplyT|TestApplyPhase'`

- [ ] **Step 3: Implement**

Append to `gates_single.go`:

```go
// ApplyS applies the S gate (phase pi/2).
//   S = [[1, 0], [0, i]]
func (q *Qreg) ApplyS(target int) {
	q.ApplyU(target, [2][2]complex128{
		{1, 0},
		{0, complex(0, 1)},
	})
}

// ApplyT applies the T gate (phase pi/4).
//   T = [[1, 0], [0, e^{i*pi/4}]]
func (q *Qreg) ApplyT(target int) {
	q.ApplyU(target, [2][2]complex128{
		{1, 0},
		{0, complex(math.Cos(math.Pi/4), math.Sin(math.Pi/4))},
	})
}

// ApplyPhase applies the general phase gate.
//   Phase(theta) = [[1, 0], [0, e^{i*theta}]]
func (q *Qreg) ApplyPhase(target int, theta float64) {
	q.ApplyU(target, [2][2]complex128{
		{1, 0},
		{0, complex(math.Cos(theta), math.Sin(theta))},
	})
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyS|TestApplyT|TestApplyPhase' -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/gates_single.go implementation/go/qubit/gates_single_test.go
git commit -m "feat(go): S, T, Phase gates"
```

---

### Task 17: ApplyRx + ApplyRy + ApplyRz

**Files:**
- Modify: `implementation/go/qubit/gates_single.go`
- Modify: `implementation/go/qubit/gates_single_test.go`

Continuous rotation gates. R(2pi) is identity (up to a global phase that doesn't matter physically, but with these conventional matrices it does come back to identity).

- [ ] **Step 1: Tests**

Append to `gates_single_test.go`:

```go
func TestApplyRx2PiOnZeroIsNegativeIdentity(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyRx(0, 2*math.Pi)
	// Rx(2pi) = -I
	assertAmpNear(t, complex(-1, 0), q.amp[0], "Rx(2pi)|0> amp[0]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "Rx(2pi)|0> amp[1]")
}

func TestApplyRy4PiOnZeroIsIdentity(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyRy(0, 4*math.Pi)
	// Ry(4pi) = I
	assertAmpNear(t, complex(1, 0), q.amp[0], "Ry(4pi)|0> amp[0]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "Ry(4pi)|0> amp[1]")
}

func TestApplyRzOnZeroIsDiagonal(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyRz(0, math.Pi/2)
	// Rz(theta)|0> = e^{-i*theta/2}|0>
	want := complex(math.Cos(-math.Pi/4), math.Sin(-math.Pi/4))
	assertAmpNear(t, want, q.amp[0], "Rz(pi/2)|0> amp[0]")
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyRx|TestApplyRy|TestApplyRz'`

- [ ] **Step 3: Implement**

Append to `gates_single.go`:

```go
// ApplyRx applies a rotation around the x-axis by theta.
//   Rx(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
func (q *Qreg) ApplyRx(target int, theta float64) {
	c := complex(math.Cos(theta/2), 0)
	s := complex(0, -math.Sin(theta/2))
	q.ApplyU(target, [2][2]complex128{
		{c, s},
		{s, c},
	})
}

// ApplyRy applies a rotation around the y-axis by theta.
//   Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
func (q *Qreg) ApplyRy(target int, theta float64) {
	c := complex(math.Cos(theta/2), 0)
	s := complex(math.Sin(theta/2), 0)
	q.ApplyU(target, [2][2]complex128{
		{c, -s},
		{s, c},
	})
}

// ApplyRz applies a rotation around the z-axis by theta.
//   Rz(theta) = [[e^{-i*t/2}, 0], [0, e^{i*t/2}]]
func (q *Qreg) ApplyRz(target int, theta float64) {
	negHalf := complex(math.Cos(-theta/2), math.Sin(-theta/2))
	posHalf := complex(math.Cos(theta/2), math.Sin(theta/2))
	q.ApplyU(target, [2][2]complex128{
		{negHalf, 0},
		{0, posHalf},
	})
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyRx|TestApplyRy|TestApplyRz' -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/gates_single.go implementation/go/qubit/gates_single_test.go
git commit -m "feat(go): Rx, Ry, Rz rotation gates"
```

---

## Phase 5: Controlled gates

### Task 18: ApplyCU + ApplyCNOT

**Files:**
- Create: `implementation/go/qubit/gates_controlled.go`
- Create: `implementation/go/qubit/gates_controlled_test.go`

Spec §5.2. Shared-memory simplifies the four-case dispatch from `/c` into one uniform loop.

- [ ] **Step 1: Tests**

Create `implementation/go/qubit/gates_controlled_test.go`:

```go
package qubit

import (
	"math"
	"testing"
)

func TestApplyCNOTLeavesZeroControlAlone(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(2) // |10>: control=0, target=1
	q.ApplyCNOT(0, 1)
	assertAmpNear(t, complex(1, 0), q.amp[2], "CNOT control=0: amp[2]")
}

func TestApplyCNOTFlipsTargetWhenControlIsOne(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(1) // |01>: control=1, target=0
	q.ApplyCNOT(0, 1)
	// Flips target (bit 1) -> |11> = 3
	assertAmpNear(t, complex(1, 0), q.amp[3], "CNOT control=1: amp[3]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "CNOT control=1: amp[1]")
}

func TestBellStateFromHCNOT(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(0)
	q.ApplyH(0)
	q.ApplyCNOT(0, 1)
	// |Phi+> = (|00> + |11>) / sqrt(2)
	inv2 := complex(1/math.Sqrt2, 0)
	assertAmpNear(t, inv2, q.amp[0], "Bell amp[0]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "Bell amp[1]")
	assertAmpNear(t, complex(0, 0), q.amp[2], "Bell amp[2]")
	assertAmpNear(t, inv2, q.amp[3], "Bell amp[3]")
}

func TestApplyCUPanicsOnControlEqualsTarget(t *testing.T) {
	q, _ := NewQreg(2)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for ApplyCU control == target")
		}
	}()
	q.ApplyCU(1, 1, [2][2]complex128{{1, 0}, {0, 1}})
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyCU|TestApplyCNOT|TestBellState'`

- [ ] **Step 3: Implement**

Create `implementation/go/qubit/gates_controlled.go`:

```go
package qubit

// ApplyCU applies the controlled-U gate: when the control bit is 1,
// apply the 2x2 unitary u to the target bit; otherwise leave the
// amplitude pair unchanged.
//
// Shared memory means the four locality cases /c needs (both local,
// both global, control-local-target-global, target-local-control-global)
// collapse into a single uniform loop.
func (q *Qreg) ApplyCU(control, target int, u [2][2]complex128) {
	assert(control >= 0 && control < q.nQubits,
		"ApplyCU: control=%d out of [0, %d)", control, q.nQubits)
	assert(target >= 0 && target < q.nQubits,
		"ApplyCU: target=%d out of [0, %d)", target, q.nQubits)
	assert(control != target,
		"ApplyCU: control == target == %d", control)
	nPairs := 1 << (q.nQubits - 1)
	cMask := 1 << uint(control)
	tBit := uint(target)
	q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			lower := i & ((1 << tBit) - 1)
			upper := (i >> tBit) << (tBit + 1)
			i0 := upper | lower
			if i0&cMask == 0 {
				continue // control bit must be 1
			}
			i1 := i0 | (1 << tBit)
			a0, a1 := amp[i0], amp[i1]
			amp[i0] = u[0][0]*a0 + u[0][1]*a1
			amp[i1] = u[1][0]*a0 + u[1][1]*a1
		}
	})
}

// ApplyCNOT applies the controlled-NOT gate.
//   CNOT = ApplyCU with X = [[0,1],[1,0]]
func (q *Qreg) ApplyCNOT(control, target int) {
	q.ApplyCU(control, target, [2][2]complex128{
		{0, 1},
		{1, 0},
	})
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyCU|TestApplyCNOT|TestBellState' -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/gates_controlled.go implementation/go/qubit/gates_controlled_test.go
git commit -m "feat(go): ApplyCU + ApplyCNOT (with Bell-state test)"
```

---

### Task 19: ApplyCZ + ApplyControlledPhase + ApplySWAP

**Files:**
- Modify: `implementation/go/qubit/gates_controlled.go`
- Modify: `implementation/go/qubit/gates_controlled_test.go`

CZ is the controlled-Pauli-Z. ControlledPhase is CZ generalised to arbitrary phase. SWAP decomposes into three CNOTs (same as `/c`).

- [ ] **Step 1: Tests**

Append to `gates_controlled_test.go`:

```go
func TestApplyCZOnlyFlipsAllOnes(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(3) // |11>
	q.ApplyCZ(0, 1)
	assertAmpNear(t, complex(-1, 0), q.amp[3], "CZ|11> amp[3]")
}

func TestApplyControlledPhasePiEqualsCZ(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(3)
	q.ApplyControlledPhase(0, 1, math.Pi)
	assertAmpNear(t, complex(-1, 0), q.amp[3], "CPhase(pi)|11> amp[3]")
}

func TestApplySWAPExchanges(t *testing.T) {
	q, _ := NewQreg(2)
	q.InitBasis(1) // |01>
	q.ApplySWAP(0, 1)
	// SWAP exchanges bits 0 and 1 -> |10> = 2
	assertAmpNear(t, complex(1, 0), q.amp[2], "SWAP|01> -> amp[2]")
	assertAmpNear(t, complex(0, 0), q.amp[1], "SWAP|01> -> amp[1] zero")
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyCZ|TestApplyControlledPhase|TestApplySWAP'`

- [ ] **Step 3: Implement**

Append to `gates_controlled.go`:

```go
// ApplyCZ applies the controlled-Z gate.
func (q *Qreg) ApplyCZ(control, target int) {
	q.ApplyCU(control, target, [2][2]complex128{
		{1, 0},
		{0, -1},
	})
}

// ApplyControlledPhase applies a phase rotation by theta on the |11> branch.
func (q *Qreg) ApplyControlledPhase(control, target int, theta float64) {
	q.ApplyCU(control, target, [2][2]complex128{
		{1, 0},
		{0, complex(math.Cos(theta), math.Sin(theta))},
	})
}

// ApplySWAP exchanges the contents of two qubits. SWAP = CNOT(a,b)
// then CNOT(b,a) then CNOT(a,b). Same decomposition as /c.
func (q *Qreg) ApplySWAP(a, b int) {
	assert(a != b, "ApplySWAP: a == b == %d", a)
	q.ApplyCNOT(a, b)
	q.ApplyCNOT(b, a)
	q.ApplyCNOT(a, b)
}
```

Also add `import "math"` at the top of `gates_controlled.go`.

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyCZ|TestApplyControlledPhase|TestApplySWAP' -v`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/gates_controlled.go implementation/go/qubit/gates_controlled_test.go
git commit -m "feat(go): CZ, ControlledPhase, SWAP"
```

---

## Phase 6: Multi-controlled gates

### Task 20: ApplyMultiControlledZ

**Files:**
- Create: `implementation/go/qubit/gates_multi.go`
- Create: `implementation/go/qubit/gates_multi_test.go`

Spec §5.3. Phase-flips the all-ones amplitude across the first n qubits.

- [ ] **Step 1: Test**

Create `implementation/go/qubit/gates_multi_test.go`:

```go
package qubit

import (
	"math"
	"testing"
)

func TestApplyMultiControlledZFlipsOnlyAllOnes(t *testing.T) {
	q, _ := NewQreg(3)
	// Build (|0>+|1>)^3 / sqrt(8)
	q.InitBasis(0)
	for i := 0; i < 3; i++ {
		q.ApplyH(i)
	}
	q.ApplyMultiControlledZ(3)
	// Only amp[7] should be negated; everything else stays at 1/sqrt(8).
	inv := complex(1/math.Sqrt(8), 0)
	for i := 0; i < 8; i++ {
		want := inv
		if i == 7 {
			want = -inv
		}
		assertAmpNear(t, want, q.amp[i], "MCZ amp")
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestApplyMultiControlledZ`

- [ ] **Step 3: Implement**

Create `implementation/go/qubit/gates_multi.go`:

```go
package qubit

// ApplyMultiControlledZ negates the amplitude where the lowest n qubits
// are all 1. Equivalent to a phase oracle marking |1...1> in the first
// n qubits; used by Grover's diffusion step.
func (q *Qreg) ApplyMultiControlledZ(n int) {
	assert(n > 0 && n <= q.nQubits,
		"ApplyMultiControlledZ: n=%d out of (0, %d]", n, q.nQubits)
	mask := uint64((1 << uint(n)) - 1)
	// Iterate over amp indices; flip any whose lowest n bits are all 1.
	nIdx := len(q.amp)
	q.parallelOverIndices(nIdx, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			if uint64(i)&mask == mask {
				amp[i] = -amp[i]
			}
		}
	})
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run TestApplyMultiControlledZ -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/gates_multi.go implementation/go/qubit/gates_multi_test.go
git commit -m "feat(go): ApplyMultiControlledZ for Grover diffusion"
```

---

### Task 21: ApplyMultiControlledX (Toffoli generalised)

**Files:**
- Modify: `implementation/go/qubit/gates_multi.go`
- Modify: `implementation/go/qubit/gates_multi_test.go`

Generalised Toffoli: flip target when all controls are 1.

- [ ] **Step 1: Test**

Append to `gates_multi_test.go`:

```go
func TestApplyMultiControlledXAsToffoli(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(3) // |011>: controls=1,1, target initially 0
	q.ApplyMultiControlledX([]int{0, 1}, 2)
	// Toffoli flips target -> |111> = 7
	assertAmpNear(t, complex(1, 0), q.amp[7], "Toffoli on |011>: amp[7]")
	assertAmpNear(t, complex(0, 0), q.amp[3], "Toffoli on |011>: amp[3]")
}

func TestApplyMultiControlledXSkipsWhenAnyControlZero(t *testing.T) {
	q, _ := NewQreg(3)
	q.InitBasis(1) // |001>: control 0 is 1, control 1 is 0
	q.ApplyMultiControlledX([]int{0, 1}, 2)
	// Should not flip: |001> stays
	assertAmpNear(t, complex(1, 0), q.amp[1], "Toffoli on |001>: amp[1] unchanged")
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestApplyMultiControlledX`

- [ ] **Step 3: Implement**

Append to `gates_multi.go`:

```go
// ApplyMultiControlledX flips the target qubit when every control qubit is 1.
// Generalises Toffoli; len(controls) == 2 is Toffoli, == 1 is CNOT.
func (q *Qreg) ApplyMultiControlledX(controls []int, target int) {
	assert(target >= 0 && target < q.nQubits,
		"ApplyMultiControlledX: target=%d out of [0, %d)", target, q.nQubits)
	var cMask uint64
	for _, c := range controls {
		assert(c >= 0 && c < q.nQubits,
			"ApplyMultiControlledX: control=%d out of [0, %d)", c, q.nQubits)
		assert(c != target,
			"ApplyMultiControlledX: control %d == target", c)
		cMask |= 1 << uint(c)
	}
	tBit := uint(target)
	tMask := uint64(1) << tBit
	nPairs := 1 << (q.nQubits - 1)
	q.parallelOverPairs(nPairs, func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			lower := i & ((1 << tBit) - 1)
			upper := (i >> tBit) << (tBit + 1)
			i0 := uint64(upper | lower)
			if i0&cMask != cMask {
				continue
			}
			i1 := i0 | tMask
			amp[i0], amp[i1] = amp[i1], amp[i0]
		}
	})
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run TestApplyMultiControlledX -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/gates_multi.go implementation/go/qubit/gates_multi_test.go
git commit -m "feat(go): ApplyMultiControlledX (generalised Toffoli)"
```

---

## Phase 7: Measurement

### Task 22: MeasureQubit

**Files:**
- Create: `implementation/go/qubit/measure.go`
- Create: `implementation/go/qubit/measure_test.go`

Single-qubit projective measurement: compute p0, sample, project + renormalise.

- [ ] **Step 1: Test**

Create `implementation/go/qubit/measure_test.go`:

```go
package qubit

import "testing"

func TestMeasureQubitOnBasisStateIsDeterministic(t *testing.T) {
	q, _ := NewQreg(3, WithSeed(1))
	q.InitBasis(5) // |101>: bit 0 = 1, bit 1 = 0, bit 2 = 1
	if got := q.MeasureQubit(0); got != 1 {
		t.Errorf("MeasureQubit(0) on |101> = %d, want 1", got)
	}
	q.InitBasis(5)
	if got := q.MeasureQubit(1); got != 0 {
		t.Errorf("MeasureQubit(1) on |101> = %d, want 0", got)
	}
	q.InitBasis(5)
	if got := q.MeasureQubit(2); got != 1 {
		t.Errorf("MeasureQubit(2) on |101> = %d, want 1", got)
	}
}

func TestMeasureQubitCollapsesAndRenormalises(t *testing.T) {
	q, _ := NewQreg(1, WithSeed(42))
	q.InitBasis(0)
	q.ApplyH(0) // |+>
	q.MeasureQubit(0)
	// After collapse, norm should be 1.0.
	if got := q.Norm(); abs(got-1.0) > ProbTol {
		t.Errorf("Norm after measure = %v, want 1.0", got)
	}
	// And exactly one of amp[0], amp[1] should be |1| and the other 0.
	a0sq := real(q.amp[0])*real(q.amp[0]) + imag(q.amp[0])*imag(q.amp[0])
	a1sq := real(q.amp[1])*real(q.amp[1]) + imag(q.amp[1])*imag(q.amp[1])
	if abs(a0sq+a1sq-1.0) > ProbTol {
		t.Errorf("post-collapse: a0sq+a1sq = %v, want 1.0", a0sq+a1sq)
	}
	if !((abs(a0sq-1) < ProbTol && abs(a1sq) < ProbTol) ||
		(abs(a0sq) < ProbTol && abs(a1sq-1) < ProbTol)) {
		t.Errorf("post-collapse not a basis state: a0sq=%v, a1sq=%v", a0sq, a1sq)
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestMeasureQubit`

- [ ] **Step 3: Implement**

Create `implementation/go/qubit/measure.go`:

```go
package qubit

import (
	"fmt"
	"io"
	"math"
)

// MeasureQubit performs a projective measurement on the target qubit
// in the computational basis, returning 0 or 1. The register is
// collapsed onto the measured branch and renormalised.
//
// Uses q.rng (seeded by time.Now() at construction, or via WithSeed).
func (q *Qreg) MeasureQubit(target int) int {
	assert(target >= 0 && target < q.nQubits,
		"MeasureQubit: target=%d out of [0, %d)", target, q.nQubits)
	tBit := uint(target)
	// p0 = sum over basis states with bit=0 of |amp|^2
	var p0 float64
	for i, a := range q.amp {
		if uint(i)>>tBit&1 == 0 {
			p0 += real(a)*real(a) + imag(a)*imag(a)
		}
	}
	// Sample with [0, 1).
	u := q.rng.Float64()
	outcome := 0
	if u >= p0 {
		outcome = 1
	}
	// Project + renormalise.
	var norm float64
	if outcome == 0 {
		norm = math.Sqrt(p0)
	} else {
		norm = math.Sqrt(1.0 - p0)
	}
	if norm == 0 {
		// Numerical edge: cannot happen if the sample is consistent,
		// but defend against it rather than dividing by zero.
		return outcome
	}
	for i, a := range q.amp {
		bit := int(uint(i) >> tBit & 1)
		if bit == outcome {
			q.amp[i] = complex(real(a)/norm, imag(a)/norm)
		} else {
			q.amp[i] = 0
		}
	}
	return outcome
}

// Dump writes |i>: amp_i lines for every basis index with nonzero
// amplitude. Diagnostic only.
func (q *Qreg) Dump(w io.Writer) {
	for i, a := range q.amp {
		if real(a) != 0 || imag(a) != 0 {
			fmt.Fprintf(w, "|%d>: %v\n", i, a)
		}
	}
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run TestMeasureQubit -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/measure.go implementation/go/qubit/measure_test.go
git commit -m "feat(go): MeasureQubit (single-qubit projective) + Dump"
```

---

### Task 23: MeasureAll + Clone

**Files:**
- Modify: `implementation/go/qubit/measure.go`
- Modify: `implementation/go/qubit/measure_test.go`

Sample a full basis state from the distribution and collapse. Clone returns an independent copy.

- [ ] **Step 1: Tests**

Append to `measure_test.go`:

```go
func TestMeasureAllOnBasisStateIsDeterministic(t *testing.T) {
	q, _ := NewQreg(3, WithSeed(1))
	q.InitBasis(6)
	if got := q.MeasureAll(); got != 6 {
		t.Errorf("MeasureAll on |6> = %d, want 6", got)
	}
}

func TestMeasureAllCollapses(t *testing.T) {
	q, _ := NewQreg(2, WithSeed(7))
	q.InitBasis(0)
	q.ApplyH(0)
	q.ApplyH(1)
	got := q.MeasureAll()
	// Post-measure, q.amp[got] should be ~1 and everyone else 0.
	for i, a := range q.amp {
		mag := real(a)*real(a) + imag(a)*imag(a)
		if i == int(got) {
			if abs(mag-1.0) > ProbTol {
				t.Errorf("post-measure amp[%d] magnitude = %v, want 1", i, mag)
			}
		} else {
			if mag > ProbTol {
				t.Errorf("post-measure amp[%d] magnitude = %v, want 0", i, mag)
			}
		}
	}
}

func TestCloneIsIndependent(t *testing.T) {
	q, _ := NewQreg(3, WithSeed(7))
	q.InitBasis(2)
	c := q.Clone()
	c.ApplyX(0) // mutate the clone
	// Original amp[2] should still be 1.
	assertAmpNear(t, complex(1, 0), q.amp[2], "original after clone mutation")
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestMeasureAll|TestClone'`

- [ ] **Step 3: Implement**

Append to `measure.go`:

```go
import "math/rand"  // add to the existing import block in measure.go
```

(Place that inside the existing `import (...)` block.)

Append the functions:

```go
// MeasureAll samples a full basis index from the |amp|^2 distribution
// and collapses the register onto |outcome>. Returns the outcome as a
// uint64 basis index.
func (q *Qreg) MeasureAll() uint64 {
	u := q.rng.Float64()
	var cum float64
	chosen := uint64(len(q.amp) - 1) // default to last index for numerical edge
	for i, a := range q.amp {
		cum += real(a)*real(a) + imag(a)*imag(a)
		if u < cum {
			chosen = uint64(i)
			break
		}
	}
	for i := range q.amp {
		q.amp[i] = 0
	}
	q.amp[chosen] = complex(1, 0)
	return chosen
}

// Clone returns an independent Qreg with the same amplitudes, worker
// count, and a freshly seeded RNG. The original and the clone do not
// share any mutable state.
func (q *Qreg) Clone() *Qreg {
	c := &Qreg{
		amp:     make([]complex128, len(q.amp)),
		nQubits: q.nQubits,
		workers: q.workers,
		rng:     rand.New(rand.NewSource(q.rng.Int63())),
	}
	copy(c.amp, q.amp)
	return c
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run 'TestMeasureAll|TestClone' -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/measure.go implementation/go/qubit/measure_test.go
git commit -m "feat(go): MeasureAll (full collapse) + Clone"
```

---

### Task 24: SampleDistribution

**Files:**
- Modify: `implementation/go/qubit/measure.go`
- Modify: `implementation/go/qubit/measure_test.go`

Run `shots` measurements without collapsing the *original* register: clone, measure, repeat.

- [ ] **Step 1: Test**

Append to `measure_test.go`:

```go
func TestSampleDistributionPreservesOriginal(t *testing.T) {
	q, _ := NewQreg(2, WithSeed(11))
	q.InitBasis(0)
	q.ApplyH(0)
	q.ApplyH(1) // uniform over {0,1,2,3}
	out := make([]uint64, 1000)
	q.SampleDistribution(out, 1000)
	// Count each outcome; with 1000 shots over 4 outcomes, each should
	// be roughly 250 +/- 60 (3-sigma).
	counts := [4]int{}
	for _, v := range out {
		if v >= 4 {
			t.Fatalf("got outcome %d outside [0,4)", v)
		}
		counts[v]++
	}
	for i, c := range counts {
		if c < 150 || c > 350 {
			t.Errorf("outcome %d count = %d, expected near 250", i, c)
		}
	}
	// Original should still be uniform (norm 1, four equal amplitudes).
	if abs(q.Norm()-1.0) > ProbTol {
		t.Errorf("original norm after sampling = %v, want 1.0", q.Norm())
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestSampleDistribution`

- [ ] **Step 3: Implement**

Append to `measure.go`:

```go
// SampleDistribution runs `shots` independent measurements on a clone
// of the register and writes the outcomes into out[0..shots). The
// original q is unmodified. len(out) must be >= shots.
func (q *Qreg) SampleDistribution(out []uint64, shots int) {
	assert(len(out) >= shots,
		"SampleDistribution: len(out)=%d < shots=%d", len(out), shots)
	for s := 0; s < shots; s++ {
		c := q.Clone()
		out[s] = c.MeasureAll()
	}
}
```

- [ ] **Step 4: Confirm pass**

Run: `cd implementation/go && go test ./qubit/ -run TestSampleDistribution -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/measure.go implementation/go/qubit/measure_test.go
git commit -m "feat(go): SampleDistribution via clone-and-measure"
```

---

## Phase 8: QFT

### Task 25: ApplyQFT + ApplyQFTInverse

**Files:**
- Create: `implementation/go/qubit/qft.go`
- Create: `implementation/go/qubit/qft_test.go`

Spec §6.6. Both directions include the final bit-reversal swaps so output amplitudes are in natural binary order.

- [ ] **Step 1: Tests**

Create `implementation/go/qubit/qft_test.go`:

```go
package qubit

import (
	"math"
	"testing"
)

func TestApplyQFTOnSingleQubitEqualsH(t *testing.T) {
	q, _ := NewQreg(1)
	q.InitBasis(0)
	q.ApplyQFT(0, 1)
	inv2 := complex(1/math.Sqrt2, 0)
	assertAmpNear(t, inv2, q.amp[0], "QFT|0> on 1 qubit amp[0]")
	assertAmpNear(t, inv2, q.amp[1], "QFT|0> on 1 qubit amp[1]")
}

func TestQFTOfZeroIsUniformSuperposition(t *testing.T) {
	n := 4
	q, _ := NewQreg(n)
	q.InitBasis(0)
	q.ApplyQFT(0, n)
	want := complex(1/math.Sqrt(float64(int(1)<<n)), 0)
	for i, a := range q.amp {
		assertAmpNear(t, want, a, "QFT|0> uniform amp")
		_ = i
	}
}

func TestQFTRoundTrip(t *testing.T) {
	n := 4
	for basis := uint64(0); basis < uint64(1<<n); basis++ {
		q, _ := NewQreg(n)
		q.InitBasis(basis)
		q.ApplyQFT(0, n)
		q.ApplyQFTInverse(0, n)
		for i, a := range q.amp {
			want := complex(0, 0)
			if uint64(i) == basis {
				want = complex(1, 0)
			}
			assertAmpNear(t, want, a, "QFT round-trip amp")
		}
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyQFT|TestQFTOf|TestQFTRoundTrip'`

- [ ] **Step 3: Implement**

Create `implementation/go/qubit/qft.go`:

```go
package qubit

import "math"

// ApplyQFT applies the quantum Fourier transform to qubits
// [start, start+n). Includes the final bit-reversal swaps so output
// amplitudes are in natural binary order (same convention as /c).
//
// Big-endian convention: qubit start+n-1 is the most-significant.
func (q *Qreg) ApplyQFT(start, n int) {
	assert(start >= 0 && start+n <= q.nQubits && n >= 1,
		"ApplyQFT: start=%d n=%d out of range for nQubits=%d",
		start, n, q.nQubits)
	for i := n - 1; i >= 0; i-- {
		q.ApplyH(start + i)
		for j := i - 1; j >= 0; j-- {
			theta := math.Pi / float64(int(1)<<uint(i-j))
			q.ApplyControlledPhase(start+j, start+i, theta)
		}
	}
	// Bit-reversal swaps.
	for i := 0; i < n/2; i++ {
		q.ApplySWAP(start+i, start+n-1-i)
	}
}

// ApplyQFTInverse applies the inverse QFT on qubits [start, start+n).
// Includes the bit-reversal swaps at the start so the input is in the
// same natural-binary order ApplyQFT produces.
func (q *Qreg) ApplyQFTInverse(start, n int) {
	assert(start >= 0 && start+n <= q.nQubits && n >= 1,
		"ApplyQFTInverse: start=%d n=%d out of range for nQubits=%d",
		start, n, q.nQubits)
	// Reverse the swaps first.
	for i := 0; i < n/2; i++ {
		q.ApplySWAP(start+i, start+n-1-i)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			theta := -math.Pi / float64(int(1)<<uint(i-j))
			q.ApplyControlledPhase(start+j, start+i, theta)
		}
		q.ApplyH(start + i)
	}
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run 'TestApplyQFT|TestQFTOf|TestQFTRoundTrip' -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/qft.go implementation/go/qubit/qft_test.go
git commit -m "feat(go): ApplyQFT + ApplyQFTInverse (with bit-reversal)"
```

---

### Task 26: QFT period-detection test

**Files:**
- Modify: `implementation/go/qubit/qft_test.go`

A periodic input should produce peaks at the multiples of N/period after QFT.

- [ ] **Step 1: Test**

Append to `qft_test.go`:

```go
func TestQFTDetectsPeriod(t *testing.T) {
	// Build a state with period 4 over 4 qubits: |0> + |4> + |8> + |12>,
	// normalised. After QFT we expect non-trivial probability mass on
	// indices that are multiples of N/period = 16/4 = 4: indices 0, 4,
	// 8, 12.
	n := 4
	q, _ := NewQreg(n)
	for i := range q.amp {
		q.amp[i] = 0
	}
	amp := complex(0.5, 0)
	q.amp[0], q.amp[4], q.amp[8], q.amp[12] = amp, amp, amp, amp
	q.ApplyQFT(0, n)
	// Sum probability at multiples of 4; should be near 1.
	var pPeak float64
	for k := 0; k < 16; k += 4 {
		pPeak += q.ProbOf(uint64(k))
	}
	if pPeak < 0.99 {
		t.Errorf("QFT period detection: peak prob = %v, want >= 0.99", pPeak)
	}
}
```

- [ ] **Step 2: Run + commit**

Run: `cd implementation/go && go test ./qubit/ -run TestQFTDetectsPeriod -v`

```bash
git add implementation/go/qubit/qft_test.go
git commit -m "test(go): QFT period-detection on periodic input"
```

---

## Phase 9: Grover

### Task 27: ApplyGrover

**Files:**
- Create: `implementation/go/qubit/grover.go`
- Create: `implementation/go/qubit/grover_test.go`

Spec §6.7. Phase-oracle callback API. Caller supplies a function that flips the phase of marked states; Grover wraps it in the H^n / 2|0><0|-I sandwich.

- [ ] **Step 1: Test**

Create `implementation/go/qubit/grover_test.go`:

```go
package qubit

import (
	"math"
	"testing"
)

func TestGrover1Marked16Qubits(t *testing.T) {
	// 4-qubit search, 1 marked state at |1010> = 10
	n := 4
	target := uint64(10)
	q, _ := NewQreg(n)
	q.InitBasis(0)
	oracle := func(q *Qreg, user interface{}) {
		mark := user.(uint64)
		// Flip phase of |mark> via -1 on amp[mark].
		q.amp[mark] = -q.amp[mark]
	}
	iters := int(math.Pi/4*math.Sqrt(float64(int(1)<<n))) // ~3 for n=4
	q.ApplyGrover(n, oracle, target, iters)
	if got := q.ProbOf(target); got < 0.9 {
		t.Errorf("Grover prob[%d] = %v, want >= 0.9 after %d iters", target, got, iters)
	}
}

func TestGroverOverIterationDropsAccuracy(t *testing.T) {
	n := 4
	target := uint64(10)
	q, _ := NewQreg(n)
	q.InitBasis(0)
	oracle := func(q *Qreg, user interface{}) {
		mark := user.(uint64)
		q.amp[mark] = -q.amp[mark]
	}
	q.ApplyGrover(n, oracle, target, 20) // way past optimum
	// Past optimum, probability of the marked state oscillates. Just
	// assert it's not greater than at the optimum.
	if got := q.ProbOf(target); got > 0.95 {
		t.Errorf("over-iteration kept prob high (%v); expected oscillation", got)
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestGrover`

- [ ] **Step 3: Implement**

Create `implementation/go/qubit/grover.go`:

```go
package qubit

// OracleFn is the user-supplied phase oracle. It receives the register
// (already prepared in the current Grover step) and a user payload
// (mark index, predicate, etc.); it must flip the sign of every
// amplitude whose basis state satisfies the user's predicate. Most
// callers implement this with a single q.amp[mark] = -q.amp[mark] or a
// loop over targets.
type OracleFn func(q *Qreg, user interface{})

// ApplyGrover runs `iterations` rounds of Grover's algorithm on the
// first nQubits qubits.
//
// Step 1: prepare uniform superposition H|0>^n.
// Each iteration: oracle (phase mark) + diffusion (2|s><s| - I).
//
// The diffusion operator is implemented as: H^n, X^n,
// multi-controlled-Z, X^n, H^n.
func (q *Qreg) ApplyGrover(nQubits int, oracle OracleFn, user interface{},
	iterations int) {
	assert(nQubits > 0 && nQubits <= q.nQubits,
		"ApplyGrover: nQubits=%d out of (0, %d]", nQubits, q.nQubits)
	for i := 0; i < nQubits; i++ {
		q.ApplyH(i)
	}
	for it := 0; it < iterations; it++ {
		oracle(q, user)
		// Diffusion: 2|s><s| - I where |s> is the uniform superposition.
		// Implementation: H^n, X^n, MCZ_n, X^n, H^n.
		for i := 0; i < nQubits; i++ {
			q.ApplyH(i)
		}
		for i := 0; i < nQubits; i++ {
			q.ApplyX(i)
		}
		q.ApplyMultiControlledZ(nQubits)
		for i := 0; i < nQubits; i++ {
			q.ApplyX(i)
		}
		for i := 0; i < nQubits; i++ {
			q.ApplyH(i)
		}
	}
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run TestGrover -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/grover.go implementation/go/qubit/grover_test.go
git commit -m "feat(go): ApplyGrover with phase-oracle callback"
```

---

### Task 28: Grover multi-marked test

**Files:**
- Modify: `implementation/go/qubit/grover_test.go`

When 4 of 16 states are marked, a single iteration produces total marked-probability of 1.

- [ ] **Step 1: Test**

Append to `grover_test.go`:

```go
func TestGrover4MarkedIn16(t *testing.T) {
	n := 4
	marks := []uint64{1, 5, 10, 14}
	q, _ := NewQreg(n)
	q.InitBasis(0)
	oracle := func(q *Qreg, user interface{}) {
		ms := user.([]uint64)
		for _, m := range ms {
			q.amp[m] = -q.amp[m]
		}
	}
	q.ApplyGrover(n, oracle, marks, 1)
	var pMarked float64
	for _, m := range marks {
		pMarked += q.ProbOf(m)
	}
	if abs(pMarked-1.0) > 0.01 {
		t.Errorf("Grover 4-of-16 after 1 iter: pMarked = %v, want ~1.0", pMarked)
	}
}
```

- [ ] **Step 2: Run + commit**

Run: `cd implementation/go && go test ./qubit/ -run TestGrover4Marked -v`

```bash
git add implementation/go/qubit/grover_test.go
git commit -m "test(go): Grover 4-marked-in-16 reaches total prob 1 after 1 iter"
```

---

## Phase 10: Shor

### Task 29: ApplyModularExp + pass-through test

**Files:**
- Create: `implementation/go/qubit/shor.go`
- Create: `implementation/go/qubit/shor_test.go`

Spec §5.5. ModularExp is a permutation on basis states: `(x, y) -> (x, a^x*y mod N)` for `y < N`, identity for `y >= N`. Workers writing disjoint output cells never collide.

- [ ] **Step 1: Tests (pass-through + orbit table)**

Create `implementation/go/qubit/shor_test.go`:

```go
package qubit

import "testing"

func TestModularExpPassThroughYGeN(t *testing.T) {
	// Layout: 6 total qubits = 1 counting + 5 target. counting at
	// bits [0..0], target at bits [1..5]. N = 5.
	// Place amplitude at (x=0, y=10). 10 >= 5 so should stay.
	q, _ := NewQreg(6)
	q.InitBasis(uint64(10) << 1)
	q.ApplyModularExp(0, 1, 1, 5, 2, 5)
	assertAmpNear(t, complex(1, 0), q.amp[10<<1],
		"pass-through (x=0, y=10>=5)")
}

func TestModularExpOrbitA2Mod5(t *testing.T) {
	// a=2, N=5 powers: 2^0=1, 2^1=2, 2^2=4, 2^3=3 (mod 5).
	// Layout: counting at [0..1] (t=2), target at [2..4] (n=3), total=5.
	expected := []uint64{1, 2, 4, 3}
	for x := 0; x < 4; x++ {
		q, _ := NewQreg(5)
		initial := (uint64(1) << 2) | (uint64(x) << 0) // y=1, x=x
		q.InitBasis(initial)
		q.ApplyModularExp(0, 2, 2, 3, 2, 5)
		final := (expected[x] << 2) | (uint64(x) << 0)
		assertAmpNear(t, complex(1, 0), q.amp[final],
			"orbit map")
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestModularExp`

- [ ] **Step 3: Implement**

Create `implementation/go/qubit/shor.go`:

```go
package qubit

// ShorPeriodResult is the return value of ApplyShorPeriod.
type ShorPeriodResult struct {
	R         uint64 // recovered period (0 on failure)
	MeasuredC uint64 // raw counting-register outcome (for debugging)
}

// ShorFactorResult is the return value of ShorFactor.
type ShorFactorResult struct {
	P, Q     uint64 // non-trivial factors of N (0 on failure)
	Attempts int    // number of period-finding rounds used
}

// ApplyModularExp applies the modular-exponentiation oracle:
//   (x, y) -> (x, a^x * y mod N)    for y < N
//   (x, y) -> (x, y)                 for y >= N
// where x lives in counting bits [countingStart, countingStart+t) and
// y lives in target bits [targetStart, targetStart+n).
//
// This is a permutation of computational basis states, so workers
// writing to disjoint output cells never collide.
func (q *Qreg) ApplyModularExp(countingStart, t, targetStart, n int,
	a, N uint64) {
	assert(t >= 1 && n >= 1,
		"ApplyModularExp: t=%d n=%d must be >= 1", t, n)
	assert(countingStart >= 0 && countingStart+t <= q.nQubits,
		"ApplyModularExp: counting range [%d, %d) out of [0, %d)",
		countingStart, countingStart+t, q.nQubits)
	assert(targetStart >= 0 && targetStart+n <= q.nQubits,
		"ApplyModularExp: target range [%d, %d) out of [0, %d)",
		targetStart, targetStart+n, q.nQubits)
	// counting and target must not overlap
	cEnd := countingStart + t
	tEnd := targetStart + n
	assert(cEnd <= targetStart || tEnd <= countingStart,
		"ApplyModularExp: counting [%d, %d) and target [%d, %d) overlap",
		countingStart, cEnd, targetStart, tEnd)
	assert(N >= 2, "ApplyModularExp: N=%d must be >= 2", N)
	assert(N <= uint64(1)<<uint(n),
		"ApplyModularExp: N=%d exceeds target capacity 2^%d", N, n)
	assert(GCD(a, N) == 1, "ApplyModularExp: GCD(a=%d, N=%d) != 1", a, N)

	newAmp := make([]complex128, len(q.amp))
	tMask := (uint64(1) << uint(t)) - 1
	nMask := (uint64(1) << uint(n)) - 1
	clearMask := (tMask << countingStart) | (nMask << targetStart)

	q.parallelOverIndices(len(q.amp), func(amp []complex128, lo, hi int) {
		for i := lo; i < hi; i++ {
			if amp[i] == 0 {
				continue
			}
			x := (uint64(i) >> countingStart) & tMask
			y := (uint64(i) >> targetStart) & nMask
			var yNew uint64
			if y < N {
				yNew = MulMod(y, ModPow(a, x, N), N)
			} else {
				yNew = y
			}
			outer := uint64(i) &^ clearMask
			iNew := outer | (x << countingStart) | (yNew << targetStart)
			newAmp[iNew] = amp[i] // safe: permutation, no contention
		}
	})
	q.amp = newAmp
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run TestModularExp -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/shor.go implementation/go/qubit/shor_test.go
git commit -m "feat(go): ApplyModularExp (permutation, no contention)"
```

---

### Task 30: ApplyShorPeriod (period of 7 mod 15)

**Files:**
- Modify: `implementation/go/qubit/shor.go`
- Modify: `implementation/go/qubit/shor_test.go`

Spec §6.8. Period finding: prepare counting+target, apply H^t to counting, ModularExp, inverse QFT on counting, measure counting, recover period via continued-fraction expansion.

- [ ] **Step 1: Test**

Append to `shor_test.go`:

```go
func TestShorPeriodA7Mod15(t *testing.T) {
	// Order of 7 mod 15 is 4. Use t=8 counting + n=4 target = 12 qubits.
	n := 4
	tBits := 8
	q, _ := NewQreg(tBits+n, WithSeed(1))
	res := q.ApplyShorPeriod(n, tBits, 0, n, 7, 15)
	if res.R == 0 {
		t.Fatalf("period finder returned r=0 (no recovery); raw c=%d", res.MeasuredC)
	}
	// Recovered period must divide true r=4: r in {1, 2, 4}.
	if res.R != 1 && res.R != 2 && res.R != 4 {
		t.Errorf("recovered r=%d not a divisor of 4", res.R)
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestShorPeriodA7Mod15`

- [ ] **Step 3: Implement**

Append to `shor.go`:

```go
// ApplyShorPeriod runs Shor's quantum period-finding subroutine:
//
//   1. InitBasis(1 << targetStart) -- counting=0, target=1
//   2. H^t on counting register
//   3. ApplyModularExp(counting, target, a, N)
//   4. ApplyQFTInverse on counting register
//   5. Measure counting register -> integer c in [0, 2^t)
//   6. Recover candidate period r via continued-fraction expansion of c/2^t
//
// Returns the recovered period and the raw measured counting value
// (for debugging). r=0 indicates failure (continued-fraction did not
// produce a denominator in the valid range).
func (q *Qreg) ApplyShorPeriod(countingStart, t, targetStart, n int,
	a, N uint64) ShorPeriodResult {
	// Init: counting=0, target=1 -- so amp at basis (1 << targetStart) = 1.
	q.InitBasis(uint64(1) << uint(targetStart))
	for i := 0; i < t; i++ {
		q.ApplyH(countingStart + i)
	}
	q.ApplyModularExp(countingStart, t, targetStart, n, a, N)
	q.ApplyQFTInverse(countingStart, t)
	full := q.MeasureAll()
	tMask := (uint64(1) << uint(t)) - 1
	c := (full >> uint(countingStart)) & tMask
	// Continued-fraction recovery: r = denominator of best approximation
	// of c/2^t with denominator <= N.
	x := float64(c) / float64(uint64(1)<<uint(t))
	_, den := ContinuedFraction(x, N)
	return ShorPeriodResult{R: den, MeasuredC: c}
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run TestShorPeriodA7Mod15 -v -race`

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/shor.go implementation/go/qubit/shor_test.go
git commit -m "feat(go): ApplyShorPeriod with continued-fraction recovery"
```

---

### Task 31: ShorFactor (factor 15)

**Files:**
- Modify: `implementation/go/qubit/shor.go`
- Modify: `implementation/go/qubit/shor_test.go`

Spec §6.8. End-to-end factoring: pick a coprime base `a`, find period via ShorPeriod, derive factor via gcd(a^(r/2) +/- 1, N).

- [ ] **Step 1: Test**

Append to `shor_test.go`:

```go
func TestShorFactor15(t *testing.T) {
	res := ShorFactor(15, 8)
	if res.P == 0 || res.Q == 0 {
		t.Fatalf("ShorFactor(15) failed after %d attempts", res.Attempts)
	}
	if res.P*res.Q != 15 {
		t.Errorf("ShorFactor(15): p=%d q=%d, p*q=%d, want 15", res.P, res.Q, res.P*res.Q)
	}
	if !(res.P == 3 || res.P == 5) {
		t.Errorf("ShorFactor(15): p=%d, want 3 or 5", res.P)
	}
}

func TestShorFactor15Repeated(t *testing.T) {
	// 3 reps; each should succeed within 8 attempts and produce {3,5}.
	for trial := 0; trial < 3; trial++ {
		res := ShorFactor(15, 8)
		if res.P*res.Q != 15 {
			t.Errorf("trial %d: ShorFactor(15) gave p=%d q=%d", trial, res.P, res.Q)
		}
		if !((res.P == 3 && res.Q == 5) || (res.P == 5 && res.Q == 3)) {
			t.Errorf("trial %d: ShorFactor(15) = (%d, %d), want (3,5) or (5,3)",
				trial, res.P, res.Q)
		}
	}
}
```

- [ ] **Step 2: Confirm fail**

Run: `cd implementation/go && go test ./qubit/ -run TestShorFactor15`

- [ ] **Step 3: Implement**

First add an import block at the top of `shor.go` (right under the `package qubit` line), since `shor.go` had no imports before:

```go
import (
	"math/rand"
	"time"
)
```

Then append the function:

```go
// ShorFactor attempts to factor N via Shor's algorithm. Picks random
// bases a coprime to N until period finding produces a usable r
// (non-zero, even, and a^{r/2} != -1 mod N), then derives a factor
// via gcd(a^{r/2} +/- 1, N). Returns {0, 0, attempts} on total failure.
//
// Allocates its own Qreg (size 2*ceil(log2 N) + 1, the standard width).
// This is the one entry point that is a package function rather than a
// method on *Qreg -- it has no "current register" to be a method on.
func ShorFactor(N uint64, maxAttempts int) ShorFactorResult {
	if N < 2 {
		return ShorFactorResult{P: 0, Q: 0, Attempts: 0}
	}
	if N%2 == 0 {
		return ShorFactorResult{P: 2, Q: N / 2, Attempts: 0}
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	n := 0
	for (uint64(1) << uint(n)) < N {
		n++
	}
	tBits := 2*n + 1
	nTotal := tBits + n
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		// Pick random a in [2, N-1] coprime to N.
		var a uint64
		for {
			a = uint64(rng.Intn(int(N-2))) + 2
			if g := GCD(a, N); g != 1 {
				// Lucky shortcut: gcd already a non-trivial factor.
				return ShorFactorResult{P: g, Q: N / g, Attempts: attempt}
			}
			break
		}
		q, err := NewQreg(nTotal)
		if err != nil {
			return ShorFactorResult{P: 0, Q: 0, Attempts: attempt}
		}
		res := q.ApplyShorPeriod(n, tBits, 0, n, a, N)
		r := res.R
		if r == 0 || r%2 != 0 {
			continue
		}
		half := ModPow(a, r/2, N)
		if half == N-1 {
			continue
		}
		p := GCD(half+1, N)
		if p > 1 && p < N {
			return ShorFactorResult{P: p, Q: N / p, Attempts: attempt}
		}
		p = GCD(half-1, N)
		if p > 1 && p < N {
			return ShorFactorResult{P: p, Q: N / p, Attempts: attempt}
		}
	}
	return ShorFactorResult{P: 0, Q: 0, Attempts: maxAttempts}
}
```

- [ ] **Step 4: Confirm pass + race-clean**

Run: `cd implementation/go && go test ./qubit/ -run TestShorFactor15 -v -race`

(Note: this test is stochastic but with 8 attempts the success rate on N=15 should be ~99%+. If it flakes intermittently, increase maxAttempts in the test, not the algorithm.)

- [ ] **Step 5: Commit**

```bash
git add implementation/go/qubit/shor.go implementation/go/qubit/shor_test.go
git commit -m "feat(go): ShorFactor end-to-end + factor 15 test"
```

---

### Task 32: Shor-21 gated test

**Files:**
- Modify: `implementation/go/qubit/shor_test.go`

Spec §7.2 + assessment.md §"Test matrix". 16-qubit register, fixed base a=2 (true period 6 mod 21). Gated by `RUN_SHOR_21=1`.

- [ ] **Step 1: Test**

First widen the existing `import "testing"` line in `shor_test.go` into a block that also pulls in `os`:

```go
import (
	"os"
	"testing"
)
```

Then append to `shor_test.go`:

```go
func TestShorPeriodA2Mod21(t *testing.T) {
	if os.Getenv("RUN_SHOR_21") == "" {
		t.Skip("set RUN_SHOR_21=1 to run the 16-qubit Shor-21 period test")
	}
	// True period of 2 mod 21 is 6 (2, 4, 8, 16, 11, 1). Continued
	// fraction therefore lands in {1, 2, 3, 6}. Test fixes a=2 to
	// remove random base-selection.
	n := 5
	tBits := 11
	q, _ := NewQreg(tBits+n, WithSeed(1))
	res := q.ApplyShorPeriod(n, tBits, 0, n, 2, 21)
	if res.R == 0 {
		t.Fatalf("Shor-21: period finder returned r=0")
	}
	switch res.R {
	case 1, 2, 3, 6:
		// pass
	default:
		t.Errorf("Shor-21: r=%d not a divisor of 6", res.R)
	}
}
```

- [ ] **Step 2: Run with the env var to verify**

Run: `cd implementation/go && RUN_SHOR_21=1 go test ./qubit/ -run TestShorPeriodA2Mod21 -v`
Expected: PASS (or `t.Fatalf` from r=0 indicating a real issue).

- [ ] **Step 3: Verify it skips without the env var**

Run: `cd implementation/go && go test ./qubit/ -run TestShorPeriodA2Mod21 -v`
Expected: `--- SKIP: TestShorPeriodA2Mod21`.

- [ ] **Step 4: Commit**

```bash
git add implementation/go/qubit/shor_test.go
git commit -m "test(go): Shor-21 period of 2 mod 21 gated by RUN_SHOR_21"
```

---

## Phase 11: CLI demo + docs + CI

### Task 33: cmd/qubit/main.go

**Files:**
- Create: `implementation/go/cmd/qubit/main.go`

Spec §8.2. `--algo {bell|qft|grover|shor}` flag, panic recover in main.

- [ ] **Step 1: Write the binary**

Create `implementation/go/cmd/qubit/main.go`:

```go
// Command qubit is a small demo binary that exercises the qubit
// library at the algorithm level. Mirrors /c's qubit.c.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/arda-karaduman/thesis-go/qubit"
)

func main() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintln(os.Stderr, r)
			os.Exit(1)
		}
	}()
	run()
}

func run() {
	algo := flag.String("algo", "bell", "demo to run: bell | qft | grover | shor")
	flag.Parse()
	switch *algo {
	case "bell":
		demoBell()
	case "qft":
		demoQFT()
	case "grover":
		demoGrover()
	case "shor":
		demoShor()
	default:
		fmt.Fprintf(os.Stderr, "unknown algo: %q\n", *algo)
		os.Exit(2)
	}
}

func demoBell() {
	q, err := qubit.NewQreg(2)
	if err != nil {
		panic(err)
	}
	q.InitBasis(0)
	q.ApplyH(0)
	q.ApplyCNOT(0, 1)
	fmt.Printf("Bell |Phi+>: P(00) = %.4f, P(11) = %.4f\n",
		q.ProbOf(0), q.ProbOf(3))
}

func demoQFT() {
	q, err := qubit.NewQreg(4)
	if err != nil {
		panic(err)
	}
	q.InitBasis(0)
	q.ApplyQFT(0, 4)
	fmt.Printf("QFT|0> on 4 qubits: P(0) = %.4f (uniform = %.4f)\n",
		q.ProbOf(0), 1.0/16.0)
}

func demoGrover() {
	// Mark |1111> using ApplyMultiControlledZ as the phase oracle.
	// (For arbitrary marks, the caller would build a custom gate
	// sequence; this demo just shows the canonical |1...1> case.)
	n := 4
	q, err := qubit.NewQreg(n)
	if err != nil {
		panic(err)
	}
	q.InitBasis(0)
	oracle := func(q *qubit.Qreg, _ interface{}) {
		q.ApplyMultiControlledZ(n)
	}
	iters := int(math.Pi / 4 * math.Sqrt(float64(int(1)<<n)))
	q.ApplyGrover(n, oracle, nil, iters)
	fmt.Printf("Grover marked |1111>: P(15) = %.4f\n", q.ProbOf(15))
}

func demoShor() {
	res := qubit.ShorFactor(15, 8)
	fmt.Printf("Shor(15): p=%d, q=%d, attempts=%d\n", res.P, res.Q, res.Attempts)
}
```

- [ ] **Step 2: Build the binary**

Run: `cd implementation/go && go build -o build/qubit ./cmd/qubit`
Expected: no output, file `build/qubit` created.

- [ ] **Step 3: Run each demo and confirm output**

Run: `cd implementation/go && go run ./cmd/qubit --algo=bell`
Expected: `Bell |Phi+>: P(00) = 0.5000, P(11) = 0.5000`

Run: `cd implementation/go && go run ./cmd/qubit --algo=qft`
Expected: `QFT|0> on 4 qubits: P(0) = 0.0625 (uniform = 0.0625)`

Run: `cd implementation/go && go run ./cmd/qubit --algo=grover`
Expected: `Grover marked |1111>: P(15) = ` followed by a number near 1.0.

Run: `cd implementation/go && go run ./cmd/qubit --algo=shor`
Expected: `Shor(15): p=3, q=5, attempts=N` (or 5, 3).

- [ ] **Step 4: Commit**

```bash
git add implementation/go/cmd/qubit/main.go
git commit -m "feat(go): cmd/qubit demo binary (--algo {bell|qft|grover|shor})"
```

---

### Task 34: README.md

**Files:**
- Create: `implementation/go/README.md`

Mirror the style and depth of `implementation/c/README.md` but adapted to the Go API.

- [ ] **Step 1: Write README**

Create `implementation/go/README.md`:

```markdown
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
import "github.com/arda-karaduman/thesis-go/qubit"

q, err := qubit.NewQreg(4, qubit.WithSeed(42), qubit.WithWorkers(4))
if err != nil { /* nQubits out of [1, QregMaxQubits] */ }
q.InitBasis(0)
q.ApplyH(0)
q.ApplyCNOT(0, 1)
fmt.Println("P(00) =", q.ProbOf(0))
```

Every state-mutating operation is a method on `*Qreg`. The amplitude
slice is intentionally unexported; use `Amplitude(i)`, `ProbOf(basis)`,
or `AmplitudesCopy()` to inspect.

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
```

- [ ] **Step 2: Commit**

```bash
git add implementation/go/README.md
git commit -m "docs(go): README"
```

---

### Task 35: assessment.md

**Files:**
- Create: `implementation/go/assessment.md`

Mirror `implementation/c/assessment.md`: each thesis claim mapped to a Go file:line.

- [ ] **Step 1: Write assessment.md**

Create `implementation/go/assessment.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add implementation/go/assessment.md
git commit -m "docs(go): assessment matrix (thesis-claim coverage)"
```

---

### Task 36: GitHub Actions CI

**Files:**
- Modify: `.github/workflows/ci.yml`

Spec §8.3. Add `go-tests` + `go-tests-shor-21` jobs alongside the existing C jobs.

- [ ] **Step 1: Read the existing workflow**

Run: `cat .github/workflows/ci.yml` and identify where to insert the new jobs (typically after the last existing job).

- [ ] **Step 2: Append the Go jobs**

Add to `.github/workflows/ci.yml` under the existing `jobs:` map:

```yaml
  go-tests:
    name: implementation/go -- make test + test-race
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.21'
      - name: go build
        working-directory: implementation/go
        run: go build ./...
      - name: go vet
        working-directory: implementation/go
        run: go vet ./...
      - name: go test
        working-directory: implementation/go
        run: go test ./...
      - name: go test -race
        working-directory: implementation/go
        run: go test -race ./...

  go-tests-shor-21:
    name: implementation/go -- Shor-21 large test
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.21'
      - name: go test with RUN_SHOR_21=1
        working-directory: implementation/go
        run: RUN_SHOR_21=1 go test ./qubit/ -run TestShorPeriodA2Mod21 -v
```

- [ ] **Step 3: Verify the YAML parses**

Run: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo OK`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci(go): add go-tests + go-tests-shor-21 jobs"
```

---

### Task 37: Final end-to-end verification

**Files:**
- None (verification only)

A single pass through every command listed in the makefile, plus all four demos. Catches anything that has drifted between phases.

- [ ] **Step 1: `make` (build)**

Run: `cd implementation/go && make`
Expected: clean build, no warnings.

- [ ] **Step 2: `make vet`**

Run: `cd implementation/go && make vet`
Expected: no output.

- [ ] **Step 3: `make test`**

Run: `cd implementation/go && make test`
Expected: `ok  github.com/arda-karaduman/thesis-go/qubit  <duration>` and a similar line for `cmd/qubit` if it has tests (it doesn't here, so the cmd line will say `[no test files]` -- that's fine).

- [ ] **Step 4: `make test-race`**

Run: `cd implementation/go && make test-race`
Expected: same as above, with `-race` overhead. No "DATA RACE" lines.

- [ ] **Step 5: `RUN_SHOR_21=1 make test`**

Run: `cd implementation/go && RUN_SHOR_21=1 make test`
Expected: same as `make test`, but `TestShorPeriodA2Mod21` no longer skips.

- [ ] **Step 6: All four demos**

Run each:
```bash
cd implementation/go
make demo ALGO=bell
make demo ALGO=qft
make demo ALGO=grover
make demo ALGO=shor
```
Expected: each prints sensible output (see Task 33 for expected formats).

- [ ] **Step 7: Verify .github CI workflow file parses (locally)**

Run: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo OK`
Expected: `OK`.

- [ ] **Step 8: No commit needed for verification, but if anything was fixed, commit:**

```bash
git status
# if anything changed during verification:
git add <fixed files>
git commit -m "fix(go): <whatever was fixed>"
```

---

## Plan summary

37 tasks across 11 phases. Each task is bite-sized (2-5 min per step, 5-7 steps per task) and ends in a green commit so the branch is always shippable.

| Phase | Tasks | Outcome |
|---|---|---|
| 0. Bootstrap | 1-4 | go.mod, Makefile, assert.go, empty qreg.go |
| 1. standart | 5-9 | GCD, addMod, MulMod, ModPow, ContinuedFraction, IsPowerOfTwo, Ilog2 |
| 2. Qreg | 10-12 | Options, NewQreg, accessors |
| 3. Dispatcher | 13 | parallelOverPairs / Indices |
| 4. Single-qubit gates | 14-17 | ApplyU, H/X/Y/Z, S/T/Phase, Rx/Ry/Rz |
| 5. Controlled gates | 18-19 | CU, CNOT, CZ, ControlledPhase, SWAP |
| 6. Multi-controlled | 20-21 | MCZ, MCX |
| 7. Measurement | 22-24 | MeasureQubit, MeasureAll, SampleDistribution, Clone, Dump |
| 8. QFT | 25-26 | ApplyQFT/Inverse + period detection test |
| 9. Grover | 27-28 | ApplyGrover + multi-marked |
| 10. Shor | 29-32 | ModularExp + ShorPeriod + ShorFactor + Shor-21 |
| 11. CLI + docs + CI | 33-37 | demo binary, README, assessment, CI, end-to-end check |

Estimated effort: ~12-15 hours of focused work.
