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
