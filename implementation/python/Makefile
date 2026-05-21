# ---------------------------------------------------------------------------
# Makefile for implementation/python -- PyTorch state-vector quantum simulator.
#
# Targets:
#   make sync        uv sync --group dev          (install/refresh .venv)
#   make test        uv run pytest                (full test suite)
#   make test-cpu    PYTORCH_DISABLE_MPS=1 ... (force CPU even on Apple Silicon)
#   make lint        uv run ruff check .
#   make format      uv run ruff format .
#   make typecheck   uv run mypy qubit tests
#   make check       lint + typecheck + test      (the standard PR gate)
#   make demo ALGO=bell|qft|grover|shor
#                    uv run qubit-demo --algo $(ALGO)
#   make clean       rm -rf .venv .pytest_cache .ruff_cache .mypy_cache
#
# The implementation/python module is managed by `uv`; the Makefile is a
# thin convenience wrapper. Most one-off operations can use `uv run <tool>`
# directly.
# ---------------------------------------------------------------------------

ALGO ?= bell

.PHONY: sync test test-cpu lint format typecheck check demo clean

sync:
	uv sync --group dev

test:
	uv run pytest

test-cpu:
	PYTORCH_DISABLE_MPS=1 CUDA_VISIBLE_DEVICES= uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy qubit tests

check: lint typecheck test

demo:
	uv run qubit-demo --algo $(ALGO)

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
