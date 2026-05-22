"""Command-line demo for the qubit library.

Installed as ``qubit-demo`` via ``[project.scripts]`` in
:file:`pyproject.toml`. Mirrors the demos in ``/c``'s ``cmd/qubit`` and
``/go``'s ``cmd/qubit/main.go``: a quick smoke check that the library
runs end-to-end on whatever device the user has, for each of the four
canonical algorithm-level demos (Bell state, QFT, Grover, Shor).

Run from a checkout of the repo::

    cd implementation/python
    uv sync --group dev    # one-time install
    uv run qubit-demo --algo bell
    uv run qubit-demo --algo qft
    uv run qubit-demo --algo grover
    uv run qubit-demo --algo shor

All demos are deterministic (seeded RNG where used) so successive runs
produce identical output, and they exit with status 0 on success. Any
:class:`qubit:`-prefixed exception or unexpected error is printed to
stderr and the process exits non-zero.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections.abc import Callable
from typing import Any

# PyTorch's optional NumPy bridge warns on import when NumPy is absent.
# We don't depend on NumPy, so the warning is pure noise in demo output.
# Filter it before importing the modules that pull torch in.
warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy",
    category=UserWarning,
)

from .grover import apply_grover  # noqa: E402  -- after warning filter
from .qreg import Qreg  # noqa: E402
from .shor import shor_factor  # noqa: E402


def _demo_bell() -> None:
    """Bell state ``(|00> + |11>) / sqrt(2)`` from H + CNOT."""
    q = Qreg(2, device="cpu")
    q.init_basis(0)
    q.apply_h(0)
    q.apply_cnot(0, 1)
    print(
        f"Bell |Phi+>: P(00) = {q.prob_of(0):.4f}, "
        f"P(11) = {q.prob_of(3):.4f}"
    )


def _demo_qft() -> None:
    """``QFT|0>`` on a 4-qubit register -- uniform distribution."""
    n = 4
    q = Qreg(n, device="cpu")
    q.init_basis(0)
    q.apply_qft()
    expected = 1.0 / (1 << n)
    print(
        f"QFT|0> on {n} qubits: "
        f"P(0) = {q.prob_of(0):.4f} (uniform = {expected:.4f})"
    )


def _demo_grover() -> None:
    """Grover finds ``|1111>`` on a 4-qubit register.

    Uses :func:`apply_multi_controlled_z` as the oracle (which marks
    ``|1...1>`` over the searched qubits). The default iteration count
    is the textbook optimum for one marked item among 16 -- 3 rounds,
    giving an analytic recovery probability of ~0.961.
    """
    n = 4
    q = Qreg(n, device="cpu")
    q.init_basis(0)

    def oracle(q_inner: Qreg, _user: Any) -> None:
        # Mark |1...1> by phase-flipping the all-ones amplitude.
        q_inner.apply_multi_controlled_z(list(range(n)))

    iters = int(math.pi / 4.0 * math.sqrt(1 << n))
    apply_grover(q, n, oracle, iterations=iters)
    print(
        f"Grover marked |1{'1' * (n - 1)}>: "
        f"P({(1 << n) - 1}) = {q.prob_of((1 << n) - 1):.4f}"
    )


def _demo_shor() -> None:
    """Factor ``N = 15`` end-to-end via :func:`shor_factor`.

    Seeded for determinism so this demo is reproducible across runs.
    """
    res = shor_factor(15, seed=42)
    print(
        f"Shor(15, seed=42): p={res.p}, q={res.q}, "
        f"attempts={res.attempts}"
    )


_DEMOS: dict[str, Callable[[], None]] = {
    "bell": _demo_bell,
    "qft": _demo_qft,
    "grover": _demo_grover,
    "shor": _demo_shor,
}


def main(argv: list[str] | None = None) -> int:
    """Run the ``qubit-demo`` CLI. Returns 0 on success, 1 on error.

    Argparse handles ``--help`` and unknown options; both exit through
    ``SystemExit`` rather than our exception path. Demo errors go to
    stderr with a non-zero return.
    """
    parser = argparse.ArgumentParser(
        prog="qubit-demo",
        description=(
            "Run a quick algorithm-level demo of the qubit library. "
            "Mirrors the bell/qft/grover/shor demos in /c and /go."
        ),
    )
    parser.add_argument(
        "--algo",
        choices=sorted(_DEMOS),
        default="bell",
        help="Which algorithm to demo (default: bell).",
    )
    args = parser.parse_args(argv)

    try:
        _DEMOS[args.algo]()
    except Exception as exc:  # noqa: BLE001 -- intentional catch-all at CLI top.
        print(f"qubit-demo: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
