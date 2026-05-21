"""Programmer-error validation helpers.

Every exception raised by the `qubit` package carries a ``"qubit: "``
prefix so callers can identify simulator-originated errors at a glance
without inspecting tracebacks. Construction-time and gate preconditions
both flow through these helpers; the CLI catches at the top level.
"""

from __future__ import annotations


def raise_value(cond: bool, fmt: str, *args: object) -> None:
    """Raise ``ValueError`` with ``"qubit: " + fmt % args`` if ``cond`` is False.

    Used at the top of every public method for programmer-error
    preconditions (out-of-range qubit indices, control == target, etc.).
    Construction-time input validation (bad ``n_qubits``) goes through the
    same helper. There is no separate ``panic`` vs ``error`` split as in
    /go; Python's exception model is uniform across both layers.
    """
    if not cond:
        raise ValueError("qubit: " + fmt % args)


def raise_type(cond: bool, fmt: str, *args: object) -> None:
    """Raise ``TypeError`` with ``"qubit: " + fmt % args`` if ``cond`` is False.

    Reserved for type-shape failures (caller passed a non-tensor where a
    tensor was expected, etc.). Value-domain failures use
    :func:`raise_value` so the exception class signals which axis of the
    contract was violated.
    """
    if not cond:
        raise TypeError("qubit: " + fmt % args)
