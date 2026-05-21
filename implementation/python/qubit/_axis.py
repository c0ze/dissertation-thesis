"""Qubit-to-tensor-axis mapping.

The flat amplitude vector ``q._amp`` is indexed by basis state with the
same convention as ``/c`` and ``/go``: **qubit 0 is the least significant
bit** of the basis index. For a 3-qubit register, the state
``|q2 q1 q0>`` corresponds to ``amp[4*q2 + 2*q1 + q0]``.

When a gate function reshapes the flat vector to the n-D view
``(2,) * n_qubits`` for einsum/tensordot, the tensor axis corresponding
to qubit ``q`` is :func:`qubit_axis` ``= n_qubits - 1 - q`` (so qubit 0
lives on the rightmost / fastest-varying axis). This module is the single
home of that conversion; gates call it instead of recomputing
``n - 1 - q`` ad hoc, so a silent endian flip cannot be introduced one
gate at a time.
"""

from __future__ import annotations

from ._assert import raise_value


def qubit_axis(target: int, n_qubits: int) -> int:
    """Return the tensor-axis index corresponding to qubit ``target``.

    For the ``(2,) * n`` reshape view of a flat ``2**n``-amplitude state
    vector, qubit 0 lives on axis ``n - 1`` (LSB, fastest-varying),
    qubit 1 on axis ``n - 2``, ..., qubit ``n - 1`` on axis 0 (MSB).

    Raises :class:`ValueError` if ``target`` is not in ``[0, n_qubits)``.
    """
    raise_value(
        0 <= target < n_qubits,
        "qubit_axis: target=%d out of [0, %d)",
        target,
        n_qubits,
    )
    return n_qubits - 1 - target
