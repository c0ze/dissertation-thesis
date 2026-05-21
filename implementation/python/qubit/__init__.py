"""qubit: PyTorch state-vector quantum-circuit simulator.

Sibling of ``implementation/c`` (MPI) and ``implementation/go``
(goroutines). The PyTorch backend gives device-agnostic GPU support:
the same code runs on NVIDIA CUDA, AMD ROCm, Apple Metal (MPS), and CPU.

Phase 0+1 of the implementation ships only the data model:
:class:`Qreg` construction, accessors, and the qubit-axis helper. Gate
methods, measurement (beyond ``prob_of`` / ``norm``), QFT, Grover, and
Shor land in later phases.

Public API surface (Phase 0+1):

* :class:`Qreg` -- the state-vector register class.
* :func:`qubit_axis` -- the LSB-first qubit-to-tensor-axis helper.
* Numerical-tolerance constants ``AMP_TOL_C64`` / ``AMP_TOL_C128`` /
  ``PROB_TOL_C64`` / ``PROB_TOL_C128`` plus the dtype-keyed helpers
  ``amp_tol_for`` and ``prob_tol_for`` for test-side use.
"""

from __future__ import annotations

from ._axis import qubit_axis
from .qreg import (
    AMP_TOL_C64,
    AMP_TOL_C128,
    PROB_TOL_C64,
    PROB_TOL_C128,
    Qreg,
    amp_tol_for,
    prob_tol_for,
)

__all__ = [
    "AMP_TOL_C128",
    "AMP_TOL_C64",
    "PROB_TOL_C128",
    "PROB_TOL_C64",
    "Qreg",
    "amp_tol_for",
    "prob_tol_for",
    "qubit_axis",
]
