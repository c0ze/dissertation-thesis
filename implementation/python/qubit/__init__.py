"""qubit: PyTorch state-vector quantum-circuit simulator.

Sibling of ``implementation/c`` (MPI) and ``implementation/go``
(goroutines). The PyTorch backend gives device-agnostic GPU support:
the same code runs on NVIDIA CUDA, AMD ROCm, Apple Metal (MPS), and CPU.

Phase 0+1+2 ships the data model and the shared arithmetic helpers:
:class:`Qreg` construction, accessors, the qubit-axis helper, and the
``standart`` module (gcd, mod_pow, continued_fraction, is_power_of_two,
ilog2_u32) that gates and Shor's period finder will consume. Gate
methods, measurement (beyond ``prob_of`` / ``norm``), QFT, Grover, and
Shor itself land in later phases.

Public API surface:

* :class:`Qreg` -- the state-vector register class.
* :func:`qubit_axis` -- the LSB-first qubit-to-tensor-axis helper.
* Arithmetic helpers from :mod:`qubit.standart`:
  :func:`gcd_u64`, :func:`mod_pow`, :func:`continued_fraction`,
  :func:`is_power_of_two`, :func:`ilog2_u32`.
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
from .standart import (
    continued_fraction,
    gcd_u64,
    ilog2_u32,
    is_power_of_two,
    mod_pow,
)

__all__ = [
    "AMP_TOL_C128",
    "AMP_TOL_C64",
    "PROB_TOL_C128",
    "PROB_TOL_C64",
    "Qreg",
    "amp_tol_for",
    "continued_fraction",
    "gcd_u64",
    "ilog2_u32",
    "is_power_of_two",
    "mod_pow",
    "prob_tol_for",
    "qubit_axis",
]
