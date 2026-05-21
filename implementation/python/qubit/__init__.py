"""qubit: PyTorch state-vector quantum-circuit simulator.

Sibling of ``implementation/c`` (MPI) and ``implementation/go``
(goroutines). The PyTorch backend gives device-agnostic GPU support:
the same code runs on NVIDIA CUDA, AMD ROCm, Apple Metal (MPS), and CPU.

Through Phase 6 this ships: the data model (:class:`Qreg` construction,
accessors, the qubit-axis helper), the ``standart`` arithmetic helpers
(gcd, mod_pow, continued_fraction, is_power_of_two, ilog2_u32), the
single-qubit gate primitives (``apply_u`` plus the standard named
gates), the controlled and multi-controlled gates (``apply_cu`` /
``apply_cnot`` / ``apply_cz`` / ``apply_controlled_phase`` /
``apply_swap`` / ``apply_multi_controlled_z`` /
``apply_multi_controlled_x``), the measurement primitives
(``measure_qubit`` / ``measure_all`` / ``sample_distribution`` /
``clone`` / ``dump``), and the Quantum Fourier Transform
(``apply_qft`` / ``apply_qft_inverse``). Grover and Shor land in
later phases.

Public API surface:

* :class:`Qreg` -- the state-vector register class with methods for
  every gate (``q.apply_h(0)``).
* Function-style gate equivalents (``apply_h(q, 0)``) re-exported from
  :mod:`qubit.gates_single`. Both shapes are first-class.
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
from .gates_controlled import (
    apply_cnot,
    apply_controlled_phase,
    apply_cu,
    apply_cz,
    apply_swap,
)
from .gates_multi import (
    apply_multi_controlled_x,
    apply_multi_controlled_z,
)
from .gates_single import (
    apply_h,
    apply_phase,
    apply_rx,
    apply_ry,
    apply_rz,
    apply_s,
    apply_t,
    apply_u,
    apply_x,
    apply_y,
    apply_z,
)
from .measure import (
    clone,
    dump,
    measure_all,
    measure_qubit,
    sample_distribution,
)
from .qft import (
    apply_qft,
    apply_qft_inverse,
)
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
    "apply_cnot",
    "apply_controlled_phase",
    "apply_cu",
    "apply_cz",
    "apply_h",
    "apply_multi_controlled_x",
    "apply_multi_controlled_z",
    "apply_phase",
    "apply_qft",
    "apply_qft_inverse",
    "apply_rx",
    "apply_ry",
    "apply_rz",
    "apply_s",
    "apply_swap",
    "apply_t",
    "apply_u",
    "apply_x",
    "apply_y",
    "apply_z",
    "clone",
    "continued_fraction",
    "dump",
    "gcd_u64",
    "ilog2_u32",
    "is_power_of_two",
    "measure_all",
    "measure_qubit",
    "mod_pow",
    "prob_tol_for",
    "qubit_axis",
    "sample_distribution",
]
