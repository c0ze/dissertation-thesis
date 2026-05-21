"""Phase-0 smoke test: the package imports and re-exports its public surface."""

from __future__ import annotations


def test_package_imports() -> None:
    import qubit

    # Public surface promised by __init__.py.
    assert hasattr(qubit, "Qreg")
    assert hasattr(qubit, "qubit_axis")
    assert hasattr(qubit, "amp_tol_for")
    assert hasattr(qubit, "prob_tol_for")
    for name in (
        "AMP_TOL_C64",
        "AMP_TOL_C128",
        "PROB_TOL_C64",
        "PROB_TOL_C128",
    ):
        assert hasattr(qubit, name), f"missing public constant {name}"


def test_torch_available() -> None:
    # Sanity-check the only third-party dependency.
    import torch

    # Any modern PyTorch is fine for Phase 0+1; gate code (Phase 4+) will
    # exercise the complex-tensor surface that requires >= 2.4 in practice.
    assert torch.__version__.split(".")[0] in {"2", "3"}
