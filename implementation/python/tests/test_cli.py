"""Tests for the qubit-demo CLI.

Drives :func:`qubit.cli.main` directly (no subprocess) so the tests
stay fast and the captured stdout is straightforward to assert on.
``--help`` and unknown-algo paths both exit through argparse's
``SystemExit``; demo paths return integer status codes.
"""

from __future__ import annotations

import pytest

from qubit.cli import main


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    # argparse's --help prints to stdout and SystemExits with code 0.
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "qubit-demo" in captured.out
    assert "--algo" in captured.out


def test_cli_default_algo_is_bell(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "Bell" in out
    assert "P(00)" in out


@pytest.mark.parametrize("algo", ["bell", "qft", "grover", "shor"])
def test_cli_each_demo_runs(
    capsys: pytest.CaptureFixture[str], algo: str
) -> None:
    code = main(["--algo", algo])
    assert code == 0
    out = capsys.readouterr().out
    assert out.strip() != "", f"{algo} demo produced no output"


def test_cli_bell_output_shape(capsys: pytest.CaptureFixture[str]) -> None:
    # Pin the specific numbers so any regression in the Bell-state
    # pipeline (H, CNOT, prob_of) gets caught here too.
    code = main(["--algo", "bell"])
    assert code == 0
    out = capsys.readouterr().out
    assert "P(00) = 0.5000" in out
    assert "P(11) = 0.5000" in out


def test_cli_shor_output_shape(capsys: pytest.CaptureFixture[str]) -> None:
    # The Shor demo is seeded -> deterministic; pin the factors.
    code = main(["--algo", "shor"])
    assert code == 0
    out = capsys.readouterr().out
    # The seeded run lands on attempt 1 with p=3, q=5 (order
    # depends on which of a^(r/2) +/- 1 gcd-resolves first).
    assert "p=3" in out or "p=5" in out
    assert "q=5" in out or "q=3" in out


def test_cli_unknown_algo_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # argparse rejects unknown choices and SystemExits with code 2.
    with pytest.raises(SystemExit) as exc_info:
        main(["--algo", "nonsense"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_cli_handles_unexpected_exception(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # If a demo function raises, the CLI catches at the top level,
    # prints a qubit-demo-prefixed error to stderr, and returns 1.
    def boom() -> None:
        raise RuntimeError("qubit: simulated failure")

    monkeypatch.setitem(
        __import__("qubit.cli", fromlist=["_DEMOS"])._DEMOS,
        "bell",
        boom,
    )
    code = main(["--algo", "bell"])
    assert code == 1
    err = capsys.readouterr().err
    assert "qubit-demo:" in err
    assert "simulated failure" in err
