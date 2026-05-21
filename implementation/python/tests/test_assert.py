"""Tests for the package-private exception helpers."""

from __future__ import annotations

import pytest

from qubit._assert import raise_type, raise_value


def test_raise_value_passes_when_true() -> None:
    raise_value(True, "should not fire")


def test_raise_value_raises_with_prefix() -> None:
    with pytest.raises(ValueError) as exc_info:
        raise_value(False, "bad: %d", 42)
    msg = str(exc_info.value)
    assert msg.startswith("qubit: ")
    assert "bad: 42" in msg


def test_raise_value_formats_multiple_args() -> None:
    with pytest.raises(ValueError, match=r"qubit: range \[5, 10\)"):
        raise_value(False, "range [%d, %d)", 5, 10)


def test_raise_type_passes_when_true() -> None:
    raise_type(True, "should not fire")


def test_raise_type_raises_with_prefix() -> None:
    with pytest.raises(TypeError) as exc_info:
        raise_type(False, "expected %s, got %s", "Tensor", "list")
    msg = str(exc_info.value)
    assert msg.startswith("qubit: ")
    assert "expected Tensor, got list" in msg


def test_raise_value_and_raise_type_use_different_classes() -> None:
    # A test that catches ValueError must NOT catch the TypeError version.
    with pytest.raises(TypeError):
        raise_type(False, "should be TypeError")
    with pytest.raises(ValueError):
        raise_value(False, "should be ValueError")
