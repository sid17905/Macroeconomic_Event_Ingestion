import math
import pytest
from src.utils.parsing import parse_numeric_value, compute_surprise


def test_parse_numeric_percentages():
    assert parse_numeric_value("3.2%") == 3.2
    assert parse_numeric_value("-0.4%") == -0.4
    assert parse_numeric_value("+1.5%") == 1.5


def test_parse_numeric_multipliers():
    assert parse_numeric_value("250K") == 250000.0
    assert parse_numeric_value("150k") == 150000.0
    assert parse_numeric_value("-45K") == -45000.0
    assert parse_numeric_value("1.5M") == 1500000.0
    assert parse_numeric_value("2.1B") == 2100000000.0
    assert parse_numeric_value("1,250K") == 1250000.0


def test_parse_numeric_currency_and_signs():
    assert parse_numeric_value("$25.4") == 25.4
    assert parse_numeric_value("+$100") == 100.0
    assert parse_numeric_value("-0.75") == -0.75
    assert parse_numeric_value(42) == 42.0
    assert parse_numeric_value(-3.14) == -3.14


def test_parse_numeric_nulls():
    assert parse_numeric_value(None) is None
    assert parse_numeric_value("") is None
    assert parse_numeric_value("   ") is None
    assert parse_numeric_value("-") is None
    assert parse_numeric_value("--") is None
    assert parse_numeric_value("N/A") is None
    assert parse_numeric_value("nan") is None
    assert parse_numeric_value(float("nan")) is None


def test_compute_surprise():
    assert compute_surprise(3.2, 3.0) == pytest.approx(0.2)
    assert compute_surprise(250000.0, 200000.0) == pytest.approx(50000.0)
    assert compute_surprise(None, 2.0) is None
    assert compute_surprise(2.0, None) is None
    assert compute_surprise(float("nan"), 2.0) is None
