import math

import pandas as pd
import pytest

from rawaj.audit import audit
from rawaj.metrics import (
    best_threshold,
    confusion_at,
    expected_cost,
    precision_at_k,
    precision_recall_at,
    threshold_for_precision,
)


def test_confusion_at():
    assert confusion_at([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], 0.5) == {
        "tp": 1, "fp": 1, "tn": 1, "fn": 1
    }


def test_precision_recall_at():
    assert precision_recall_at([1, 0, 1], [0.9, 0.8, 0.2], 0.5) == (
        0.5, 0.5
    )


def test_precision_at_k_and_stable_ties():
    assert precision_at_k([1, 0, 1], [0.9, 0.8, 0.2], 2) == 0.5
    assert precision_at_k([1, 0, 1], [0.5, 0.5, 0.5], 1) == 1.0
    assert precision_at_k([1, 0], [0.9, 0.1], 10) == 0.5


def test_expected_cost():
    assert expected_cost(
        [1, 0], [0.1, 0.9], 0.5, cost_fp=2, cost_fn=3
    ) == 5


def test_best_threshold():
    assert best_threshold(
        [1, 0], [0.9, 0.1], cost_fp=1, cost_fn=1
    ) == 0.9


def test_best_threshold_tie_prefers_larger_threshold():
    assert best_threshold(
        [1, 0], [0.5, 0.5], cost_fp=1, cost_fn=1
    ) == math.inf


def test_threshold_for_precision_checks_thresholds():
    assert threshold_for_precision(
        [1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1], 0.75
    ) == 0.9


def test_threshold_for_precision_returns_none():
    assert threshold_for_precision(
        [0, 0], [0.8, 0.2], 0.8
    ) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: confusion_at([0, 1], [0.2], 0.5),
        lambda: confusion_at([], [], 0.5),
        lambda: confusion_at([0, 2], [0.2, 0.8], 0.5),
        lambda: confusion_at([0, 1], [0.2, float("nan")], 0.5),
        lambda: precision_at_k([0, 1], [0.2, 0.8], 0),
    ],
)
def test_invalid_metric_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_audit_basic_report():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "target": [1, 0, 1],
        "time": ["2026-01-01", "2026-01-02", "bad-date"],
        "value": [5, 5, 5],
    })
    result = audit(df, target="target", id_col="id", time_col="time")

    assert result["n_rows"] == 3
    assert result["duplicate_rows"] == 0
    assert result["constant_columns"] == ["value"]
    assert result["bad_timestamps"] == 1
    assert result["positive_rate"] == pytest.approx(2 / 3)


def test_audit_missing_values_and_duplicate_ids():
    df = pd.DataFrame({
        "id": [10, 20, 20, None],
        "target": [0, 1, 1, None],
        "time": ["2026-01-01", "2026-01-02", "2026-01-03", ""],
        "value": [5, 6, 7, 8],
    })
    result = audit(df, target="target", id_col="id", time_col="time")

    assert result["duplicate_ids"] == [20.0]
    assert result["unlabelled_rows"] == 1
    assert result["missing"]["target"] == pytest.approx(0.25)


def test_audit_missing_required_column_raises_key_error():
    df = pd.DataFrame({"target": [0], "id": [1]})
    with pytest.raises(KeyError):
        audit(df, target="target", id_col="id", time_col="time")
