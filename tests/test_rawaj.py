from rawaj.metrics import confusion_counts, precision_recall, precision_at_k, expected_cost, choose_threshold
from rawaj.audit import audit_extract
import pandas as pd

def test_confusion_counts():
    result = confusion_counts([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], 0.5)
    assert result == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}

def test_precision_recall():
    result = precision_recall([1, 0, 1], [0.9, 0.8, 0.2], 0.5)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5

def test_precision_at_k():
    assert precision_at_k([1, 0, 1], [0.9, 0.8, 0.2], 2) == 0.5

def test_expected_cost():
    assert expected_cost([1, 0], [0.1, 0.9], 0.5, 2, 3) == 5

def test_choose_threshold():
    result = choose_threshold([1, 0], [0.9, 0.1])
    assert result["expected_cost"] == 0

def test_audit_extract():
    df = pd.DataFrame({"label": [1, 0, 1], "value": [5, 5, 5]})
    result = audit_extract(df, "label")
    assert result["rows"] == 3
    assert result["duplicate_rows"] == 1
    assert "value" in result["constant_columns"]
