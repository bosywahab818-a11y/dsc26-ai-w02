from collections import Counter

def audit_extract(df, label_col=None, timestamp_col=None):
    """Return a basic data-quality audit for a pandas DataFrame."""
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [
            c for c in df.columns if df[c].nunique(dropna=False) <= 1
        ],
    }
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"Label column not found: {label_col}")
        report["label_distribution"] = df[label_col].value_counts(
            dropna=False
        ).to_dict()
    if timestamp_col is not None:
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column not found: {timestamp_col}")
        parsed = __import__("pandas").to_datetime(
            df[timestamp_col], errors="coerce", utc=True
        )
        report["invalid_timestamps"] = int(parsed.isna().sum())
    return report

def audit_data(df, label_col=None, timestamp_col=None):
    return audit_extract(df, label_col, timestamp_col)
