import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _missing_mask(series):
    """Treat null values and blank strings as missing."""
    blank_strings = series.map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    return series.isna() | blank_strings


def _roc_auc(y_true, scores):
    """Calculate ROC AUC using average ranks, including tied scores."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return None

    ranks = pd.Series(scores).rank(method="average").to_numpy()
    positive_rank_sum = ranks[y == 1].sum()

    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def audit(
    df: pd.DataFrame,
    *,
    target: str,
    id_col: str,
    time_col: str,
) -> dict:
    """Audit transfer data quality and flag potential target leakage."""
    for column in (target, id_col, time_col):
        if column not in df.columns:
            raise KeyError(column)

    missing_masks = {
        column: _missing_mask(df[column])
        for column in df.columns
    }

    missing = {
        column: float(mask.mean()) if len(df) else 0.0
        for column, mask in missing_masks.items()
    }

    id_missing = missing_masks[id_col]
    duplicate_ids = (
        df.loc[~id_missing, id_col]
        .loc[lambda values: values.duplicated(keep=False)]
        .drop_duplicates()
        .tolist()
    )
    duplicate_ids = sorted(duplicate_ids)

    constant_columns = []
    for column in df.columns:
        if column == target:
            continue
        values = df.loc[~missing_masks[column], column]
        if values.nunique(dropna=True) <= 1:
            constant_columns.append(column)

    target_missing = missing_masks[target]
    labelled = df.loc[~target_missing]
    labels = labelled[target]
    n_labelled = len(labelled)

    positive_rate = (
        float((labels == 1).sum() / n_labelled)
        if n_labelled else 0.0
    )

    timestamp_missing = missing_masks[time_col]
    parsed_timestamps = pd.to_datetime(
        df[time_col].where(~timestamp_missing),
        errors="coerce",
    )
    bad_timestamps = int(
        ((~timestamp_missing) & parsed_timestamps.isna()).sum()
    )
    valid_timestamps = parsed_timestamps.dropna()

    time_range = (
        [None, None] if valid_timestamps.empty else [
            valid_timestamps.min().isoformat(),
            valid_timestamps.max().isoformat(),
        ]
    )

    leakage_suspects = []
    if n_labelled >= 20 and labels.nunique() == 2:
        y = labels.to_numpy()

        for column in df.columns:
            if column in {target, id_col, time_col}:
                continue

            series = df[column]
            missing_mask = missing_masks[column]

            # Missingness is itself a distinct value for this check.
            distinct_values = series.mask(missing_mask)
            if distinct_values.nunique(dropna=False) <= 1:
                continue

            labelled_series = series.loc[~target_missing]
            labelled_missing = missing_mask.loc[~target_missing]
            suspect = False

            if is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
                observed = ~labelled_missing
                observed_values = labelled_series.loc[observed]
                observed_labels = labels.loc[observed]

                if (
                    len(observed_values) >= 20
                    and observed_labels.nunique() == 2
                ):
                    auc = _roc_auc(
                        observed_labels,
                        observed_values,
                    )
                    suspect = auc is not None and (
                        auc >= 0.98 or auc <= 0.02
                    )

            if not suspect:
                missing_auc = _roc_auc(
                    y,
                    labelled_missing.astype(int).to_numpy(),
                )
                suspect = missing_auc is not None and (
                    missing_auc >= 0.98 or missing_auc <= 0.02
                )

            if suspect:
                leakage_suspects.append(column)

    return {
        "n_rows": int(len(df)),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": sorted(constant_columns),
        "unlabelled_rows": int(target_missing.sum()),
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": sorted(leakage_suspects),
    }
