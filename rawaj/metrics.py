import math
from numbers import Real


def _validate_inputs(y_true, scores):
    y = list(y_true)
    s = list(scores)

    if len(y) != len(s):
        raise ValueError("y_true and scores must have equal lengths")
    if not y:
        raise ValueError("Inputs must not be empty")

    for label in y:
        if label not in (0, 1):
            raise ValueError("Labels must be 0 or 1")

    for score in s:
        if not isinstance(score, Real) or math.isnan(score):
            raise ValueError("Scores must be numeric and not NaN")

    return y, s


def confusion_at(y_true, scores, threshold: float) -> dict:
    y, s = _validate_inputs(y_true, scores)
    tp = fp = tn = fn = 0

    for actual, score in zip(y, s):
        predicted = score >= threshold
        if actual == 1 and predicted:
            tp += 1
        elif actual == 0 and predicted:
            fp += 1
        elif actual == 0 and not predicted:
            tn += 1
        else:
            fn += 1

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    counts = confusion_at(y_true, scores, threshold)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def precision_at_k(y_true, scores, k: int) -> float:
    y, s = _validate_inputs(y_true, scores)
    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, len(y))
    ranked_indices = sorted(range(len(s)), key=lambda i: (-s[i], i))
    hits = sum(y[i] == 1 for i in ranked_indices[:k])
    return hits / k


def expected_cost(
    y_true,
    scores,
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    counts = confusion_at(y_true, scores, threshold)
    return counts["fp"] * cost_fp + counts["fn"] * cost_fn


def best_threshold(
    y_true,
    scores,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    y, s = _validate_inputs(y_true, scores)
    candidates = set(s)
    candidates.add(math.inf)

    best_t = None
    best_cost = math.inf

    for threshold in candidates:
        cost = expected_cost(
            y, s, threshold, cost_fp=cost_fp, cost_fn=cost_fn
        )
        if cost < best_cost or (
            cost == best_cost and (best_t is None or threshold > best_t)
        ):
            best_cost = cost
            best_t = threshold

    return best_t


def threshold_for_precision(
    y_true,
    scores,
    min_precision: float,
) -> float | None:
    y, s = _validate_inputs(y_true, scores)

    for threshold in sorted(set(s)):
        precision, _ = precision_recall_at(y, s, threshold)
        if precision >= min_precision:
            return threshold

    return None
