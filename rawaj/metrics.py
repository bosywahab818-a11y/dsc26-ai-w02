def confusion_counts(y_true, y_score, threshold=0.5, positive_label=1):
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have equal lengths")
    tp = fp = tn = fn = 0
    for actual, score in zip(y_true, y_score):
        predicted = score >= threshold
        positive = actual == positive_label
        if positive and predicted:
            tp += 1
        elif not positive and predicted:
            fp += 1
        elif not positive and not predicted:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def precision_recall(y_true, y_score, threshold=0.5, positive_label=1):
    counts = confusion_counts(y_true, y_score, threshold, positive_label)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }

def precision_at_k(y_true, y_score, k, positive_label=1):
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have equal lengths")
    if k <= 0:
        raise ValueError("k must be positive")
    if not y_true:
        return 0.0
    k = min(k, len(y_true))
    ranked = sorted(zip(y_true, y_score), key=lambda pair: pair[1], reverse=True)
    hits = sum(actual == positive_label for actual, _ in ranked[:k])
    return hits / k

def expected_cost(y_true, y_score, threshold=0.5,
                  false_positive_cost=1.0, false_negative_cost=1.0,
                  positive_label=1):
    c = confusion_counts(y_true, y_score, threshold, positive_label)
    return c["fp"] * false_positive_cost + c["fn"] * false_negative_cost

def choose_threshold(y_true, y_score, false_positive_cost=1.0,
                     false_negative_cost=1.0, thresholds=None,
                     positive_label=1):
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have equal lengths")
    if not y_true:
        raise ValueError("Inputs must not be empty")
    if thresholds is None:
        thresholds = sorted(set([0.0, 0.5, 1.0, *y_score]))
    results = [
        (expected_cost(y_true, y_score, t, false_positive_cost,
                       false_negative_cost, positive_label), t)
        for t in thresholds
    ]
    cost, threshold = min(results, key=lambda item: (item[0], item[1]))
    return {"threshold": threshold, "expected_cost": cost}
