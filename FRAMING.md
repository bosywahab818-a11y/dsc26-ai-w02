# Rawaj Pay — Threshold Decision

## Decision
The threshold should reflect the business impact of false positives and false negatives, not accuracy alone. I will choose a final threshold only after auditing the extract and evaluating candidate thresholds on validation data. No final threshold is claimed yet.

## Metric
I will compare thresholds using true positives, false positives, true negatives, false negatives, precision, recall, and expected cost. The cost calculation will use the business costs assigned to false positives and false negatives.

## Baseline
The baseline will be a simple threshold-based classifier evaluated on the same validation data. I will compare its results with the cost-minimizing threshold and report measured results only after running the evaluation.

## Classical ML or LLM
I would start with a classical ML approach if labelled data is available and supports reliable validation. An LLM is not my default for this numerical threshold-selection task because explicit scores, metrics, and costs provide a more direct and reproducible decision process.
