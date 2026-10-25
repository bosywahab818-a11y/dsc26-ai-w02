# Threshold Decision — Rawaj Pay

## Objective
Choose a decision threshold based on the business cost of false positives and false negatives, rather than accuracy alone.

## Audit Before Evaluation
Before choosing a threshold, inspect missing values, duplicate records, label distribution, timestamp validity, constant columns, and possible target leakage.

## Evaluation Plan
Compare candidate thresholds on validation data using confusion counts, precision, recall, and expected business cost. Keep the final test set separate from threshold selection.

## Decision
The final threshold must be justified using measured validation results and documented business costs. Do not claim a threshold is optimal until the data and evaluation results have been inspected.
