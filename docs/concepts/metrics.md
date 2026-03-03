# Evaluation Metrics

This document provides a comprehensive reference for all metrics used in CHIMERA.

## Overview

CHIMERA uses different metrics for each evaluation track:

| Track | Primary Metrics |
|-------|----------------|
| Calibration | ECE, MCE, Brier Score |
| Error Detection | F1, Precision, Recall |
| Knowledge Boundary | Abstention Rate, Appropriate Abstention F1 |
| Self-Correction | Detection Rate, Correction Accuracy |

## Calibration Metrics

### Expected Calibration Error (ECE)

**ECE** measures the average gap between confidence and accuracy across all predictions.

**Formula:**

$$ECE = \sum_{b=1}^{B} \frac{n_b}{N} |acc(b) - conf(b)|$$

Where:
- $B$ = number of bins
- $n_b$ = number of samples in bin $b$
- $N$ = total number of samples
- $acc(b)$ = accuracy of samples in bin $b$
- $conf(b)$ = average confidence in bin $b$

**Interpretation:**
- **ECE = 0**: Perfect calibration
- **ECE = 0.1**: Average 10% gap between confidence and accuracy
- **ECE > 0.2**: Poor calibration

**Example:**

```python
from chimera.metrics.calibration import expected_calibration_error

confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
correctness = [True, True, False, True, False]

ece = expected_calibration_error(confidences, correctness, n_bins=10)
print(f"ECE: {ece:.4f}")
```

### Maximum Calibration Error (MCE)

**MCE** measures the worst-case calibration gap across all bins.

**Formula:**

$$MCE = \max_{b \in \{1, ..., B\}} |acc(b) - conf(b)|$$

**Interpretation:**
- Highlights the confidence range with worst calibration
- Important for safety-critical applications
- Even if average calibration is good, MCE reveals problematic regions

### Brier Score

**Brier Score** measures the mean squared error of probabilistic predictions.

**Formula:**

$$Brier = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2$$

Where:
- $p_i$ = predicted probability (confidence)
- $o_i$ = outcome (1 if correct, 0 if incorrect)

**Interpretation:**
- **Brier = 0**: Perfect predictions
- **Brier = 0.25**: Equivalent to always predicting 50%
- **Brier < 0.2**: Generally good probabilistic predictions

**Decomposition:**

Brier Score can be decomposed into:
- **Reliability**: Calibration component
- **Resolution**: How much predictions vary
- **Uncertainty**: Inherent difficulty of the task

### Reliability Diagram

A **reliability diagram** visualizes calibration by plotting:
- X-axis: Binned confidence levels
- Y-axis: Actual accuracy within each bin

```
    1.0 ┤                        ●
        │                    ●   
    0.8 ┤                ●       
        │            ●           Perfect calibration (diagonal)
    0.6 ┤        ●               
        │    ●                   
    0.4 ┤●                       
        │
    0.2 ┤
        │
    0.0 ┼────┬────┬────┬────┬────
        0.0  0.2  0.4  0.6  0.8  1.0
                Confidence
```

Points above the diagonal indicate **underconfidence**.
Points below the diagonal indicate **overconfidence**.

## Error Detection Metrics

### Precision

**Precision** measures how many detected errors are actual errors.

$$Precision = \frac{TP}{TP + FP}$$

Where:
- $TP$ = True Positives (correctly identified errors)
- $FP$ = False Positives (flagged as errors but actually correct)

### Recall

**Recall** measures how many actual errors were detected.

$$Recall = \frac{TP}{TP + FN}$$

Where:
- $FN$ = False Negatives (errors that were missed)

### F1 Score

**F1 Score** is the harmonic mean of precision and recall.

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

**Interpretation:**
- **F1 = 1.0**: Perfect error detection
- **F1 = 0.5**: Moderate performance
- **F1 < 0.3**: Poor error detection

### Error Type Breakdown

CHIMERA provides per-error-type metrics:

```python
from chimera.metrics.error_detection import compute_error_detection_metrics

metrics = compute_error_detection_metrics(predictions, ground_truth)
print(metrics.by_error_type)
# {
#     "factual": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
#     "logical": {"precision": 0.72, "recall": 0.65, "f1": 0.68},
#     "computational": {"precision": 0.91, "recall": 0.88, "f1": 0.89},
# }
```

## Knowledge Boundary Metrics

### Abstention Rate

**Abstention Rate** measures how often the model declines to answer.

$$AbstentionRate = \frac{\text{Abstentions}}{\text{Total Questions}}$$

**Interpretation:**
- Too low: Model answers questions it shouldn't
- Too high: Model is overly cautious
- Optimal: Matches the true rate of unanswerable questions

### Appropriate Abstention

We measure whether abstentions are appropriate:

| | Model Answers | Model Abstains |
|---|---|---|
| **Answerable** | ✓ Correct behavior | ✗ False abstention |
| **Unanswerable** | ✗ False confidence | ✓ Correct abstention |

**Appropriate Abstention F1:**

$$F1_{abstention} = 2 \cdot \frac{P_{abs} \cdot R_{abs}}{P_{abs} + R_{abs}}$$

Where:
- $P_{abs}$ = Precision of abstentions (abstentions that were correct to make)
- $R_{abs}$ = Recall of abstentions (unanswerable questions that were abstained)

### Knowledge Boundary AUC

We compute the Area Under the ROC Curve for the model's ability to distinguish answerable from unanswerable questions using its confidence scores.

## Self-Correction Metrics

### Detection Rate

**Detection Rate** measures how often the model correctly identifies the corrupted step.

$$DetectionRate = \frac{\text{Correctly Identified Corruptions}}{\text{Total Corrupted Chains}}$$

### Correction Accuracy

**Correction Accuracy** measures how often the model fixes the error correctly.

$$CorrectionAccuracy = \frac{\text{Correct Fixes}}{\text{Attempted Fixes}}$$

### End-to-End Success

**E2E Success** requires both detection and correction to be correct.

$$E2E = DetectionRate \times CorrectionAccuracy$$

### Correction Quality

For partial credit, we measure how close the correction is to the ground truth:

- **Exact Match**: Correction matches expected answer
- **Semantic Similarity**: Embedding similarity to ground truth
- **Reasoning Alignment**: Steps align with expected reasoning

## Aggregate Metrics

### Track Score

Each track produces a normalized score from 0-1:

| Track | Score Formula |
|-------|--------------|
| Calibration | $1 - ECE$ |
| Error Detection | $F1$ |
| Knowledge Boundary | $AppropriateAbstention_{F1}$ |
| Self-Correction | $E2E$ |

### Overall CHIMERA Score

The **overall CHIMERA score** combines track scores:

$$CHIMERA = \sum_{t=1}^{4} w_t \cdot Score_t$$

Default weights: $w_1 = w_2 = w_3 = w_4 = 0.25$

Custom weights can emphasize specific capabilities:

```python
from chimera.evaluation import EvaluationPipeline

pipeline = EvaluationPipeline(config)
results = pipeline.run()

# Default equal weights
print(f"Overall: {results.overall_score:.2%}")

# Custom weights emphasizing calibration
custom_weights = {
    "calibration": 0.4,
    "error_detection": 0.2,
    "knowledge_boundary": 0.2,
    "self_correction": 0.2,
}
weighted_score = results.get_weighted_score(custom_weights)
```

## Statistical Significance

### Confidence Intervals

CHIMERA computes bootstrap confidence intervals for all metrics:

```python
from chimera.metrics.calibration import compute_ece_with_ci

ece, ci_low, ci_high = compute_ece_with_ci(
    confidences, correctness, 
    n_bootstrap=1000,
    confidence_level=0.95
)
print(f"ECE: {ece:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
```

### Model Comparison

When comparing models, we use:
- **Paired t-test** for mean differences
- **Wilcoxon signed-rank test** for non-parametric comparison
- **Effect size (Cohen's d)** for practical significance

## Metric Implementation

All metrics are implemented in `chimera.metrics`:

```python
from chimera.metrics.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
)
from chimera.metrics.error_detection import (
    compute_error_detection_metrics,
)
from chimera.metrics.knowledge_boundary import (
    compute_boundary_metrics,
)
from chimera.metrics.self_correction import (
    compute_correction_metrics,
)
```

## See Also

- [Calibration](calibration.md) - Deep dive into calibration
- [Introspection](introspection.md) - Theoretical foundations
- [Evaluation API](../api/evaluation.md) - Using metrics programmatically
