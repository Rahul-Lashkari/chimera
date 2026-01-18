# Calibration

This document explains the concept of calibration and how CHIMERA measures it.

## What is Calibration?

**Calibration** measures whether a model's expressed confidence matches its actual accuracy. A well-calibrated model that says "I'm 80% confident" should be correct approximately 80% of the time.

## Why Calibration Matters

### Safety
Overconfident wrong answers are dangerous. A model that's 95% confident but only 50% accurate will mislead users.

### Trust
Users need to know when to trust model outputs. Calibrated confidence enables appropriate human-AI collaboration.

### Decision Making
Many downstream applications (medicine, law, finance) require knowing uncertainty to make good decisions.

## Expected Calibration Error (ECE)

ECE is the primary calibration metric in CHIMERA. It measures the average gap between stated confidence and actual accuracy:

$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$

Where:
- $B_m$ is the set of predictions in bin $m$
- $acc(B_m)$ is the accuracy of predictions in bin $m$
- $conf(B_m)$ is the average confidence in bin $m$

### Interpretation

| ECE Value | Interpretation |
|-----------|----------------|
| 0.00-0.05 | Excellent calibration |
| 0.05-0.10 | Good calibration |
| 0.10-0.20 | Moderate miscalibration |
| 0.20+ | Severe miscalibration |

## Reliability Diagrams

CHIMERA generates reliability diagrams that visualize calibration:

- **X-axis**: Confidence bins
- **Y-axis**: Actual accuracy
- **Diagonal line**: Perfect calibration
- **Gap**: Calibration error

## Common Failure Modes

### Overconfidence
Model expresses high confidence but accuracy is lower.

### Underconfidence
Model hedges excessively even when correct.

### Calibration Collapse
Well-calibrated on easy tasks, miscalibrated on hard ones.

## See Also

- [Metrics Reference](metrics.md)
- [Track 1 Implementation](../api/evaluation.md)
