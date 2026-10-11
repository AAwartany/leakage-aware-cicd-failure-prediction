# Canonical E2 Static Baseline Validation

This folder contains the reproducibility artifacts for the canonical E2 static baselines in the CI/CD failure prediction study.

## Models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Evaluation
- Strict temporal E2 split
- Training executions: 105,814
- Test executions: 36,036
- Test commits: 5,311
- Test failures: 1,610
- Boundary-spanning commits excluded: 25

## Reproducibility
The validation script reproduces the canonical leakage-ladder L4 static baseline results.

The manifest records dataset fingerprints, feature specifications, split identifiers, and software versions.

The prediction CSV files preserve run-level identifiers for further verification.

## Research Archive
DOI: https://doi.org/10.5281/zenodo.23237520
