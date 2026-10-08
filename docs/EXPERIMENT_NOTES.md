# Experiment provenance and limitations

- **Static vs history:** H1 and H2 use completion-aware historical features. Historical completion must precede the current execution trigger and same-commit records are excluded.
- **Leakage ladder:** L1–L3 use the same commit-group split *within the ladder*. L3 reports RF PR-AUC 0.203180 and XGB 0.197398. A separately executed E1 robustness analysis reports approximately RF 0.205764 and XGB 0.201018. These are distinct saved experimental outputs, not rounding variations. The precise cause of the discrepancy is **not yet established**. Do not claim that the separately executed runs used exactly identical instantiated partitions, pipelines, and model states.
- **E2:** The chronological test period has 36,036 executions / 5,311 commits / 1,610 failures. H1 LightGBM pooled PR-AUC is approximately 0.236608. Within-repository summaries should be reported separately from pooled results.
- **Bootstrap:** 5,000 paired commit-cluster resamples; eight planned contrasts, Holm correction. Source results are in `h1_e2_paired_bootstrap.csv`.
- **SHAP:** H1 explanations are descriptive, not causal.
- **E3 and cancellation:** E3 results concern static models; cancellation sensitivity belongs to the earlier H0 specification, not a direct H1 cancellation experiment.
- **Retry interpretation:** The source audit found no run ID with multiple observed attempt values; first-attempt filtering removes later-attempt run rows rather than reconstructing multiple attempts from a single run ID.
- **Reproducibility limitation:** The original raw-to-prepared dataset creation code and a clean-room end-to-end rerun are not included. Scripts were retained substantially as originally executed and may require local path adjustments. Do not state that these materials guarantee exact independent replication.
