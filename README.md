# Leakage-aware CI/CD failure prediction — research artifacts

Supporting research artifacts for **How Well Do CI/CD Failure Predictors Generalize? A Leakage-Aware Temporal, Historical, and Cross-Project Study of GitHub Actions**.

## Status and scope

This is a **reviewer-facing artifact bundle**, not yet a verified one-command reproduction environment. It includes original analysis scripts and saved experiment outputs. The code has passed Python syntax compilation but **has not been end-to-end rerun in this packaging environment**. Some analyses require the original dataset and specific working-directory layout. Do not claim exact reproducibility until the workflow is tested.

## Source data

The study uses the publicly described GitHub Actions research dataset at DOI **10.17632/mggwn7rj9f.1**. Obtain and check the dataset's terms directly from the source. **No dataset records are redistributed here.** The analysis scripts expect the prepared CSV at `cicd_prepared/cicd_clean_model_dataset.csv` and the source master CSV at `final_research_dataset_MASTER.csv` (relative to the execution directory). Preparation of the derived dataset is not fully automated in this release, so reproducing the entire pipeline from the original dataset requires additional preprocessing documentation/code.

## Contents

- `scripts/`: original experimental scripts, including H1/H2 robustness, history models, rolling validation, E3, SHAP, and H1 bootstrap.
- `results/`: frozen experiment output ZIPs, including within-repository tables, leakage ladder, paired predictions, SHAP and rolling results.
- `docs/EXPERIMENT_NOTES.md`: design caveats, result provenance and scope.
- `requirements.txt`: package versions reported for the main experiment environment; SHAP/matplotlib are intentionally unpinned pending verification.

## Key protocol

First workflow attempt only (`run_attempt == 1`), 27 eligible repositories, 142,629 execution rows, 26,555 commits. E2 strictly chronological evaluation excludes boundary-spanning commits; E2 test includes 36,036 rows, 5,311 commits and 1,610 failures. Completion-aware historical features require a completed execution strictly before the current trigger time and exclude the same commit. H1 removes the `completed_history_count` progression proxy; H2 also removes `time_since_last_completed_failure`.

## Running an analysis (advanced)

Install Python 3.13 and dependencies in an isolated environment. Place the required source and prepared data at the paths above, then execute an analysis script **from the working directory containing those input paths**. Scripts may take substantial time and some write to fixed output subdirectories. The paired-bootstrap script expects the named results ZIPs in its parent directory, and should be run only after adapting its path assumptions. Results ZIPs are provided for independent verification of reported metrics without rerunning all training.

## Citation

Cite the associated manuscript and original dataset. A Zenodo DOI for these artifacts will be added after a validated public release; no DOI is claimed here.

## Licensing

No license has been granted for this bundle pending a decision by the author and verification of upstream data and third-party terms. Do not assume permission to redistribute original dataset content.
