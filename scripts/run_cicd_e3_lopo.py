from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

INPUT = Path(
    "cicd_prepared/cicd_clean_model_dataset.csv"
)

OUT = Path("cicd_e3_results")
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "target_failure"
PROJECT = "repo"


# ============================================================
# Load dataset
# ============================================================

print("Loading:", INPUT)

df = pd.read_csv(
    INPUT,
    low_memory=False
)

print("Shape:", df.shape)

projects = sorted(
    df[PROJECT]
    .dropna()
    .unique()
)

print(
    "Projects:",
    len(projects)
)

assert len(projects) == 27, (
    f"Expected 27 projects, found {len(projects)}"
)


# ============================================================
# Feature policy — IDENTICAL TO E1/E2
# ============================================================

numeric_features = [
    "run_number",
    "msg_len",
    "num_parents",
    "additions",
    "deletions",
    "files_modified",
    "run_attempt",
    "time_since_last_commit",
    "commit_to_pipeline_delay",
    "trigger_hour",
    "trigger_weekday",
]

binary_features = [
    "is_merge_clean",
    "is_primary_branch",
    "trigger_weekend",
]

categorical_features = [
    "event",
]

feature_cols = (
    numeric_features
    + binary_features
    + categorical_features
)


# ============================================================
# Helper: create preprocessing pipeline
# ============================================================

def make_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    binary_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        ),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "binary",
            binary_pipeline,
            binary_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ])


# ============================================================
# Helper: create models
# ============================================================

def make_models(scale_pos_weight):

    return {
        "Logistic Regression":
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
                solver="liblinear",
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=2,
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "LightGBM":
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
    }


# ============================================================
# LOPO evaluation
# ============================================================

all_results = []

for project_number, test_project in enumerate(
    projects,
    start=1
):

    print("\n")
    print("=" * 90)
    print(
        f"PROJECT {project_number}/{len(projects)}:"
        f" {test_project}"
    )
    print("=" * 90)

    train_df = df[
        df[PROJECT] != test_project
    ].copy()

    test_df = df[
        df[PROJECT] == test_project
    ].copy()

    # ----------------------------------------
    # Explicit project leakage check
    # ----------------------------------------

    train_projects = set(
        train_df[PROJECT].unique()
    )

    assert (
        test_project not in train_projects
    ), (
        "ERROR: test project appears "
        "in training data."
    )

    print(
        "Project leakage check: PASSED"
    )

    X_train = train_df[
        feature_cols
    ].copy()

    X_test = test_df[
        feature_cols
    ].copy()

    y_train = (
        train_df[TARGET]
        .astype(int)
    )

    y_test = (
        test_df[TARGET]
        .astype(int)
    )

    failures = int(
        y_test.sum()
    )

    successes = int(
        (y_test == 0).sum()
    )

    prevalence = float(
        y_test.mean()
    )

    print(
        "Test rows:",
        len(test_df)
    )

    print(
        "Failures:",
        failures
    )

    print(
        "Failure prevalence:",
        f"{prevalence:.4f}"
    )

    # Every eligible repository should
    # contain both classes.

    if y_test.nunique() < 2:

        print(
            "WARNING: only one class "
            "in test repository. Skipping."
        )

        continue

    # ----------------------------------------
    # Training class weighting
    # ----------------------------------------

    n_negative = int(
        (y_train == 0).sum()
    )

    n_positive = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        n_negative / n_positive
    )

    models = make_models(
        scale_pos_weight
    )

    # ----------------------------------------
    # Train all four models
    # ----------------------------------------

    for model_name, model in models.items():

        print(
            f"\n  {model_name}"
        )

        pipeline = Pipeline([
            (
                "preprocess",
                make_preprocessor()
            ),
            (
                "model",
                model
            ),
        ])

        # Training
        start = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train
        )

        training_time = (
            time.perf_counter()
            - start
        )

        # Inference
        start = time.perf_counter()

        y_prob = (
            pipeline.predict_proba(
                X_test
            )[:, 1]
        )

        inference_time = (
            time.perf_counter()
            - start
        )

        y_pred = (
            y_prob >= 0.5
        ).astype(int)

        # ----------------------------------------
        # Metrics
        # ----------------------------------------

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        ).ravel()

        # ----------------------------------------
        # Normalized PR-AUC lift
        # ----------------------------------------

        pr_auc_lift = (
            pr_auc / prevalence
            if prevalence > 0
            else np.nan
        )

        beats_baseline = bool(
            pr_auc > prevalence
        )

        all_results.append({
            "test_project":
                test_project,

            "model":
                model_name,

            "test_rows":
                len(test_df),

            "failures":
                failures,

            "successes":
                successes,

            "failure_prevalence":
                prevalence,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "pr_auc_lift":
                pr_auc_lift,

            "beats_prevalence_baseline":
                beats_baseline,

            "mcc":
                mcc,

            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),

            "training_time_sec":
                training_time,

            "inference_time_sec":
                inference_time,
        })

        print(
            f"    PR-AUC : {pr_auc:.4f}"
        )

        print(
            f"    Baseline: {prevalence:.4f}"
        )

        print(
            f"    Lift   : {pr_auc_lift:.2f}x"
        )

        print(
            f"    ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"    F1     : {f1:.4f}"
        )


# ============================================================
# Save detailed results
# ============================================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    OUT / "e3_lopo_detailed_results.csv",
    index=False
)


# ============================================================
# Aggregate results by model
# ============================================================

summary_rows = []

for model_name, group in results.groupby(
    "model"
):

    projects_tested = len(group)

    projects_beating = int(
        group[
            "beats_prevalence_baseline"
        ].sum()
    )

    summary_rows.append({

        "model":
            model_name,

        "projects_tested":
            projects_tested,

        "mean_pr_auc":
            group["pr_auc"].mean(),

        "median_pr_auc":
            group["pr_auc"].median(),

        "mean_pr_auc_lift":
            group["pr_auc_lift"].mean(),

        "median_pr_auc_lift":
            group["pr_auc_lift"].median(),

        "mean_roc_auc":
            group["roc_auc"].mean(),

        "median_roc_auc":
            group["roc_auc"].median(),

        "mean_precision":
            group["precision"].mean(),

        "median_precision":
            group["precision"].median(),

        "mean_recall":
            group["recall"].mean(),

        "median_recall":
            group["recall"].median(),

        "mean_f1":
            group["f1"].mean(),

        "median_f1":
            group["f1"].median(),

        "mean_mcc":
            group["mcc"].mean(),

        "median_mcc":
            group["mcc"].median(),

        "projects_beating_baseline":
            projects_beating,

        "pct_projects_beating_baseline":
            (
                100
                * projects_beating
                / projects_tested
            ),

        "mean_training_time_sec":
            group[
                "training_time_sec"
            ].mean(),

        "mean_inference_time_sec":
            group[
                "inference_time_sec"
            ].mean(),
    })


summary = pd.DataFrame(
    summary_rows
)

summary = summary.sort_values(
    [
        "median_pr_auc_lift",
        "median_pr_auc"
    ],
    ascending=False
)

summary.to_csv(
    OUT / "e3_lopo_model_summary.csv",
    index=False
)


# ============================================================
# Repository summary
# ============================================================

project_summary = (
    df.groupby(
        PROJECT
    )
    .agg(
        runs=(
            TARGET,
            "size"
        ),
        failures=(
            TARGET,
            "sum"
        ),
        unique_commits=(
            "commit_sha",
            "nunique"
        ),
    )
    .reset_index()
)

project_summary[
    "failure_prevalence"
] = (
    project_summary["failures"]
    / project_summary["runs"]
)

project_summary.to_csv(
    OUT / "e3_project_summary.csv",
    index=False
)


# ============================================================
# Best model per repository
# ============================================================

best_per_project = (
    results.sort_values(
        [
            "test_project",
            "pr_auc"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "test_project",
        as_index=False
    )
    .first()
)

best_per_project.to_csv(
    OUT / "e3_best_model_per_project.csv",
    index=False
)


# ============================================================
# Final console summary
# ============================================================

print("\n")
print("=" * 110)
print("E3 — LOPO FINAL SUMMARY")
print("=" * 110)

display_cols = [
    "model",
    "median_pr_auc",
    "median_pr_auc_lift",
    "median_roc_auc",
    "median_f1",
    "median_mcc",
    "projects_beating_baseline",
    "pct_projects_beating_baseline",
]

print(
    summary[
        display_cols
    ].to_string(
        index=False
    )
)

print("\nResults saved to:")
print(
    OUT.resolve()
)