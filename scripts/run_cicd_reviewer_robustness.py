import os
import time
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — frozen from validated scripts
# ============================================================

RANDOM_STATE = 42
HISTORY_K = 20

PREPARED_FILE = os.path.join(
    "cicd_prepared",
    "cicd_clean_model_dataset.csv"
)
MASTER_FILE = "final_research_dataset_MASTER.csv"
OUTPUT_DIR = "cicd_reviewer_robustness"

TARGET = "target_failure"
GROUP = "commit_sha"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERIC_STATIC = [
    "msg_len",
    "num_parents",
    "additions",
    "deletions",
    "files_modified",
    "time_since_last_commit",
    "commit_to_pipeline_delay",
    "trigger_hour",
    "trigger_weekday",
]

BINARY_STATIC = [
    "is_merge_clean",
    "is_primary_branch",
    "trigger_weekend",
]

CATEGORICAL_STATIC = ["event"]

H0_HISTORY = [
    "repo_completed_failure_rate",
    "previous20_completed_failure_rate",
    "last_completed_outcome",
    "completed_history_count",
    "time_since_last_completed_failure",
]

H1_HISTORY = [
    "repo_completed_failure_rate",
    "previous20_completed_failure_rate",
    "last_completed_outcome",
    "time_since_last_completed_failure",
]

H2_HISTORY = [
    "repo_completed_failure_rate",
    "previous20_completed_failure_rate",
    "last_completed_outcome",
]

E2_SPECS = {
    "static": [],
    "H0_all_history": H0_HISTORY,
    "H1_no_history_count": H1_HISTORY,
    "H2_no_count_no_time_since_failure": H2_HISTORY,
}

ROLLING_SPECS = {
    "static": [],
    "H1_no_history_count": H1_HISTORY,
    "H2_no_count_no_time_since_failure": H2_HISTORY,
}

MODEL_NAMES = ["LR", "RF", "XGB", "LGBM"]
ROLLING_MODELS = ["RF", "LGBM"]

WINDOWS = [
    ("W1", 0.50, 0.60),
    ("W2", 0.60, 0.70),
    ("W3", 0.70, 0.80),
    ("W4", 0.80, 1.00),
]


# ============================================================
# HELPERS
# ============================================================

def safe_pr_auc(y, score):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, score)


def safe_roc_auc(y, score):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score)


def evaluate(y, score):
    y = np.asarray(y)
    score = np.asarray(score)

    prevalence = float(np.mean(y))
    pr = safe_pr_auc(y, score)
    roc = safe_roc_auc(y, score)

    return {
        "n": int(len(y)),
        "failures": int(np.sum(y)),
        "prevalence": prevalence,
        "pr_auc": float(pr) if not np.isnan(pr) else np.nan,
        "pr_auc_lift": (
            float(pr / prevalence)
            if prevalence > 0 and not np.isnan(pr)
            else np.nan
        ),
        "roc_auc": float(roc) if not np.isnan(roc) else np.nan,
    }


def build_preprocessor(numeric_features):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "bin",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent")
                        )
                    ]
                ),
                BINARY_STATIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent")
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            )
                        ),
                    ]
                ),
                CATEGORICAL_STATIC,
            ),
        ]
    )


def make_model(name, y_train):
    y_train = np.asarray(y_train)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos

    if name == "LR":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )

    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        )

    if name == "XGB":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "LGBM":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(name)


def add_history_features(train, test):
    """
    Validated completion-aware history semantics:
      prediction time = created_at
      outcome observable iff updated_at < current created_at
      same-commit historical executions excluded
      fixed cold-start global fallback = 0.5 when no completed history exists
    """
    timeline = pd.concat(
        [
            train.assign(_period="train"),
            test.assign(_period="test"),
        ],
        ignore_index=True,
    )

    chrono = (
        timeline
        .sort_values(
            ["created_at", "_original_order"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    completions = (
        timeline[timeline["updated_at"].notna()]
        .sort_values(
            ["updated_at", "_original_order"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    observable = defaultdict(list)
    global_failures = 0
    global_count = 0
    j = 0
    records = []

    n_groups = chrono["created_at"].nunique()
    group_counter = 0

    for current_time, batch in chrono.groupby("created_at", sort=True):
        group_counter += 1

        while (
            j < len(completions)
            and completions.iloc[j]["updated_at"] < current_time
        ):
            r = completions.iloc[j]

            observable[r["repo"]].append(
                {
                    "commit_sha": r[GROUP],
                    "updated_at": r["updated_at"],
                    "outcome": int(r[TARGET]),
                }
            )

            global_failures += int(r[TARGET])
            global_count += 1
            j += 1

        global_rate = (
            global_failures / global_count
            if global_count > 0
            else 0.5
        )

        for _, row in batch.iterrows():
            history = observable.get(row["repo"], [])

            eligible = [
                x for x in history
                if x["commit_sha"] != row[GROUP]
            ]

            if eligible:
                repo_rate = float(
                    np.mean([x["outcome"] for x in eligible])
                )
                last20 = eligible[-HISTORY_K:]
                prev20 = float(
                    np.mean([x["outcome"] for x in last20])
                )
                last_outcome = float(last20[-1]["outcome"])
            else:
                repo_rate = global_rate
                prev20 = global_rate
                last_outcome = np.nan

            last_failure = None
            for x in reversed(eligible):
                if x["outcome"] == 1:
                    last_failure = x["updated_at"]
                    break

            if last_failure is None:
                since_failure = np.nan
            else:
                since_failure = (
                    current_time - last_failure
                ).total_seconds()
                assert since_failure > 0

            records.append(
                {
                    "_original_order": row["_original_order"],
                    "repo_completed_failure_rate": repo_rate,
                    "previous20_completed_failure_rate": prev20,
                    "last_completed_outcome": last_outcome,
                    "completed_history_count": len(eligible),
                    "time_since_last_completed_failure": since_failure,
                }
            )

        if group_counter % 20000 == 0 or group_counter == n_groups:
            print(
                f"    history trigger groups: "
                f"{group_counter:,}/{n_groups:,}"
            )

    h = pd.DataFrame(records)

    assert h["_original_order"].nunique() == len(timeline)

    timeline = timeline.merge(
        h,
        on="_original_order",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # Preserve validated H0 transformation even when H1/H2 later omit it.
    timeline["completed_history_count"] = np.log1p(
        timeline["completed_history_count"].astype(float)
    )

    train_h = (
        timeline[timeline["_period"] == "train"]
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    test_h = (
        timeline[timeline["_period"] == "test"]
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    return train_h, test_h


def train_predict(train_h, test_h, history_features, model_name):
    numeric = NUMERIC_STATIC + history_features
    feature_cols = (
        numeric
        + BINARY_STATIC
        + CATEGORICAL_STATIC
    )

    X_train = train_h[feature_cols].copy()
    X_test = test_h[feature_cols].copy()

    y_train = train_h[TARGET].astype(int).to_numpy()
    y_test = test_h[TARGET].astype(int).to_numpy()

    pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric)),
            ("model", make_model(model_name, y_train)),
        ]
    )

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    training_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    score = pipe.predict_proba(X_test)[:, 1]
    inference_seconds = time.perf_counter() - t0

    return score, training_seconds, inference_seconds


def per_repo_metrics(test_h, score, specification, model_name):
    p = pd.DataFrame(
        {
            "repo": test_h["repo"].values,
            "commit_sha": test_h[GROUP].values,
            "y_true": test_h[TARGET].astype(int).values,
            "score": score,
        }
    )

    rows = []

    for repo, g in p.groupby("repo"):
        y = g["y_true"].to_numpy()

        # Require both classes for PR-AUC/ROC comparison.
        if y.sum() == 0 or y.sum() == len(y):
            continue

        m = evaluate(y, g["score"].to_numpy())

        rows.append(
            {
                "repo": repo,
                "specification": specification,
                "model": model_name,
                **m,
            }
        )

    return pd.DataFrame(rows)


def macro_summary(per_repo_df):
    rows = []

    for (spec, model_name), g in per_repo_df.groupby(
        ["specification", "model"]
    ):
        rows.append(
            {
                "specification": spec,
                "model": model_name,
                "repositories_evaluated": int(g["repo"].nunique()),
                "macro_mean_pr_auc": float(g["pr_auc"].mean()),
                "macro_median_pr_auc": float(g["pr_auc"].median()),
                "macro_mean_lift": float(g["pr_auc_lift"].mean()),
                "macro_median_lift": float(g["pr_auc_lift"].median()),
                "macro_mean_roc_auc": float(g["roc_auc"].mean()),
                "macro_median_roc_auc": float(g["roc_auc"].median()),
                "repos_beating_prevalence": int(
                    (g["pr_auc_lift"] > 1.0).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def compare_augmented_to_static(per_repo_df):
    """
    Same-repository comparison requested by reviewer.
    PR-AUC and lift comparisons are equivalent within each repository
    because both specifications share the same repository prevalence,
    but both are reported explicitly.
    """
    rows = []

    for model_name in MODEL_NAMES:
        static = per_repo_df[
            (per_repo_df["specification"] == "static")
            & (per_repo_df["model"] == model_name)
        ][
            ["repo", "pr_auc", "pr_auc_lift", "roc_auc"]
        ].rename(
            columns={
                "pr_auc": "static_pr_auc",
                "pr_auc_lift": "static_lift",
                "roc_auc": "static_roc_auc",
            }
        )

        for spec in [
            "H0_all_history",
            "H1_no_history_count",
            "H2_no_count_no_time_since_failure",
        ]:
            aug = per_repo_df[
                (per_repo_df["specification"] == spec)
                & (per_repo_df["model"] == model_name)
            ][
                ["repo", "pr_auc", "pr_auc_lift", "roc_auc"]
            ].rename(
                columns={
                    "pr_auc": "aug_pr_auc",
                    "pr_auc_lift": "aug_lift",
                    "roc_auc": "aug_roc_auc",
                }
            )

            z = static.merge(
                aug,
                on="repo",
                how="inner",
                validate="one_to_one",
            )

            if z.empty:
                continue

            z["delta_pr_auc"] = (
                z["aug_pr_auc"] - z["static_pr_auc"]
            )
            z["delta_lift"] = (
                z["aug_lift"] - z["static_lift"]
            )
            z["delta_roc_auc"] = (
                z["aug_roc_auc"] - z["static_roc_auc"]
            )

            rows.append(
                {
                    "model": model_name,
                    "augmented_specification": spec,
                    "repositories_compared": len(z),
                    "repos_aug_pr_auc_gt_static": int(
                        (z["delta_pr_auc"] > 0).sum()
                    ),
                    "repos_aug_pr_auc_eq_static": int(
                        np.isclose(
                            z["delta_pr_auc"],
                            0.0,
                            atol=1e-12,
                        ).sum()
                    ),
                    "median_static_pr_auc": float(
                        z["static_pr_auc"].median()
                    ),
                    "median_aug_pr_auc": float(
                        z["aug_pr_auc"].median()
                    ),
                    "median_delta_pr_auc": float(
                        z["delta_pr_auc"].median()
                    ),
                    "median_static_lift": float(
                        z["static_lift"].median()
                    ),
                    "median_aug_lift": float(
                        z["aug_lift"].median()
                    ),
                    "median_delta_lift": float(
                        z["delta_lift"].median()
                    ),
                    "median_static_roc_auc": float(
                        z["static_roc_auc"].median()
                    ),
                    "median_aug_roc_auc": float(
                        z["aug_roc_auc"].median()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# LOAD PREPARED + MASTER
# ============================================================

print("=" * 100)
print("REVIEWER ROBUSTNESS ANALYSIS")
print("=" * 100)

print("\nLoading prepared dataset...")

df = pd.read_csv(PREPARED_FILE, low_memory=False)
df["_original_order"] = np.arange(len(df))

print(f"Prepared rows: {len(df):,}")

print("\nLoading master fields...")

# Read enough master columns for updated_at + retry semantics audit.
master = pd.read_csv(
    MASTER_FILE,
    usecols=[
        "run_id",
        "updated_at",
        "run_attempt",
        "conclusion",
        "status",
    ],
    low_memory=False,
)

# ============================================================
# RETRY SEMANTICS AUDIT — before deduplication
# ============================================================

print("\n" + "=" * 100)
print("RETRY SEMANTICS AUDIT")
print("=" * 100)

retry_raw = master.copy()

retry_raw["run_attempt_num"] = pd.to_numeric(
    retry_raw["run_attempt"],
    errors="coerce",
)

retry_raw["conclusion_norm"] = (
    retry_raw["conclusion"]
    .astype("string")
    .str.strip()
    .str.lower()
)

retry_raw["binary_failure"] = np.where(
    retry_raw["conclusion_norm"].isin(
        ["failure", "startup_failure"]
    ),
    1.0,
    np.where(
        retry_raw["conclusion_norm"].eq("success"),
        0.0,
        np.nan,
    ),
)

total_rows_master = len(retry_raw)
unique_run_ids = retry_raw["run_id"].nunique(dropna=True)
duplicate_rows_by_run_id = int(
    retry_raw["run_id"].duplicated(keep=False).sum()
)
duplicate_run_ids = int(
    retry_raw.loc[
        retry_raw["run_id"].duplicated(keep=False),
        "run_id"
    ].nunique()
)

attempt_counts = (
    retry_raw["run_attempt_num"]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis("run_attempt")
    .reset_index(name="rows")
)

attempt_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "retry_attempt_distribution.csv"
    ),
    index=False,
)

run_attempt_sets = (
    retry_raw.dropna(subset=["run_id"])
    .groupby("run_id")["run_attempt_num"]
    .agg(lambda x: tuple(sorted(set(x.dropna().astype(int)))))
)

run_ids_with_attempt1_and_gt1 = int(
    run_attempt_sets.apply(
        lambda s: 1 in s and any(v > 1 for v in s)
    ).sum()
)

run_ids_with_multiple_attempt_values = int(
    run_attempt_sets.apply(lambda s: len(s) > 1).sum()
)

retry_group_rows = []

for label, mask in [
    ("run_attempt_eq_1", retry_raw["run_attempt_num"] == 1),
    ("run_attempt_gt_1", retry_raw["run_attempt_num"] > 1),
]:
    g = retry_raw.loc[mask].copy()
    binary = g["binary_failure"].dropna()

    retry_group_rows.append(
        {
            "group": label,
            "rows": len(g),
            "unique_run_ids": g["run_id"].nunique(dropna=True),
            "binary_target_rows": len(binary),
            "binary_failures": int(binary.sum()) if len(binary) else 0,
            "binary_failure_prevalence": (
                float(binary.mean()) if len(binary) else np.nan
            ),
            "success": int(
                (g["conclusion_norm"] == "success").sum()
            ),
            "failure": int(
                (g["conclusion_norm"] == "failure").sum()
            ),
            "startup_failure": int(
                (g["conclusion_norm"] == "startup_failure").sum()
            ),
            "cancelled": int(
                (g["conclusion_norm"] == "cancelled").sum()
            ),
            "skipped": int(
                (g["conclusion_norm"] == "skipped").sum()
            ),
        }
    )

retry_prevalence = pd.DataFrame(retry_group_rows)

retry_prevalence.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "retry_prevalence_audit.csv"
    ),
    index=False,
)

retry_summary = {
    "master_rows_read": int(total_rows_master),
    "unique_run_ids": int(unique_run_ids),
    "rows_belonging_to_duplicated_run_ids": duplicate_rows_by_run_id,
    "duplicated_run_ids": duplicate_run_ids,
    "run_ids_with_multiple_attempt_values": (
        run_ids_with_multiple_attempt_values
    ),
    "run_ids_with_attempt1_and_gt1": (
        run_ids_with_attempt1_and_gt1
    ),
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "retry_semantics_summary.json"
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(retry_summary, f, indent=2)

print(json.dumps(retry_summary, indent=2))
print("\nAttempt distribution:")
print(attempt_counts.to_string(index=False))
print("\nBinary-target prevalence by attempt group:")
print(retry_prevalence.to_string(index=False))

# ============================================================
# MERGE updated_at INTO PREPARED DATA
# ============================================================

master_updated = (
    master[
        ["run_id", "updated_at"]
    ]
    .dropna(subset=["run_id"])
    .drop_duplicates(
        subset=["run_id"],
        keep="last",
    )
)

n_before = len(df)

df = df.merge(
    master_updated,
    on="run_id",
    how="left",
    validate="many_to_one",
    sort=False,
)

assert len(df) == n_before

df = (
    df
    .sort_values("_original_order")
    .reset_index(drop=True)
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    utc=True,
    errors="coerce",
)

df["updated_at"] = pd.to_datetime(
    df["updated_at"],
    utc=True,
    errors="coerce",
)

print(
    f"\nMissing updated_at after merge: "
    f"{df['updated_at'].isna().sum():,}"
)

# ============================================================
# CORRECTED FIRST-ATTEMPT COHORT
# ============================================================

df = df[df["run_attempt"] == 1].copy()

df = (
    df
    .sort_values("_original_order")
    .reset_index(drop=True)
)

print("\nCorrected first-attempt cohort:")
print(f"Rows: {len(df):,}")
print(f"Repositories: {df['repo'].nunique():,}")
print(f"Commits: {df[GROUP].nunique():,}")
print(f"Failures: {int(df[TARGET].sum()):,}")
print(f"Failure rate: {df[TARGET].mean():.6f}")

assert len(df) == 142629
assert df["repo"].nunique() == 27
assert df[GROUP].nunique() == 26555
assert int(df[TARGET].sum()) == 8378

# ============================================================
# E2 — exact validated reconstruction
# ============================================================

print("\n" + "=" * 100)
print("E2 RECONSTRUCTION")
print("=" * 100)

commit_first = (
    df.groupby(
        GROUP,
        as_index=False,
    )["created_at"]
    .min()
    .sort_values(
        ["created_at", GROUP]
    )
    .reset_index(drop=True)
)

split_idx = int(
    np.floor(
        0.80 * len(commit_first)
    )
)

provisional_test_commits = (
    commit_first.iloc[split_idx:][GROUP]
)

boundary = df.loc[
    df[GROUP].isin(provisional_test_commits),
    "created_at",
].min()

commit_ranges = (
    df.groupby(GROUP)["created_at"]
    .agg(["min", "max"])
)

spanning_commits = commit_ranges[
    (commit_ranges["min"] < boundary)
    & (commit_ranges["max"] >= boundary)
].index

strict_df = df[
    ~df[GROUP].isin(spanning_commits)
].copy()

strict_df = (
    strict_df
    .sort_values("_original_order")
    .reset_index(drop=True)
)

train = strict_df[
    strict_df["created_at"] < boundary
].copy()

test = strict_df[
    strict_df["created_at"] >= boundary
].copy()

print(f"Boundary: {boundary}")
print(f"Boundary-spanning commits: {len(spanning_commits):,}")
print(
    f"Train: {len(train):,} rows | "
    f"{train[GROUP].nunique():,} commits | "
    f"{int(train[TARGET].sum()):,} failures"
)
print(
    f"Test: {len(test):,} rows | "
    f"{test[GROUP].nunique():,} commits | "
    f"{int(test[TARGET].sum()):,} failures | "
    f"{test[TARGET].mean():.6%}"
)

EXPECTED_BOUNDARY = pd.Timestamp(
    "2026-04-01 11:41:59",
    tz="UTC",
)

assert boundary == EXPECTED_BOUNDARY
assert len(spanning_commits) == 25
assert len(train) == 105814
assert train[GROUP].nunique() == 21219
assert int(train[TARGET].sum()) == 6683
assert len(test) == 36036
assert test[GROUP].nunique() == 5311
assert int(test[TARGET].sum()) == 1610
assert abs(test[TARGET].mean() - 0.04467865467865468) < 1e-12
assert len(set(train[GROUP]) & set(test[GROUP])) == 0

print("\nE2 integrity checks PASSED.")

# ============================================================
# E2 HISTORY FEATURES
# ============================================================

print("\nGenerating validated completion-aware E2 history...")
train_h, test_h = add_history_features(train, test)

assert len(train_h) == 105814
assert len(test_h) == 36036
assert int(test_h[TARGET].sum()) == 1610

print("E2 history generation complete.")

# ============================================================
# E2 H0/H1/H2 + STATIC
# ============================================================

print("\n" + "=" * 100)
print("E2 PROXY ABLATION")
print("=" * 100)

e2_pooled_rows = []
e2_per_repo_frames = []
e2_prediction_frames = []

y_test = test_h[TARGET].astype(int).to_numpy()

for spec_name, history_features in E2_SPECS.items():
    print(f"\n--- {spec_name} ---")

    for model_name in MODEL_NAMES:
        print(f"Training {model_name}...")

        score, train_sec, infer_sec = train_predict(
            train_h,
            test_h,
            history_features,
            model_name,
        )

        m = evaluate(y_test, score)

        e2_pooled_rows.append(
            {
                "specification": spec_name,
                "model": model_name,
                **m,
                "training_seconds": train_sec,
                "inference_seconds": infer_sec,
            }
        )

        print(
            f"{model_name}: "
            f"PR-AUC={m['pr_auc']:.6f} | "
            f"Lift={m['pr_auc_lift']:.3f}x | "
            f"ROC-AUC={m['roc_auc']:.6f}"
        )

        pr = per_repo_metrics(
            test_h,
            score,
            spec_name,
            model_name,
        )
        e2_per_repo_frames.append(pr)

        e2_prediction_frames.append(
            pd.DataFrame(
                {
                    "_original_order": test_h["_original_order"].values,
                    "run_id": test_h["run_id"].values,
                    "repo": test_h["repo"].values,
                    "commit_sha": test_h[GROUP].values,
                    "y_true": y_test,
                    "specification": spec_name,
                    "model": model_name,
                    "score": score,
                }
            )
        )

e2_pooled = pd.DataFrame(e2_pooled_rows)
e2_per_repo = pd.concat(
    e2_per_repo_frames,
    ignore_index=True,
)
e2_predictions = pd.concat(
    e2_prediction_frames,
    ignore_index=True,
)

e2_macro = macro_summary(e2_per_repo)
e2_static_vs_aug = compare_augmented_to_static(
    e2_per_repo
)

e2_pooled.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_proxy_ablation_pooled.csv"
    ),
    index=False,
)

e2_per_repo.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_proxy_ablation_per_repo.csv"
    ),
    index=False,
)

e2_macro.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_proxy_ablation_macro.csv"
    ),
    index=False,
)

e2_static_vs_aug.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_static_vs_augmented_within_repo.csv"
    ),
    index=False,
)

e2_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_proxy_ablation_predictions.csv"
    ),
    index=False,
)

print("\nE2 pooled:")
print(
    e2_pooled[
        [
            "specification",
            "model",
            "pr_auc",
            "pr_auc_lift",
            "roc_auc",
        ]
    ].to_string(index=False)
)

print("\nE2 macro:")
print(
    e2_macro[
        [
            "specification",
            "model",
            "repositories_evaluated",
            "macro_median_pr_auc",
            "macro_median_lift",
            "macro_median_roc_auc",
            "repos_beating_prevalence",
        ]
    ]
    .sort_values(
        ["model", "specification"]
    )
    .to_string(index=False)
)

print("\nStatic vs augmented — same repositories:")
print(
    e2_static_vs_aug.to_string(index=False)
)

# ============================================================
# ROLLING H1/H2 + STATIC — validated window definitions
# ============================================================

print("\n" + "=" * 100)
print("ROLLING-ORIGIN PROXY ROBUSTNESS")
print("=" * 100)

n_commits = len(commit_first)

rolling_rows = []
rolling_per_repo_frames = []
rolling_window_rows = []

for (
    window_name,
    train_fraction,
    test_end_fraction,
) in WINDOWS:

    print("\n" + "-" * 100)
    print(
        f"{window_name}: "
        f"train first {train_fraction:.0%}, "
        f"test until {test_end_fraction:.0%}"
    )

    train_cut_idx = int(
        np.floor(
            train_fraction * n_commits
        )
    )

    if test_end_fraction < 1.0:
        end_cut_idx = int(
            np.floor(
                test_end_fraction * n_commits
            )
        )
        test_end = (
            commit_first.iloc[end_cut_idx]["created_at"]
        )
    else:
        test_end = None

    test_start = (
        commit_first.iloc[train_cut_idx]["created_at"]
    )

    ranges = (
        df.groupby(GROUP)["created_at"]
        .agg(["min", "max"])
    )

    spanning_start = ranges[
        (ranges["min"] < test_start)
        & (ranges["max"] >= test_start)
    ].index

    working = df[
        ~df[GROUP].isin(spanning_start)
    ].copy()

    if test_end is not None:
        spanning_end = ranges[
            (ranges["min"] < test_end)
            & (ranges["max"] >= test_end)
        ].index

        working = working[
            ~working[GROUP].isin(spanning_end)
        ].copy()
    else:
        spanning_end = []

    train_w = working[
        working["created_at"] < test_start
    ].copy()

    if test_end is None:
        test_w = working[
            working["created_at"] >= test_start
        ].copy()
    else:
        test_w = working[
            (working["created_at"] >= test_start)
            & (working["created_at"] < test_end)
        ].copy()

    train_w = (
        train_w
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    test_w = (
        test_w
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    assert len(
        set(train_w[GROUP]) & set(test_w[GROUP])
    ) == 0

    # W4 must reconstruct E2 exactly.
    if window_name == "W4":
        assert test_start == EXPECTED_BOUNDARY
        assert len(train_w) == 105814
        assert len(test_w) == 36036
        assert test_w[GROUP].nunique() == 5311
        assert int(test_w[TARGET].sum()) == 1610

    print(
        f"Train: {len(train_w):,} | "
        f"{train_w[GROUP].nunique():,} commits"
    )
    print(
        f"Test: {len(test_w):,} | "
        f"{test_w[GROUP].nunique():,} commits | "
        f"{int(test_w[TARGET].sum()):,} failures | "
        f"{test_w[TARGET].mean():.4%}"
    )

    rolling_window_rows.append(
        {
            "window": window_name,
            "train_fraction": train_fraction,
            "test_end_fraction": test_end_fraction,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": len(train_w),
            "test_rows": len(test_w),
            "train_commits": train_w[GROUP].nunique(),
            "test_commits": test_w[GROUP].nunique(),
            "train_failures": int(train_w[TARGET].sum()),
            "test_failures": int(test_w[TARGET].sum()),
            "test_prevalence": test_w[TARGET].mean(),
            "start_spanning_commits": len(spanning_start),
            "end_spanning_commits": len(spanning_end),
        }
    )

    print("Generating completion-aware history...")
    train_wh, test_wh = add_history_features(
        train_w,
        test_w,
    )

    y_w = test_wh[TARGET].astype(int).to_numpy()

    for spec_name, history_features in ROLLING_SPECS.items():
        for model_name in ROLLING_MODELS:
            print(
                f"  {window_name} / "
                f"{spec_name} / {model_name}"
            )

            score, train_sec, infer_sec = train_predict(
                train_wh,
                test_wh,
                history_features,
                model_name,
            )

            m = evaluate(y_w, score)

            rolling_rows.append(
                {
                    "window": window_name,
                    "specification": spec_name,
                    "model": model_name,
                    **m,
                    "training_seconds": train_sec,
                    "inference_seconds": infer_sec,
                }
            )

            pr = per_repo_metrics(
                test_wh,
                score,
                spec_name,
                model_name,
            )
            pr.insert(0, "window", window_name)
            rolling_per_repo_frames.append(pr)

rolling_pooled = pd.DataFrame(rolling_rows)
rolling_per_repo = pd.concat(
    rolling_per_repo_frames,
    ignore_index=True,
)
rolling_windows = pd.DataFrame(
    rolling_window_rows
)

rolling_macro_rows = []

for (
    window_name,
    spec,
    model_name,
), g in rolling_per_repo.groupby(
    ["window", "specification", "model"]
):
    rolling_macro_rows.append(
        {
            "window": window_name,
            "specification": spec,
            "model": model_name,
            "repos_evaluated": int(g["repo"].nunique()),
            "macro_mean_pr_auc": float(g["pr_auc"].mean()),
            "macro_median_pr_auc": float(g["pr_auc"].median()),
            "macro_mean_lift": float(g["pr_auc_lift"].mean()),
            "macro_median_lift": float(g["pr_auc_lift"].median()),
            "macro_mean_roc_auc": float(g["roc_auc"].mean()),
            "macro_median_roc_auc": float(g["roc_auc"].median()),
            "repos_beating_prevalence": int(
                (g["pr_auc_lift"] > 1.0).sum()
            ),
        }
    )

rolling_macro = pd.DataFrame(
    rolling_macro_rows
)

rolling_pooled.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_proxy_ablation_pooled.csv"
    ),
    index=False,
)

rolling_per_repo.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_proxy_ablation_per_repo.csv"
    ),
    index=False,
)

rolling_macro.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_proxy_ablation_macro.csv"
    ),
    index=False,
)

rolling_windows.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_proxy_window_summary.csv"
    ),
    index=False,
)

print("\nROLLING POOLED RESULTS")
print(
    rolling_pooled[
        [
            "window",
            "specification",
            "model",
            "prevalence",
            "pr_auc",
            "pr_auc_lift",
            "roc_auc",
        ]
    ].to_string(index=False)
)

print("\nROLLING MACRO RESULTS")
print(
    rolling_macro[
        [
            "window",
            "specification",
            "model",
            "repos_evaluated",
            "macro_median_pr_auc",
            "macro_median_lift",
            "repos_beating_prevalence",
        ]
    ].to_string(index=False)
)

# ============================================================
# METADATA
# ============================================================

metadata = {
    "analysis": "reviewer_proxy_robustness",
    "random_state": RANDOM_STATE,
    "history_k": HISTORY_K,
    "prediction_time": "created_at",
    "history_availability": "updated_at < created_at",
    "same_commit_history": "excluded",
    "cold_start_fallback": 0.5,
    "first_attempt_only": True,
    "raw_repo_feature": False,
    "run_number_feature": False,
    "run_attempt_feature": False,
    "H0_history_features": H0_HISTORY,
    "H1_history_features": H1_HISTORY,
    "H2_history_features": H2_HISTORY,
    "e2_boundary": str(boundary),
    "e2_train_rows": int(len(train_h)),
    "e2_test_rows": int(len(test_h)),
    "e2_test_failures": int(test_h[TARGET].sum()),
    "rolling_windows": WINDOWS,
    "rolling_note": (
        "Nested training sets; W4 is the E2 holdout. "
        "Rolling comparisons are descriptive robustness checks, "
        "not independent confirmations."
    ),
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "reviewer_robustness_metadata.json"
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        default=str,
    )

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)
print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}")
