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
# CONFIG
# ============================================================

RANDOM_STATE = 42
HISTORY_K = 20

PREPARED_FILE = os.path.join(
    "cicd_prepared",
    "cicd_clean_model_dataset.csv"
)

MASTER_FILE = "final_research_dataset_MASTER.csv"
OUTPUT_DIR = "cicd_history_augmented_ml_v2"

TARGET = "target_failure"

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

HISTORY_FEATURES = [
    "repo_completed_failure_rate",
    "previous20_completed_failure_rate",
    "last_completed_outcome",
    "completed_history_count",
    "time_since_last_completed_failure",
]

MODEL_NAMES = ["LR", "RF", "XGB", "LGBM"]


# ============================================================
# HELPERS
# ============================================================

def safe_pr_auc(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, score)


def safe_roc_auc(y, score):
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
        "pr_auc": float(pr),
        "pr_auc_lift": (
            float(pr / prevalence)
            if prevalence > 0
            else np.nan
        ),
        "roc_auc": float(roc),
    }


def build_preprocessor(numeric_features):

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median")
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        ),
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
                            SimpleImputer(
                                strategy="most_frequent"
                            )
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
                            SimpleImputer(
                                strategy="most_frequent"
                            )
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


# ============================================================
# LOAD PREPARED DATA
# ============================================================

print("=" * 100)
print("HISTORY-AUGMENTED ML V2 — STRICT E2")
print("=" * 100)

print("\nLoading prepared dataset...")

df = pd.read_csv(
    PREPARED_FILE,
    low_memory=False
)

# Critical: immutable original order
df["_original_order"] = np.arange(len(df))

print(f"Prepared rows: {len(df):,}")

# ============================================================
# MERGE updated_at
# ============================================================

print("\nLoading updated_at from master dataset...")

master = pd.read_csv(
    MASTER_FILE,
    usecols=[
        "run_id",
        "updated_at",
    ],
    low_memory=False,
)

master = (
    master
    .dropna(subset=["run_id"])
    .drop_duplicates(
        subset=["run_id"],
        keep="last"
    )
)

n_before = len(df)

df = df.merge(
    master,
    on="run_id",
    how="left",
    validate="many_to_one",
    sort=False,
)

assert len(df) == n_before

# Restore explicitly after merge
df = (
    df
    .sort_values("_original_order")
    .reset_index(drop=True)
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    utc=True,
    errors="coerce"
)

df["updated_at"] = pd.to_datetime(
    df["updated_at"],
    utc=True,
    errors="coerce"
)

print(
    f"Missing updated_at: "
    f"{df['updated_at'].isna().sum():,}"
)

# ============================================================
# CORRECTED FIRST-ATTEMPT COHORT
# ============================================================

df = df[
    df["run_attempt"] == 1
].copy()

# Preserve original relative order
df = (
    df
    .sort_values("_original_order")
    .reset_index(drop=True)
)

print("\nCorrected cohort:")
print(f"Rows: {len(df):,}")
print(f"Repositories: {df['repo'].nunique():,}")
print(f"Commits: {df['commit_sha'].nunique():,}")
print(f"Failures: {int(df[TARGET].sum()):,}")
print(f"Failure rate: {df[TARGET].mean():.6f}")

assert len(df) == 142629
assert df["repo"].nunique() == 27
assert df["commit_sha"].nunique() == 26555
assert int(df[TARGET].sum()) == 8378

# ============================================================
# STRICT E2
# ============================================================

print("\nReconstructing strict E2...")

commit_first = (
    df.groupby(
        "commit_sha",
        as_index=False
    )["created_at"]
    .min()
    .sort_values(
        ["created_at", "commit_sha"]
    )
    .reset_index(drop=True)
)

split_idx = int(
    np.floor(
        0.80 * len(commit_first)
    )
)

provisional_test_commits = (
    commit_first
    .iloc[split_idx:]
    ["commit_sha"]
)

boundary = df.loc[
    df["commit_sha"].isin(
        provisional_test_commits
    ),
    "created_at",
].min()

commit_ranges = (
    df.groupby("commit_sha")["created_at"]
    .agg(["min", "max"])
)

spanning_commits = commit_ranges[
    (commit_ranges["min"] < boundary)
    &
    (commit_ranges["max"] >= boundary)
].index

strict_df = df[
    ~df["commit_sha"].isin(
        spanning_commits
    )
].copy()

# IMPORTANT:
# Do NOT sort chronologically for model fitting.
# Keep original prepared-data order.
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
print(
    f"Boundary-spanning commits: "
    f"{len(spanning_commits):,}"
)

print(
    f"Train: {len(train):,} rows | "
    f"{train['commit_sha'].nunique():,} commits | "
    f"{int(train[TARGET].sum()):,} failures"
)

print(
    f"Test: {len(test):,} rows | "
    f"{test['commit_sha'].nunique():,} commits | "
    f"{int(test[TARGET].sum()):,} failures"
)

assert len(spanning_commits) == 25
assert len(train) == 105814
assert train["commit_sha"].nunique() == 21219
assert int(train[TARGET].sum()) == 6683

assert len(test) == 36036
assert test["commit_sha"].nunique() == 5311
assert int(test[TARGET].sum()) == 1610

assert (
    len(
        set(train["commit_sha"])
        &
        set(test["commit_sha"])
    )
    == 0
)

# ============================================================
# HISTORY FEATURE GENERATION
# ============================================================

print(
    "\nGenerating leakage-safe "
    "completion-aware history features..."
)

# Use immutable ID to reattach features later.
train["_period"] = "train"
test["_period"] = "test"

timeline = pd.concat(
    [train, test],
    ignore_index=True
)

# Chronological copy ONLY for feature generation
chrono = (
    timeline
    .sort_values(
        [
            "created_at",
            "_original_order",
        ],
        kind="mergesort",
    )
    .reset_index(drop=True)
)

completion_events = (
    timeline[
        timeline["updated_at"].notna()
    ]
    .sort_values(
        [
            "updated_at",
            "_original_order",
        ],
        kind="mergesort",
    )
    .reset_index(drop=True)
)

observable = defaultdict(list)

j = 0
n_completion = len(completion_events)

global_completed_failures = 0
global_completed_count = 0

feature_records = []

groups = chrono.groupby(
    "created_at",
    sort=True
)

n_groups = chrono[
    "created_at"
].nunique()

group_counter = 0

for current_time, batch in groups:

    group_counter += 1

    # --------------------------------------------------------
    # Reveal only outcomes actually known BEFORE prediction.
    # Strict inequality is intentional.
    # --------------------------------------------------------

    while (
        j < n_completion
        and
        completion_events.iloc[j]["updated_at"]
        < current_time
    ):

        r = completion_events.iloc[j]

        observable[
            r["repo"]
        ].append(
            {
                "commit_sha":
                    r["commit_sha"],

                "updated_at":
                    r["updated_at"],

                "outcome":
                    int(r[TARGET]),
            }
        )

        global_completed_failures += int(
            r[TARGET]
        )

        global_completed_count += 1

        j += 1

    # No future-derived fallback.
    if global_completed_count > 0:
        global_rate = (
            global_completed_failures
            / global_completed_count
        )
    else:
        # Fixed neutral cold-start value.
        global_rate = 0.5

    # --------------------------------------------------------
    # Every execution triggered at same timestamp sees same
    # observable outcome state.
    # --------------------------------------------------------

    for _, row in batch.iterrows():

        repo = row["repo"]
        current_commit = row["commit_sha"]

        repo_hist = observable.get(
            repo,
            []
        )

        # Strict variant:
        # exclude ALL outcomes from current commit.
        eligible = [
            x for x in repo_hist
            if x["commit_sha"]
            != current_commit
        ]

        # ----------------------------------------------------
        # Full completed repository failure rate
        # ----------------------------------------------------

        if len(eligible) > 0:

            repo_rate = float(
                np.mean(
                    [
                        x["outcome"]
                        for x in eligible
                    ]
                )
            )

        else:
            repo_rate = global_rate

        # ----------------------------------------------------
        # Previous 20 completed, different-commit executions
        # ----------------------------------------------------

        last20 = eligible[-HISTORY_K:]

        if len(last20) > 0:

            prev20_rate = float(
                np.mean(
                    [
                        x["outcome"]
                        for x in last20
                    ]
                )
            )

            last_outcome = float(
                last20[-1]["outcome"]
            )

        else:

            prev20_rate = global_rate
            last_outcome = np.nan

        # ----------------------------------------------------
        # Historical volume
        # ----------------------------------------------------

        hist_count = len(eligible)

        # ----------------------------------------------------
        # Time since most recently COMPLETED failure
        # ----------------------------------------------------

        last_failure_time = None

        for x in reversed(eligible):

            if x["outcome"] == 1:
                last_failure_time = (
                    x["updated_at"]
                )
                break

        if last_failure_time is None:

            time_since_failure = np.nan

        else:

            time_since_failure = (
                current_time
                - last_failure_time
            ).total_seconds()

            assert time_since_failure > 0

        feature_records.append(
            {
                "_original_order":
                    row["_original_order"],

                "repo_completed_failure_rate":
                    repo_rate,

                "previous20_completed_failure_rate":
                    prev20_rate,

                "last_completed_outcome":
                    last_outcome,

                "completed_history_count":
                    hist_count,

                "time_since_last_completed_failure":
                    time_since_failure,
            }
        )

    if (
        group_counter % 10000 == 0
        or group_counter == n_groups
    ):
        print(
            f"Trigger-time groups: "
            f"{group_counter:,}/"
            f"{n_groups:,}"
        )

hist = pd.DataFrame(feature_records)

assert (
    hist["_original_order"].nunique()
    == len(timeline)
)

# ============================================================
# ATTACH HISTORY TO ORIGINAL MODEL ORDER
# ============================================================

timeline = timeline.merge(
    hist,
    on="_original_order",
    how="left",
    validate="one_to_one",
    sort=False,
)

# Explicitly restore immutable order
timeline = (
    timeline
    .sort_values("_original_order")
    .reset_index(drop=True)
)

# log-transform count AFTER construction
timeline[
    "completed_history_count"
] = np.log1p(
    timeline[
        "completed_history_count"
    ].astype(float)
)

train_h = timeline[
    timeline["_period"] == "train"
].copy()

test_h = timeline[
    timeline["_period"] == "test"
].copy()

# Restore exact original order one final time
train_h = (
    train_h
    .sort_values("_original_order")
    .reset_index(drop=True)
)

test_h = (
    test_h
    .sort_values("_original_order")
    .reset_index(drop=True)
)

assert len(train_h) == 105814
assert len(test_h) == 36036

# Verify test labels identical to frozen E2
assert int(test_h[TARGET].sum()) == 1610

print("\nHistory feature generation complete.")

print("\nTraining missingness:")
print(
    train_h[
        HISTORY_FEATURES
    ].isna().sum()
)

print("\nTest missingness:")
print(
    test_h[
        HISTORY_FEATURES
    ].isna().sum()
)

# ============================================================
# SAVE HISTORY FEATURE AUDIT
# ============================================================

audit_cols = [
    "_original_order",
    "run_id",
    "repo",
    "commit_sha",
    "created_at",
    "updated_at",
    "_period",
    TARGET,
] + HISTORY_FEATURES

timeline[
    audit_cols
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "history_feature_audit.csv"
    ),
    index=False,
)

# ============================================================
# EXPERIMENTS
# ============================================================

specifications = {
    "static_corrected":
        NUMERIC_STATIC,

    "history_augmented":
        NUMERIC_STATIC
        + HISTORY_FEATURES,
}

y_train = (
    train_h[TARGET]
    .astype(int)
    .to_numpy()
)

y_test = (
    test_h[TARGET]
    .astype(int)
    .to_numpy()
)

pooled_rows = []
per_repo_rows = []
prediction_frames = []

print("\n" + "=" * 100)
print("MODEL EXPERIMENTS")
print("=" * 100)

for spec_name, numeric_features in specifications.items():

    feature_cols = (
        numeric_features
        + BINARY_STATIC
        + CATEGORICAL_STATIC
    )

    X_train = train_h[
        feature_cols
    ].copy()

    X_test = test_h[
        feature_cols
    ].copy()

    print(
        f"\n--- {spec_name} ---"
    )

    for model_name in MODEL_NAMES:

        preprocessor = build_preprocessor(
            numeric_features
        )

        estimator = make_model(
            model_name,
            pd.Series(y_train)
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    estimator
                ),
            ]
        )

        t0 = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train
        )

        train_sec = (
            time.perf_counter() - t0
        )

        t0 = time.perf_counter()

        score = pipeline.predict_proba(
            X_test
        )[:, 1]

        infer_sec = (
            time.perf_counter() - t0
        )

        m = evaluate(
            y_test,
            score
        )

        pooled_rows.append(
            {
                "specification":
                    spec_name,
                "model":
                    model_name,
                **m,
                "training_seconds":
                    train_sec,
                "inference_seconds":
                    infer_sec,
            }
        )

        print(
            f"{model_name}: "
            f"PR-AUC={m['pr_auc']:.6f} | "
            f"Lift={m['pr_auc_lift']:.3f}x | "
            f"ROC-AUC={m['roc_auc']:.6f}"
        )

        p = pd.DataFrame(
            {
                "_original_order":
                    test_h[
                        "_original_order"
                    ].values,

                "run_id":
                    test_h["run_id"].values,

                "repo":
                    test_h["repo"].values,

                "commit_sha":
                    test_h[
                        "commit_sha"
                    ].values,

                "created_at":
                    test_h[
                        "created_at"
                    ].values,

                "y_true":
                    y_test,

                "specification":
                    spec_name,

                "model":
                    model_name,

                "score":
                    score,
            }
        )

        prediction_frames.append(p)

        for repo, g in p.groupby("repo"):

            yr = g[
                "y_true"
            ].to_numpy()

            sr = g[
                "score"
            ].to_numpy()

            # Cannot meaningfully compute PR-AUC lift
            # if test repository has zero failures.
            if yr.sum() == 0:
                continue

            rm = evaluate(
                yr,
                sr
            )

            per_repo_rows.append(
                {
                    "repo": repo,
                    "specification":
                        spec_name,
                    "model":
                        model_name,
                    **rm,
                }
            )

# ============================================================
# HISTORY-ONLY BASELINES
# ============================================================

print("\n" + "=" * 100)
print("HISTORY-ONLY BASELINES")
print("=" * 100)

baseline_scores = {
    "repo_completed_failure_rate":
        test_h[
            "repo_completed_failure_rate"
        ].to_numpy(),

    "previous20_completed_no_same_commit":
        test_h[
            "previous20_completed_failure_rate"
        ].to_numpy(),

    "last_completed_outcome":
        test_h[
            "last_completed_outcome"
        ]
        .fillna(0.5)
        .to_numpy(),
}

for name, score in baseline_scores.items():

    m = evaluate(
        y_test,
        score
    )

    pooled_rows.append(
        {
            "specification":
                "history_baseline",
            "model": name,
            **m,
            "training_seconds": 0.0,
            "inference_seconds": 0.0,
        }
    )

    print(
        f"{name}: "
        f"PR-AUC={m['pr_auc']:.6f} | "
        f"Lift={m['pr_auc_lift']:.3f}x | "
        f"ROC-AUC={m['roc_auc']:.6f}"
    )

    p = pd.DataFrame(
        {
            "_original_order":
                test_h[
                    "_original_order"
                ].values,

            "run_id":
                test_h["run_id"].values,

            "repo":
                test_h["repo"].values,

            "commit_sha":
                test_h[
                    "commit_sha"
                ].values,

            "created_at":
                test_h[
                    "created_at"
                ].values,

            "y_true":
                y_test,

            "specification":
                "history_baseline",

            "model":
                name,

            "score":
                score,
        }
    )

    prediction_frames.append(p)

    for repo, g in p.groupby("repo"):

        yr = g["y_true"].to_numpy()
        sr = g["score"].to_numpy()

        if yr.sum() == 0:
            continue

        rm = evaluate(
            yr,
            sr
        )

        per_repo_rows.append(
            {
                "repo": repo,
                "specification":
                    "history_baseline",
                "model": name,
                **rm,
            }
        )

# ============================================================
# TABLES
# ============================================================

pooled = pd.DataFrame(
    pooled_rows
)

per_repo = pd.DataFrame(
    per_repo_rows
)

predictions = pd.concat(
    prediction_frames,
    ignore_index=True
)

macro_rows = []

for (
    spec,
    model
), g in per_repo.groupby(
    [
        "specification",
        "model",
    ]
):

    macro_rows.append(
        {
            "specification": spec,
            "model": model,

            "repositories_evaluated":
                int(
                    g["repo"].nunique()
                ),

            "macro_mean_pr_auc":
                float(
                    g["pr_auc"].mean()
                ),

            "macro_median_pr_auc":
                float(
                    g["pr_auc"].median()
                ),

            "macro_mean_lift":
                float(
                    g[
                        "pr_auc_lift"
                    ].mean()
                ),

            "macro_median_lift":
                float(
                    g[
                        "pr_auc_lift"
                    ].median()
                ),

            "macro_mean_roc_auc":
                float(
                    g[
                        "roc_auc"
                    ].mean()
                ),

            "macro_median_roc_auc":
                float(
                    g[
                        "roc_auc"
                    ].median()
                ),

            "repos_beating_prevalence":
                int(
                    (
                        g[
                            "pr_auc_lift"
                        ] > 1.0
                    ).sum()
                ),
        }
    )

macro = pd.DataFrame(
    macro_rows
)

# ============================================================
# REPRODUCIBILITY CHECK
# ============================================================

expected_static = {
    "LR": 0.088905,
    "RF": 0.116913,
    "XGB": 0.106288,
    "LGBM": 0.109073,
}

check_rows = []

print("\n" + "=" * 100)
print("STATIC MODEL REPRODUCIBILITY CHECK")
print("=" * 100)

for model, expected in expected_static.items():

    actual = pooled[
        (
            pooled["specification"]
            == "static_corrected"
        )
        &
        (
            pooled["model"]
            == model
        )
    ]["pr_auc"].iloc[0]

    diff = actual - expected

    check_rows.append(
        {
            "model": model,
            "expected_pr_auc":
                expected,
            "actual_pr_auc":
                actual,
            "difference":
                diff,
            "absolute_difference":
                abs(diff),
        }
    )

    print(
        f"{model}: expected={expected:.6f} | "
        f"actual={actual:.6f} | "
        f"diff={diff:+.6f}"
    )

repro_check = pd.DataFrame(
    check_rows
)

# ============================================================
# SAVE
# ============================================================

pooled.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_pooled_results.csv"
    ),
    index=False
)

per_repo.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_per_repo_results.csv"
    ),
    index=False
)

macro.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_macro_results.csv"
    ),
    index=False
)

predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "e2_predictions.csv"
    ),
    index=False
)

repro_check.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "static_reproducibility_check.csv"
    ),
    index=False
)

metadata = {
    "version":
        "history_augmented_ml_v2",

    "prediction_time":
        "created_at",

    "outcome_availability":
        "updated_at strictly earlier than current created_at",

    "same_commit_history":
        "excluded",

    "history_k":
        HISTORY_K,

    "cold_start_fallback":
        0.5,

    "future_training_rate_fallback_used":
        False,

    "raw_repo_identity_feature":
        False,

    "run_number_feature":
        False,

    "run_attempt_feature":
        False,

    "first_attempt_only":
        True,

    "history_features":
        HISTORY_FEATURES,

    "boundary":
        str(boundary),

    "boundary_spanning_commits":
        int(len(spanning_commits)),

    "train_rows":
        int(len(train_h)),

    "test_rows":
        int(len(test_h)),

    "test_failures":
        int(y_test.sum()),
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "experiment_metadata.json"
    ),
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 100)
print("POOLED RESULTS")
print("=" * 100)

print(
    pooled[
        [
            "specification",
            "model",
            "pr_auc",
            "pr_auc_lift",
            "roc_auc",
        ]
    ]
    .sort_values(
        "pr_auc",
        ascending=False
    )
    .to_string(index=False)
)

print("\n" + "=" * 100)
print("MACRO / WITHIN-REPOSITORY RESULTS")
print("=" * 100)

print(
    macro[
        [
            "specification",
            "model",
            "repositories_evaluated",
            "macro_mean_pr_auc",
            "macro_median_pr_auc",
            "macro_mean_lift",
            "macro_median_lift",
            "repos_beating_prevalence",
        ]
    ]
    .sort_values(
        "macro_median_lift",
        ascending=False
    )
    .to_string(index=False)
)

print("\nSaved to:")
print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print("\nDONE.")