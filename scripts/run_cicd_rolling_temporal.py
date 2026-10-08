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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
HISTORY_K = 20

PREPARED_FILE = os.path.join(
    "cicd_prepared",
    "cicd_clean_model_dataset.csv"
)

MASTER_FILE = "final_research_dataset_MASTER.csv"
OUTPUT_DIR = "cicd_rolling_temporal"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "target_failure"
GROUP = "commit_sha"

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

BINARY = [
    "is_merge_clean",
    "is_primary_branch",
    "trigger_weekend",
]

CATEGORICAL = ["event"]

HISTORY_FEATURES = [
    "repo_completed_failure_rate",
    "previous20_completed_failure_rate",
    "last_completed_outcome",
    "completed_history_count",
    "time_since_last_completed_failure",
]

WINDOWS = [
    ("W1", 0.50, 0.60),
    ("W2", 0.60, 0.70),
    ("W3", 0.70, 0.80),
    ("W4", 0.80, 1.00),
]


def evaluate(y, score):

    y = np.asarray(y)
    score = np.asarray(score)

    prevalence = float(y.mean())

    pr = average_precision_score(
        y,
        score
    )

    roc = (
        roc_auc_score(y, score)
        if len(np.unique(y)) > 1
        else np.nan
    )

    return {
        "n": len(y),
        "failures": int(y.sum()),
        "prevalence": prevalence,
        "pr_auc": pr,
        "pr_auc_lift": (
            pr / prevalence
            if prevalence > 0
            else np.nan
        ),
        "roc_auc": roc,
    }


def preprocessor(numeric):

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),
                        (
                            "scale",
                            StandardScaler()
                        ),
                    ]
                ),
                numeric,
            ),
            (
                "bin",
                SimpleImputer(
                    strategy="most_frequent"
                ),
                BINARY,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
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
                CATEGORICAL,
            ),
        ]
    )


def model(name, y):

    neg = int((y == 0).sum())
    pos = int((y == 1).sum())

    spw = neg / pos

    if name == "RF":

        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
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
# LOAD
# ============================================================

print("=" * 100)
print("ROLLING-ORIGIN TEMPORAL VALIDATION")
print("=" * 100)

df = pd.read_csv(
    PREPARED_FILE,
    low_memory=False
)

df["_original_order"] = np.arange(
    len(df)
)

master = pd.read_csv(
    MASTER_FILE,
    usecols=[
        "run_id",
        "updated_at",
    ],
    low_memory=False
)

master = (
    master
    .dropna(subset=["run_id"])
    .drop_duplicates(
        subset=["run_id"],
        keep="last"
    )
)

df = df.merge(
    master,
    on="run_id",
    how="left",
    validate="many_to_one",
    sort=False
)

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

# First attempts only
df = df[
    df["run_attempt"] == 1
].copy()

df = (
    df
    .sort_values("_original_order")
    .reset_index(drop=True)
)

assert len(df) == 142629
assert df[GROUP].nunique() == 26555
assert int(df[TARGET].sum()) == 8378

print(f"\nCorrected cohort: {len(df):,}")
print(f"Commits: {df[GROUP].nunique():,}")
print(f"Failures: {int(df[TARGET].sum()):,}")

# ============================================================
# COMMIT ORDER
# ============================================================

commit_first = (
    df.groupby(
        GROUP,
        as_index=False
    )["created_at"]
    .min()
    .sort_values(
        ["created_at", GROUP]
    )
    .reset_index(drop=True)
)

n_commits = len(commit_first)

results = []
per_repo_results = []
composition_rows = []
window_rows = []


# ============================================================
# HISTORY BUILDER
# ============================================================

def add_history_features(train, test):

    timeline = pd.concat(
        [
            train.assign(_period="train"),
            test.assign(_period="test"),
        ],
        ignore_index=True
    )

    chrono = (
        timeline
        .sort_values(
            [
                "created_at",
                "_original_order"
            ],
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    completions = (
        timeline[
            timeline["updated_at"].notna()
        ]
        .sort_values(
            [
                "updated_at",
                "_original_order"
            ],
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    observable = defaultdict(list)

    global_failures = 0
    global_count = 0

    j = 0
    records = []

    for current_time, batch in chrono.groupby(
        "created_at",
        sort=True
    ):

        while (
            j < len(completions)
            and
            completions.iloc[j]["updated_at"]
            < current_time
        ):

            r = completions.iloc[j]

            observable[
                r["repo"]
            ].append(
                {
                    "commit_sha":
                        r[GROUP],
                    "updated_at":
                        r["updated_at"],
                    "outcome":
                        int(r[TARGET]),
                }
            )

            global_failures += int(
                r[TARGET]
            )

            global_count += 1
            j += 1

        global_rate = (
            global_failures / global_count
            if global_count > 0
            else 0.5
        )

        for _, row in batch.iterrows():

            history = observable.get(
                row["repo"],
                []
            )

            # Same commit excluded
            eligible = [
                x for x in history
                if x["commit_sha"]
                != row[GROUP]
            ]

            if eligible:

                repo_rate = np.mean(
                    [
                        x["outcome"]
                        for x in eligible
                    ]
                )

                last20 = eligible[
                    -HISTORY_K:
                ]

                prev20 = np.mean(
                    [
                        x["outcome"]
                        for x in last20
                    ]
                )

                last_outcome = (
                    last20[-1]["outcome"]
                )

            else:

                repo_rate = global_rate
                prev20 = global_rate
                last_outcome = np.nan

            last_failure = None

            for x in reversed(eligible):

                if x["outcome"] == 1:
                    last_failure = (
                        x["updated_at"]
                    )
                    break

            if last_failure is None:

                since_failure = np.nan

            else:

                since_failure = (
                    current_time
                    - last_failure
                ).total_seconds()

            records.append(
                {
                    "_original_order":
                        row["_original_order"],

                    "repo_completed_failure_rate":
                        repo_rate,

                    "previous20_completed_failure_rate":
                        prev20,

                    "last_completed_outcome":
                        last_outcome,

                    "completed_history_count":
                        len(eligible),

                    "time_since_last_completed_failure":
                        since_failure,
                }
            )

    h = pd.DataFrame(records)

    timeline = timeline.merge(
        h,
        on="_original_order",
        how="left",
        validate="one_to_one",
        sort=False
    )

    timeline[
        "completed_history_count"
    ] = np.log1p(
        timeline[
            "completed_history_count"
        ].astype(float)
    )

    train_h = (
        timeline[
            timeline["_period"] == "train"
        ]
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    test_h = (
        timeline[
            timeline["_period"] == "test"
        ]
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    return train_h, test_h


# ============================================================
# WINDOWS
# ============================================================

for (
    window_name,
    train_fraction,
    test_end_fraction
) in WINDOWS:

    print("\n" + "=" * 100)
    print(
        f"{window_name}: "
        f"train first {train_fraction:.0%}, "
        f"test until {test_end_fraction:.0%}"
    )
    print("=" * 100)

    train_cut_idx = int(
        np.floor(
            train_fraction
            * n_commits
        )
    )

    if test_end_fraction < 1.0:

        end_cut_idx = int(
            np.floor(
                test_end_fraction
                * n_commits
            )
        )

        test_end = (
            commit_first
            .iloc[end_cut_idx]
            ["created_at"]
        )

    else:
        test_end = None

    test_start = (
        commit_first
        .iloc[train_cut_idx]
        ["created_at"]
    )

    # --------------------------------------------------------
    # Exclude commits spanning train/test start boundary
    # --------------------------------------------------------

    ranges = (
        df.groupby(GROUP)["created_at"]
        .agg(["min", "max"])
    )

    spanning_start = ranges[
        (ranges["min"] < test_start)
        &
        (ranges["max"] >= test_start)
    ].index

    working = df[
        ~df[GROUP].isin(
            spanning_start
        )
    ].copy()

    # --------------------------------------------------------
    # For W1-W3 also exclude commits spanning the test-end edge
    # --------------------------------------------------------

    if test_end is not None:

        spanning_end = ranges[
            (ranges["min"] < test_end)
            &
            (ranges["max"] >= test_end)
        ].index

        working = working[
            ~working[GROUP].isin(
                spanning_end
            )
        ].copy()

    else:

        spanning_end = []

    train = working[
        working["created_at"]
        < test_start
    ].copy()

    if test_end is None:

        test = working[
            working["created_at"]
            >= test_start
        ].copy()

    else:

        test = working[
            (working["created_at"] >= test_start)
            &
            (working["created_at"] < test_end)
        ].copy()

    train = (
        train
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    test = (
        test
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    overlap = (
        set(train[GROUP])
        &
        set(test[GROUP])
    )

    assert len(overlap) == 0

    print(
        f"Train: {len(train):,} | "
        f"{train[GROUP].nunique():,} commits"
    )

    print(
        f"Test: {len(test):,} | "
        f"{test[GROUP].nunique():,} commits | "
        f"{int(test[TARGET].sum()):,} failures | "
        f"{test[TARGET].mean():.4%}"
    )

    print(
        f"Start-spanning commits removed: "
        f"{len(spanning_start):,}"
    )

    print(
        f"End-spanning commits removed: "
        f"{len(spanning_end):,}"
    )

    # --------------------------------------------------------
    # Per-repository composition requested by reviewer
    # --------------------------------------------------------

    repos = sorted(
        set(train["repo"])
        |
        set(test["repo"])
    )

    for repo in repos:

        tr = train[
            train["repo"] == repo
        ]

        te = test[
            test["repo"] == repo
        ]

        composition_rows.append(
            {
                "window":
                    window_name,

                "repo":
                    repo,

                "train_rows":
                    len(tr),

                "train_failures":
                    int(
                        tr[TARGET].sum()
                    ),

                "train_prevalence":
                    (
                        tr[TARGET].mean()
                        if len(tr)
                        else np.nan
                    ),

                "test_rows":
                    len(te),

                "test_failures":
                    int(
                        te[TARGET].sum()
                    ),

                "test_prevalence":
                    (
                        te[TARGET].mean()
                        if len(te)
                        else np.nan
                    ),
            }
        )

    window_rows.append(
        {
            "window":
                window_name,

            "train_fraction":
                train_fraction,

            "test_end_fraction":
                test_end_fraction,

            "test_start":
                test_start,

            "test_end":
                test_end,

            "train_rows":
                len(train),

            "test_rows":
                len(test),

            "train_commits":
                train[GROUP].nunique(),

            "test_commits":
                test[GROUP].nunique(),

            "train_failures":
                int(
                    train[TARGET].sum()
                ),

            "test_failures":
                int(
                    test[TARGET].sum()
                ),

            "test_prevalence":
                test[TARGET].mean(),

            "start_spanning_commits":
                len(spanning_start),

            "end_spanning_commits":
                len(spanning_end),
        }
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    print(
        "Generating completion-aware history..."
    )

    train_h, test_h = (
        add_history_features(
            train,
            test
        )
    )

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

    # --------------------------------------------------------
    # History-only baseline
    # --------------------------------------------------------

    history_score = (
        test_h[
            "previous20_completed_failure_rate"
        ]
        .to_numpy()
    )

    hm = evaluate(
        y_test,
        history_score
    )

    results.append(
        {
            "window":
                window_name,
            "specification":
                "history_baseline",
            "model":
                "Previous20",
            **hm,
        }
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    for spec in [
        "static",
        "history_augmented"
    ]:

        if spec == "static":

            numeric = (
                NUMERIC_STATIC
            )

        else:

            numeric = (
                NUMERIC_STATIC
                + HISTORY_FEATURES
            )

        features = (
            numeric
            + BINARY
            + CATEGORICAL
        )

        X_train = train_h[
            features
        ].copy()

        X_test = test_h[
            features
        ].copy()

        for model_name in [
            "RF",
            "LGBM"
        ]:

            print(
                f"Training "
                f"{spec} / "
                f"{model_name}..."
            )

            pipe = Pipeline(
                [
                    (
                        "preprocessor",
                        preprocessor(
                            numeric
                        )
                    ),
                    (
                        "model",
                        model(
                            model_name,
                            pd.Series(
                                y_train
                            )
                        )
                    ),
                ]
            )

            t0 = time.perf_counter()

            pipe.fit(
                X_train,
                y_train
            )

            training_time = (
                time.perf_counter()
                - t0
            )

            score = (
                pipe.predict_proba(
                    X_test
                )[:, 1]
            )

            m = evaluate(
                y_test,
                score
            )

            results.append(
                {
                    "window":
                        window_name,

                    "specification":
                        spec,

                    "model":
                        model_name,

                    **m,

                    "training_seconds":
                        training_time,
                }
            )

            # ----------------------------------------------
            # Per-repo evaluation
            # ----------------------------------------------

            pred = pd.DataFrame(
                {
                    "repo":
                        test_h[
                            "repo"
                        ].values,

                    "y":
                        y_test,

                    "score":
                        score,
                }
            )

            for repo, g in pred.groupby(
                "repo"
            ):

                # Need at least one failure and one success
                if (
                    g["y"].sum() == 0
                    or
                    g["y"].sum()
                    == len(g)
                ):
                    continue

                rm = evaluate(
                    g["y"],
                    g["score"]
                )

                per_repo_results.append(
                    {
                        "window":
                            window_name,

                        "repo":
                            repo,

                        "specification":
                            spec,

                        "model":
                            model_name,

                        **rm,
                    }
                )

    # --------------------------------------------------------
    # Also per-repo Previous20
    # --------------------------------------------------------

    history_pred = pd.DataFrame(
        {
            "repo":
                test_h["repo"].values,
            "y":
                y_test,
            "score":
                history_score,
        }
    )

    for repo, g in history_pred.groupby(
        "repo"
    ):

        if (
            g["y"].sum() == 0
            or
            g["y"].sum()
            == len(g)
        ):
            continue

        rm = evaluate(
            g["y"],
            g["score"]
        )

        per_repo_results.append(
            {
                "window":
                    window_name,

                "repo":
                    repo,

                "specification":
                    "history_baseline",

                "model":
                    "Previous20",

                **rm,
            }
        )


# ============================================================
# SUMMARIES
# ============================================================

results_df = pd.DataFrame(
    results
)

per_repo_df = pd.DataFrame(
    per_repo_results
)

composition_df = pd.DataFrame(
    composition_rows
)

windows_df = pd.DataFrame(
    window_rows
)

macro_rows = []

for (
    window,
    spec,
    model_name
), g in per_repo_df.groupby(
    [
        "window",
        "specification",
        "model"
    ]
):

    macro_rows.append(
        {
            "window":
                window,

            "specification":
                spec,

            "model":
                model_name,

            "repos_evaluated":
                g["repo"].nunique(),

            "macro_mean_pr_auc":
                g["pr_auc"].mean(),

            "macro_median_pr_auc":
                g["pr_auc"].median(),

            "macro_mean_lift":
                g[
                    "pr_auc_lift"
                ].mean(),

            "macro_median_lift":
                g[
                    "pr_auc_lift"
                ].median(),

            "repos_beating_prevalence":
                int(
                    (
                        g[
                            "pr_auc_lift"
                        ] > 1
                    ).sum()
                ),
        }
    )

macro_df = pd.DataFrame(
    macro_rows
)

# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_pooled_results.csv"
    ),
    index=False
)

per_repo_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_per_repo_results.csv"
    ),
    index=False
)

macro_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_macro_results.csv"
    ),
    index=False
)

composition_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_repo_composition.csv"
    ),
    index=False
)

windows_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_window_summary.csv"
    ),
    index=False
)

metadata = {
    "windows": WINDOWS,

    "prediction_time":
        "created_at",

    "history_availability":
        "updated_at < created_at",

    "same_commit_history":
        "excluded",

    "first_attempt_only":
        True,

    "run_number":
        False,

    "run_attempt_feature":
        False,

    "raw_repo_feature":
        False,

    "history_k":
        HISTORY_K,
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "rolling_metadata.json"
    ),
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 100)
print("ROLLING TEMPORAL RESULTS")
print("=" * 100)

print(
    results_df[
        [
            "window",
            "specification",
            "model",
            "prevalence",
            "pr_auc",
            "pr_auc_lift",
            "roc_auc",
        ]
    ].to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print("MACRO RESULTS")
print("=" * 100)

print(
    macro_df.to_string(
        index=False
    )
)

print("\nSaved to:")
print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print("\nDONE.")