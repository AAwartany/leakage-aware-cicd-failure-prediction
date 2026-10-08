import os
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
HISTORY_K = 20

SHAP_SAMPLE_SIZE = None  # use the complete E2 test set
N_BOOTSTRAP = 1000

PREPARED_FILE = os.path.join(
    "cicd_prepared",
    "cicd_clean_model_dataset.csv"
)

MASTER_FILE = "final_research_dataset_MASTER.csv"

OUTPUT_DIR = "cicd_final_shap_h1"
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

CATEGORICAL = [
    "event"
]

HISTORY_FEATURES = [
    "repo_completed_failure_rate",
    "previous20_completed_failure_rate",
    "last_completed_outcome",
    "time_since_last_completed_failure",
]

NUMERIC = (
    NUMERIC_STATIC
    + HISTORY_FEATURES
)

FEATURES = (
    NUMERIC
    + BINARY
    + CATEGORICAL
)

# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("FINAL H1 HISTORY-AUGMENTED LIGHTGBM SHAP")
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

# ============================================================
# STRICT E2
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

split_idx = int(
    np.floor(
        0.80 * len(commit_first)
    )
)

provisional_test_commits = (
    commit_first
    .iloc[split_idx:]
    [GROUP]
)

boundary = df.loc[
    df[GROUP].isin(
        provisional_test_commits
    ),
    "created_at",
].min()

ranges = (
    df.groupby(GROUP)["created_at"]
    .agg(["min", "max"])
)

spanning = ranges[
    (ranges["min"] < boundary)
    &
    (ranges["max"] >= boundary)
].index

strict = df[
    ~df[GROUP].isin(
        spanning
    )
].copy()

train = strict[
    strict["created_at"] < boundary
].copy()

test = strict[
    strict["created_at"] >= boundary
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

assert len(train) == 105814
assert len(test) == 36036
assert train[GROUP].nunique() == 21219
assert test[GROUP].nunique() == 5311
assert int(test[TARGET].sum()) == 1610

print(f"\nBoundary: {boundary}")
print(f"Train rows: {len(train):,}")
print(f"Test rows: {len(test):,}")
print(f"Test failures: {int(test[TARGET].sum()):,}")

# ============================================================
# COMPLETION-AWARE HISTORY
# ============================================================

def add_history(train, test):

    """
    Efficient completion-aware history construction.

    Semantics are identical to the frozen V2 specification:

    - prediction time = created_at
    - an outcome is observable only when updated_at < created_at
    - same-commit historical executions are excluded
    - previous-20 uses the most recent 20 eligible completed executions
    - repository failure rate excludes same-commit history
    - last completed outcome excludes same-commit history
    - time since last completed failure excludes same-commit history
    - cold-start global rate = 0.5 when no completed global history exists
    """

    print("  Building train/test timeline...")

    timeline = pd.concat(
        [
            train.assign(_period="train"),
            test.assign(_period="test"),
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Prediction stream
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Completion stream
    # --------------------------------------------------------

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

    print(
        f"  Timeline rows: {len(timeline):,}"
    )

    print(
        f"  Rows with usable completion time: "
        f"{len(completions):,}"
    )

    print(
        f"  Missing updated_at: "
        f"{timeline['updated_at'].isna().sum():,}"
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    # Complete repository history:
    #
    # repo_history[repo] =
    #     [(commit_sha, outcome, updated_at), ...]
    #
    # Used only for previous-20 / last outcome.
    # We scan backwards and STOP after 20 eligible records.
    repo_history = defaultdict(list)

    # Failure-only history:
    #
    # Used for time since last failure.
    # Normally only one or a few entries need to be inspected.
    repo_failure_history = defaultdict(list)

    # Repository totals.
    repo_count = defaultdict(int)
    repo_failures = defaultdict(int)

    # Same-commit totals.
    #
    # This lets us calculate repository history excluding the
    # current commit in O(1), rather than copying the entire
    # repository history.
    commit_count = defaultdict(int)
    commit_failures = defaultdict(int)

    global_count = 0
    global_failures = 0

    # --------------------------------------------------------
    # Convert completion columns to arrays once.
    # This avoids thousands of expensive DataFrame.iloc calls.
    # --------------------------------------------------------

    c_updated = completions[
        "updated_at"
    ].tolist()

    c_repo = completions[
        "repo"
    ].tolist()

    c_commit = completions[
        GROUP
    ].tolist()

    c_outcome = (
        completions[
            TARGET
        ]
        .astype(int)
        .to_numpy()
    )

    j = 0
    n_completions = len(completions)

    records = []

    # --------------------------------------------------------
    # Process prediction timestamps in chronological order.
    # --------------------------------------------------------

    grouped = chrono.groupby(
        "created_at",
        sort=True
    )

    total_groups = chrono[
        "created_at"
    ].nunique()

    print(
        f"  Prediction timestamps: "
        f"{total_groups:,}"
    )

    group_counter = 0
    processed_rows = 0

    for current_time, batch in grouped:

        group_counter += 1

        # ----------------------------------------------------
        # Reveal only outcomes completed STRICTLY before the
        # current prediction time.
        #
        # Equality is intentionally excluded.
        # ----------------------------------------------------

        while (
            j < n_completions
            and
            c_updated[j] < current_time
        ):

            repo = c_repo[j]
            commit = c_commit[j]
            outcome = int(
                c_outcome[j]
            )
            completed_time = c_updated[j]

            repo_history[
                repo
            ].append(
                (
                    commit,
                    outcome,
                    completed_time,
                )
            )

            repo_count[
                repo
            ] += 1

            repo_failures[
                repo
            ] += outcome

            key = (
                repo,
                commit
            )

            commit_count[
                key
            ] += 1

            commit_failures[
                key
            ] += outcome

            if outcome == 1:

                repo_failure_history[
                    repo
                ].append(
                    (
                        commit,
                        completed_time,
                    )
                )

            global_count += 1
            global_failures += outcome

            j += 1

        global_rate = (
            global_failures / global_count
            if global_count > 0
            else 0.5
        )

        # ----------------------------------------------------
        # Process all predictions having this exact created_at.
        # Since we do not reveal completions at equality, every
        # row in this batch sees the same observable time state.
        # ----------------------------------------------------

        needed = batch[
            [
                "_original_order",
                "repo",
                GROUP,
            ]
        ]

        for (
            original_order,
            repo,
            current_commit,
        ) in needed.itertuples(
            index=False,
            name=None
        ):

            key = (
                repo,
                current_commit
            )

            # ------------------------------------------------
            # Repository historical failure rate excluding
            # same-commit completed executions.
            # ------------------------------------------------

            eligible_count = (
                repo_count[repo]
                - commit_count[key]
            )

            eligible_failures = (
                repo_failures[repo]
                - commit_failures[key]
            )

            if eligible_count > 0:

                repo_rate = (
                    eligible_failures
                    / eligible_count
                )

            else:

                repo_rate = global_rate

            # ------------------------------------------------
            # Most recent 20 completed executions excluding
            # current commit.
            #
            # Critical optimization:
            # scan BACKWARDS and stop after finding 20 rather
            # than constructing a copy of all eligible history.
            # ------------------------------------------------

            recent_outcomes = []

            history = repo_history.get(
                repo,
                []
            )

            for (
                hist_commit,
                hist_outcome,
                hist_time,
            ) in reversed(history):

                if (
                    hist_commit
                    == current_commit
                ):
                    continue

                recent_outcomes.append(
                    hist_outcome
                )

                if (
                    len(recent_outcomes)
                    == HISTORY_K
                ):
                    break

            if recent_outcomes:

                previous20 = float(
                    np.mean(
                        recent_outcomes
                    )
                )

                # Because we scanned in reverse chronological
                # order, element zero is the most recently
                # completed eligible execution.
                last_outcome = int(
                    recent_outcomes[0]
                )

            else:

                previous20 = global_rate
                last_outcome = np.nan

            # ------------------------------------------------
            # Most recent completed failure excluding current
            # commit.
            # ------------------------------------------------

            last_failure_time = None

            failure_history = (
                repo_failure_history.get(
                    repo,
                    []
                )
            )

            for (
                failure_commit,
                failure_time,
            ) in reversed(
                failure_history
            ):

                if (
                    failure_commit
                    != current_commit
                ):

                    last_failure_time = (
                        failure_time
                    )

                    break

            if last_failure_time is None:

                since_failure = np.nan

            else:

                since_failure = (
                    current_time
                    - last_failure_time
                ).total_seconds()

            records.append(
                {
                    "_original_order":
                        original_order,

                    "repo_completed_failure_rate":
                        repo_rate,

                    "previous20_completed_failure_rate":
                        previous20,

                    "last_completed_outcome":
                        last_outcome,

                    "completed_history_count":
                        eligible_count,

                    "time_since_last_completed_failure":
                        since_failure,
                }
            )

        processed_rows += len(batch)

        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        if (
            processed_rows % 10000
            < len(batch)
        ):

            print(
                f"  Processed "
                f"{processed_rows:,}/"
                f"{len(chrono):,} predictions "
                f"({processed_rows / len(chrono):.1%}) | "
                f"observable completions="
                f"{j:,}"
            )

    # --------------------------------------------------------
    # Merge generated features
    # --------------------------------------------------------

    print(
        "  Merging history features..."
    )

    history_df = pd.DataFrame(
        records
    )

    assert (
        len(history_df)
        == len(timeline)
    )

    assert (
        history_df[
            "_original_order"
        ].is_unique
    )

    timeline = timeline.merge(
        history_df,
        on="_original_order",
        how="left",
        validate="one_to_one",
        sort=False
    )

    # Preserve frozen V2 transformation.
    timeline[
        "completed_history_count"
    ] = np.log1p(
        timeline[
            "completed_history_count"
        ].astype(float)
    )

    train_h = (
        timeline[
            timeline["_period"]
            == "train"
        ]
        .sort_values(
            "_original_order"
        )
        .reset_index(drop=True)
    )

    test_h = (
        timeline[
            timeline["_period"]
            == "test"
        ]
        .sort_values(
            "_original_order"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    assert len(
        train_h
    ) == len(train)

    assert len(
        test_h
    ) == len(test)

    assert (
        train_h[
            "_original_order"
        ].tolist()
        ==
        train[
            "_original_order"
        ].tolist()
    )

    assert (
        test_h[
            "_original_order"
        ].tolist()
        ==
        test[
            "_original_order"
        ].tolist()
    )

    print(
        "  History generation complete."
    )

    return train_h, test_h
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

            outcome = int(
                r[TARGET]
            )

            observable[
                r["repo"]
            ].append(
                {
                    "commit_sha":
                        r[GROUP],

                    "updated_at":
                        r["updated_at"],

                    "outcome":
                        outcome,
                }
            )

            global_failures += outcome
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

            # Exclude same-commit outcomes
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

                recent = eligible[
                    -HISTORY_K:
                ]

                previous20 = np.mean(
                    [
                        x["outcome"]
                        for x in recent
                    ]
                )

                last_outcome = (
                    recent[-1]["outcome"]
                )

            else:

                repo_rate = global_rate
                previous20 = global_rate
                last_outcome = np.nan

            last_failure = None

            for x in reversed(
                eligible
            ):

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
                        previous20,

                    "last_completed_outcome":
                        last_outcome,

                    "completed_history_count":
                        len(eligible),

                    "time_since_last_completed_failure":
                        since_failure,
                }
            )

    history_df = pd.DataFrame(
        records
    )

    timeline = timeline.merge(
        history_df,
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


print("\nGenerating completion-aware history...")

train_h, test_h = add_history(
    train,
    test
)

# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
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
                ]
            ),
            NUMERIC,
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
            CATEGORICAL,
        ),
    ]
)

X_train_raw = train_h[
    FEATURES
].copy()

X_test_raw = test_h[
    FEATURES
].copy()

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

X_train = preprocessor.fit_transform(
    X_train_raw
)

X_test = preprocessor.transform(
    X_test_raw
)

# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

# ============================================================
# TRAIN FINAL REFERENCE MODEL
# ============================================================

neg = int(
    (y_train == 0).sum()
)

pos = int(
    (y_train == 1).sum()
)

spw = neg / pos

model = LGBMClassifier(
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

print("\nTraining final history-augmented LightGBM...")

model.fit(
    X_train,
    y_train
)

score = model.predict_proba(
    X_test
)[:, 1]

pr = average_precision_score(
    y_test,
    score
)

roc = roc_auc_score(
    y_test,
    score
)

prevalence = y_test.mean()

print(
    f"PR-AUC={pr:.6f} | "
    f"Lift={pr/prevalence:.3f}x | "
    f"ROC-AUC={roc:.6f}"
)

# ============================================================
# COMPLETE E2 SHAP SET
#
# Reviewer-robust H1 specification: use every E2 test execution.
# No outcome-enriched SHAP subsampling is performed.
# ============================================================

sample_idx = np.arange(len(y_test))
X_shap = X_test
y_shap = y_test
commit_shap = (
    test_h[GROUP]
    .astype(str)
    .to_numpy()
)

print(
    f"\nSHAP set: complete E2 test set | "
    f"{len(sample_idx):,} rows | "
    f"{int(y_shap.sum()):,} failures | "
    f"{int((y_shap == 0).sum()):,} successes | "
    f"{pd.Series(commit_shap).nunique():,} commits"
)

# ============================================================
# SHAP
# ============================================================

print("\nCalculating TreeSHAP...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_shap
)

# SHAP version compatibility
if isinstance(
    shap_values,
    list
):

    if len(shap_values) == 2:

        shap_values = (
            shap_values[1]
        )

    else:

        shap_values = (
            shap_values[0]
        )

shap_values = np.asarray(
    shap_values
)

if shap_values.ndim == 3:

    shap_values = (
        shap_values[:, :, -1]
    )

assert (
    shap_values.shape[0]
    == len(sample_idx)
)

assert (
    shap_values.shape[1]
    == len(feature_names)
)

# ============================================================
# MAP ENCODED FEATURES BACK TO CONCEPTUAL FEATURES
# ============================================================

def conceptual_feature(name):

    # numeric:
    # num__msg_len
    if name.startswith(
        "num__"
    ):
        return name.replace(
            "num__",
            "",
            1
        )

    # binary:
    if name.startswith(
        "bin__"
    ):
        return name.replace(
            "bin__",
            "",
            1
        )

    # categorical event levels:
    if name.startswith(
        "cat__event_"
    ):
        return "event"

    return name


conceptual_names = [
    conceptual_feature(x)
    for x in feature_names
]

# ============================================================
# AGGREGATE MEAN ABS SHAP
# ============================================================

abs_shap = np.abs(
    shap_values
)

encoded_importance = pd.DataFrame(
    {
        "encoded_feature":
            feature_names,

        "conceptual_feature":
            conceptual_names,

        "mean_abs_shap":
            abs_shap.mean(
                axis=0
            ),
    }
)

conceptual_importance = (
    encoded_importance
    .groupby(
        "conceptual_feature",
        as_index=False
    )["mean_abs_shap"]
    .sum()
)

conceptual_importance[
    "importance_share"
] = (
    conceptual_importance[
        "mean_abs_shap"
    ]
    /
    conceptual_importance[
        "mean_abs_shap"
    ].sum()
)

conceptual_importance = (
    conceptual_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

conceptual_importance[
    "rank"
] = (
    np.arange(
        1,
        len(
            conceptual_importance
        ) + 1
    )
)

# ============================================================
# GROUPED INFORMATION FAMILIES
# ============================================================

family_map = {
    "repo_completed_failure_rate":
        "historical_state",

    "previous20_completed_failure_rate":
        "historical_state",

    "last_completed_outcome":
        "historical_state",

    "completed_history_count":
        "historical_state",

    "time_since_last_completed_failure":
        "historical_state",

    "event":
        "workflow_context",

    "msg_len":
        "change_characteristics",

    "num_parents":
        "change_characteristics",

    "additions":
        "change_characteristics",

    "deletions":
        "change_characteristics",

    "files_modified":
        "change_characteristics",

    "is_merge_clean":
        "change_characteristics",

    "commit_to_pipeline_delay":
        "timing_context",

    "time_since_last_commit":
        "timing_context",

    "trigger_hour":
        "timing_context",

    "trigger_weekday":
        "timing_context",

    "trigger_weekend":
        "timing_context",

    "is_primary_branch":
        "branch_context",
}

conceptual_importance[
    "family"
] = (
    conceptual_importance[
        "conceptual_feature"
    ].map(
        family_map
    )
)

family_importance = (
    conceptual_importance
    .groupby(
        "family",
        as_index=False
    )["mean_abs_shap"]
    .sum()
)

family_importance[
    "importance_share"
] = (
    family_importance[
        "mean_abs_shap"
    ]
    /
    family_importance[
        "mean_abs_shap"
    ].sum()
)

family_importance = (
    family_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

# ============================================================
# COMMIT-CLUSTER BOOTSTRAP IMPORTANCE / RANK STABILITY
#
# Resample commit_sha clusters with replacement. Each selected
# commit contributes all of its E2 executions, preserving the
# within-commit dependence structure.
# ============================================================

print(
    f"\nBootstrapping SHAP importance by commit "
    f"{N_BOOTSTRAP:,} times..."
)

conceptual_unique = (
    conceptual_importance["conceptual_feature"].tolist()
)

feature_to_cols = {}
for feature in conceptual_unique:
    feature_to_cols[feature] = np.where(
        np.array(conceptual_names) == feature
    )[0]

# Collapse encoded columns to conceptual-feature absolute SHAP
# values for every E2 execution.
row_conceptual = np.column_stack([
    abs_shap[:, cols].sum(axis=1)
    for feature, cols in feature_to_cols.items()
])

commit_codes, unique_commits = pd.factorize(
    commit_shap, sort=False
)
n_commits = len(unique_commits)
n_concepts = len(conceptual_unique)

# Pre-aggregate row counts and SHAP sums by commit so cluster
# bootstrap does not need to materialize large row-index arrays.
cluster_n = np.bincount(commit_codes, minlength=n_commits).astype(float)
cluster_sum = np.zeros((n_commits, n_concepts), dtype=float)
for j in range(n_concepts):
    cluster_sum[:, j] = np.bincount(
        commit_codes,
        weights=row_conceptual[:, j],
        minlength=n_commits,
    )

bootstrap_records = []
rng_boot = np.random.default_rng(RANDOM_STATE + 1)

for b in range(N_BOOTSTRAP):
    sampled_clusters = rng_boot.integers(
        0, n_commits, size=n_commits
    )
    denom = cluster_n[sampled_clusters].sum()
    means = cluster_sum[sampled_clusters].sum(axis=0) / denom

    values = dict(zip(conceptual_unique, means))
    ordered = sorted(values.items(), key=lambda x: x[1], reverse=True)
    ranks = {feature: rank for rank, (feature, _) in enumerate(ordered, start=1)}
    total = float(means.sum())

    for feature in conceptual_unique:
        bootstrap_records.append({
            "bootstrap": b,
            "conceptual_feature": feature,
            "mean_abs_shap": values[feature],
            "importance_share": values[feature] / total,
            "rank": ranks[feature],
        })

    if (b + 1) % 100 == 0:
        print(f"Bootstrap {b + 1:,}/{N_BOOTSTRAP:,}")

bootstrap_df = pd.DataFrame(bootstrap_records)

stability_rows = []

for feature, g in (
    bootstrap_df.groupby(
        "conceptual_feature"
    )
):

    stability_rows.append(
        {
            "conceptual_feature":
                feature,

            "importance_share_ci_low":
                g[
                    "importance_share"
                ].quantile(
                    0.025
                ),

            "importance_share_ci_high":
                g[
                    "importance_share"
                ].quantile(
                    0.975
                ),

            "median_rank":
                g["rank"].median(),

            "rank_ci_low":
                g[
                    "rank"
                ].quantile(
                    0.025
                ),

            "rank_ci_high":
                g[
                    "rank"
                ].quantile(
                    0.975
                ),

            "top_3_frequency":
                (
                    g["rank"] <= 3
                ).mean(),

            "top_5_frequency":
                (
                    g["rank"] <= 5
                ).mean(),
        }
    )

stability_df = pd.DataFrame(
    stability_rows
)

conceptual_importance = (
    conceptual_importance
    .merge(
        stability_df,
        on="conceptual_feature",
        how="left",
        validate="one_to_one"
    )
)

# ============================================================
# FAMILY BOOTSTRAP
# ============================================================

feature_family = {
    row[
        "conceptual_feature"
    ]: row["family"]

    for _, row in (
        conceptual_importance.iterrows()
    )
}

family_bootstrap_rows = []

for b, g in (
    bootstrap_df.groupby(
        "bootstrap"
    )
):

    g = g.copy()

    g["family"] = (
        g[
            "conceptual_feature"
        ].map(
            feature_family
        )
    )

    f = (
        g.groupby(
            "family",
            as_index=False
        )["mean_abs_shap"]
        .sum()
    )

    total = (
        f["mean_abs_shap"]
        .sum()
    )

    f[
        "importance_share"
    ] = (
        f["mean_abs_shap"]
        / total
    )

    f["bootstrap"] = b

    family_bootstrap_rows.append(
        f
    )

family_bootstrap_df = (
    pd.concat(
        family_bootstrap_rows,
        ignore_index=True
    )
)

family_stability = (
    family_bootstrap_df
    .groupby(
        "family"
    )["importance_share"]
    .agg(
        importance_share_mean="mean",

        importance_share_ci_low=lambda x:
            x.quantile(0.025),

        importance_share_ci_high=lambda x:
            x.quantile(0.975),
    )
    .reset_index()
)

family_importance = (
    family_importance
    .merge(
        family_stability,
        on="family",
        how="left",
        validate="one_to_one"
    )
)

# ============================================================
# SAVE
# ============================================================

encoded_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_encoded_feature_importance.csv"
    ),
    index=False
)

conceptual_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_conceptual_feature_importance.csv"
    ),
    index=False
)

family_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_family_importance.csv"
    ),
    index=False
)

stability_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_rank_stability.csv"
    ),
    index=False
)

bootstrap_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_bootstrap_distribution.csv"
    ),
    index=False
)

family_bootstrap_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_family_bootstrap_distribution.csv"
    ),
    index=False
)

sample_manifest = pd.DataFrame(
    {
        "_original_order":
            test_h.iloc[
                sample_idx
            ][
                "_original_order"
            ].values,

        "run_id":
            test_h.iloc[
                sample_idx
            ][
                "run_id"
            ].values,

        "repo":
            test_h.iloc[
                sample_idx
            ][
                "repo"
            ].values,

        "commit_sha":
            test_h.iloc[
                sample_idx
            ][
                GROUP
            ].values,

        "y_true":
            y_shap,
    }
)

sample_manifest.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_sample_manifest.csv"
    ),
    index=False
)

metadata = {
    "reference_model":
        "history-augmented LightGBM",

    "selection_reason":
        (
            "highest observed pooled PR-AUC "
            "under strict E2; not interpreted "
            "as universal model superiority"
        ),

    "e2_boundary":
        str(boundary),

    "test_rows":
        int(len(test_h)),

    "test_failures":
        int(y_test.sum()),

    "test_prevalence":
        float(prevalence),

    "model_pr_auc":
        float(pr),

    "model_pr_auc_lift":
        float(
            pr / prevalence
        ),

    "model_roc_auc":
        float(roc),

    "shap_sample_size":
        int(
            len(sample_idx)
        ),

    "shap_failures":
        int(
            y_shap.sum()
        ),

    "shap_successes":
        int(
            (y_shap == 0).sum()
        ),

    "sampling": "complete E2 test set",

    "shap_method":
        "TreeSHAP",

    "bootstrap_samples":
        N_BOOTSTRAP,

    "bootstrap_unit": "commit_sha cluster",

    "history_availability":
        "updated_at < created_at",

    "same_commit_history":
        "excluded",

    "history_k":
        HISTORY_K,

    "raw_repo_feature":
        False,

    "history_specification":
        "H1: completed_history_count removed",

    "run_number_feature":
        False,

    "run_attempt_feature":
        False,
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "shap_metadata.json"
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
# PRINT
# ============================================================

print("\n" + "=" * 100)
print("CONCEPTUAL FEATURE IMPORTANCE")
print("=" * 100)

print(
    conceptual_importance[
        [
            "rank",
            "conceptual_feature",
            "family",
            "importance_share",
            "importance_share_ci_low",
            "importance_share_ci_high",
            "median_rank",
            "rank_ci_low",
            "rank_ci_high",
            "top_5_frequency",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print("INFORMATION FAMILY IMPORTANCE")
print("=" * 100)

print(
    family_importance.to_string(
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