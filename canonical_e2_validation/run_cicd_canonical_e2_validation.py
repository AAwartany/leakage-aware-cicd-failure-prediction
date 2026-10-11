"""Canonical E2 static-baseline validation, matching leakage ladder L4.
Run from the research folder: python run_cicd_canonical_e2_validation.py
Requires: cicd_prepared/cicd_clean_model_dataset.csv
Outputs metrics, per-run predictions with run_id, fingerprints, environment.
No history/H1 models are retrained; this does not validate H1 deltas.
"""
import os,sys,json,hashlib,platform,time,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn,xgboost,lightgbm,scipy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
RANDOM_STATE=42
TARGET='target_failure'
GROUP='commit_sha'
CORRECTED_NUMERIC=['msg_len','num_parents','additions','deletions','files_modified','time_since_last_commit','commit_to_pipeline_delay','trigger_hour','trigger_weekday']
BINARY=['is_merge_clean','is_primary_branch','trigger_weekend']
CATEGORICAL=['event']

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

def make_model(name, y_train):

    y_train = np.asarray(y_train)

    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))

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

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def hash_ids(df):
    return sha_bytes(('\n'.join(df['run_id'].astype(str))).encode('utf-8'))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='cicd_prepared/cicd_clean_model_dataset.csv')
    p.add_argument('--out',default='cicd_canonical_e2_validation')
    p.add_argument('--models',nargs='+',choices=['LR','RF','XGB','LGBM'],default=['LR','RF','XGB','LGBM'])
    args=p.parse_args()
    path=Path(args.data)
    if not path.is_file(): raise SystemExit(f'Missing input: {path.resolve()}')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(path,low_memory=False)
    df['_original_order']=np.arange(len(df))
    df['created_at']=pd.to_datetime(df['created_at'],utc=True,errors='coerce')
    assert len(df)==145161 and df['repo'].nunique()==27 and df[GROUP].nunique()==26989 and int(df[TARGET].sum())==8603,'Unexpected prepared cohort'
    assert df['run_id'].notna().all() and df['run_id'].is_unique,'run_id must be unique and non-null'
    first=df[df['run_attempt']==1].copy().sort_values('_original_order')
    assert len(first)==142629 and first[GROUP].nunique()==26555 and int(first[TARGET].sum())==8378
    commit_first=first.groupby(GROUP,as_index=False)['created_at'].min().sort_values(['created_at',GROUP]).reset_index(drop=True)
    split_idx=int(np.floor(.8*len(commit_first)))
    provisional_test_commits=commit_first.iloc[split_idx:][GROUP]
    boundary=first.loc[first[GROUP].isin(provisional_test_commits),'created_at'].min()
    ranges=first.groupby(GROUP)['created_at'].agg(['min','max'])
    spanning=ranges[(ranges['min']<boundary)&(ranges['max']>=boundary)].index
    strict=first[~first[GROUP].isin(spanning)].sort_values('_original_order').reset_index(drop=True)
    train=strict[strict['created_at']<boundary].copy()
    test=strict[strict['created_at']>=boundary].copy()
    assert len(spanning)==25
    assert str(boundary)=='2026-04-01 11:41:59+00:00',boundary
    assert (len(train),train[GROUP].nunique(),int(train[TARGET].sum()))==(105814,21219,6683)
    assert (len(test),test[GROUP].nunique(),int(test[TARGET].sum()))==(36036,5311,1610)
    assert set(train[GROUP]).isdisjoint(set(test[GROUP]))
    features=CORRECTED_NUMERIC+BINARY+CATEGORICAL
    assert all(f in df.columns for f in features)
    X_train=train[features]; X_test=test[features]
    y_train=train[TARGET].to_numpy(dtype=int); y_test=test[TARGET].to_numpy(dtype=int)
    manifest={'input_file':str(path.resolve()),'input_sha256':hash_file(path),'boundary':str(boundary),'excluded_boundary_spanning_commits':len(spanning),'train_rows':len(train),'test_rows':len(test),'train_commits':train[GROUP].nunique(),'test_commits':test[GROUP].nunique(),'train_failures':int(y_train.sum()),'test_failures':int(y_test.sum()),'train_run_id_order_sha256':hash_ids(train),'test_run_id_order_sha256':hash_ids(test),'features_in_order':features,'feature_spec_sha256':sha_bytes(json.dumps(features).encode()),'random_state':RANDOM_STATE,'versions':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__,'lightgbm':lightgbm.__version__,'scipy':scipy.__version__}}
    (out/'canonical_e2_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    metrics=[]
    for name in args.models:
        print('Training',name,flush=True)
        t0=time.perf_counter()
        pipe=Pipeline([('preprocessor',build_preprocessor(CORRECTED_NUMERIC)),('model',make_model(name,y_train))])
        pipe.fit(X_train,y_train)
        elapsed=time.perf_counter()-t0
        prob=pipe.predict_proba(X_test)[:,1]
        ap=float(average_precision_score(y_test,prob)); roc=float(roc_auc_score(y_test,prob))
        pred=pd.DataFrame({'run_id':test['run_id'].to_numpy(),'repo':test['repo'].to_numpy(),'commit_sha':test[GROUP].to_numpy(),'created_at':test['created_at'].astype(str).to_numpy(),'y_true':y_test,'score':prob})
        pred.to_csv(out/f'canonical_e2_{name.lower()}_predictions.csv',index=False)
        rec={'model':name,'pr_auc':ap,'roc_auc':roc,'test_rows':len(test),'failures':int(y_test.sum()),'prevalence':float(y_test.mean()),'train_seconds':round(elapsed,3),'predictions_sha256':sha_bytes(np.asarray(prob,dtype='<f8').tobytes()),'predictions_csv_sha256':hash_file(out/f'canonical_e2_{name.lower()}_predictions.csv')}
        metrics.append(rec)
        print(f'{name}: PR-AUC={ap:.9f}; ROC-AUC={roc:.9f}; elapsed={elapsed:.1f}s',flush=True)
    pd.DataFrame(metrics).to_csv(out/'canonical_e2_metrics.csv',index=False)
    print('Saved to:',out.resolve())
    print('Historical ladder L4 reference: LR=.088905 RF=.118190 XGB=.104995 LGBM=.109073')
    print('Historical robustness reference: RF=.116913 XGB=.106288 (not canonical)')
    print('Do NOT replace H1/bootstrap tables without separate validation.')
if __name__=='__main__':main()
