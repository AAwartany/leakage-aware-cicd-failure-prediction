"""Frozen E2 H1 paired commit-cluster bootstrap; run with Python 3.13."""
import io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
N_BOOT=5000
SEED=42

def load_zip(name, member):
    with zipfile.ZipFile(PARENT/name) as z:
        return pd.read_csv(io.BytesIO(z.read(member)))

h1=load_zip('cicd_reviewer_robustness.zip','cicd_reviewer_robustness/e2_proxy_ablation_predictions.csv')
base=load_zip('cicd_history_augmented_ml_v2.zip','cicd_history_augmented_ml_v2/e2_predictions.csv')
key=['_original_order','run_id','repo','commit_sha','y_true']
assert len(h1[h1.specification.eq('H1') & h1.model.eq('LGBM')])==36036
pairs=[('H1 LGBM','H1','LGBM','Static LGBM','static','LGBM'),
       ('H1 RF','H1','RF','Static RF','static','RF'),
       ('H1 XGB','H1','XGB','Static XGB','static','XGB'),
       ('H1 LR','H1','LR','Static LR','static','LR')]
frames=[]
for label,spec,model,_,_,_ in pairs:
    x=h1[(h1.specification==spec)&(h1.model==model)][key+['score']].rename(columns={'score':label})
    frames.append(x)
for model in ['LGBM','RF','XGB','LR']:
    label='Static '+model
    x=h1[(h1.specification=='static')&(h1.model==model)][key+['score']].rename(columns={'score':label})
    frames.append(x)
for model,label in [('previous20_completed_no_same_commit','Previous20'),('repo_completed_failure_rate','RepoHistory')]:
    x=base[(base.specification=='history_baseline')&(base.model==model)][key+['score']].rename(columns={'score':label})
    frames.append(x)
merged=frames[0]
for frame in frames[1:]:
    merged=merged.merge(frame,on=key,how='inner',validate='one_to_one')
assert len(merged)==36036 and merged.commit_sha.nunique()==5311 and merged.y_true.sum()==1610
comparisons=[('H1 LGBM','Static LGBM'),('H1 RF','Static RF'),('H1 XGB','Static XGB'),('H1 LR','Static LR'),('H1 LGBM','Previous20'),('H1 LGBM','RepoHistory'),('H1 RF','Previous20'),('H1 RF','RepoHistory')]
columns=sorted(set(sum(([a,b] for a,b in comparisons),[])))
y=merged.y_true.to_numpy(dtype=np.int8)
scores={c:merged[c].to_numpy(dtype=np.float64) for c in columns}
commits=merged.commit_sha.astype(str).to_numpy()
unique,inv=np.unique(commits,return_inverse=True)
assert len(unique)==5311
members=[np.flatnonzero(inv==i) for i in range(len(unique))]
point={c:average_precision_score(y,scores[c]) for c in columns}
rng=np.random.default_rng(SEED)
draws={f'{a} - {b}':[] for a,b in comparisons}
print('Validated E2 rows:',len(y),'commits:',len(unique),'failures:',int(y.sum()),flush=True)
for iteration in range(N_BOOT):
    selected=rng.integers(0,len(unique),size=len(unique))
    idx=np.concatenate([members[i] for i in selected])
    yy=y[idx]
    if yy.min()==yy.max():
        continue
    boot={c:average_precision_score(yy,scores[c][idx]) for c in columns}
    for a,b in comparisons:
        draws[f'{a} - {b}'].append(boot[a]-boot[b])
    if (iteration+1)%500==0:
        print(f'Completed {iteration+1}/{N_BOOT}',flush=True)
rows=[]
for a,b in comparisons:
    delta=np.asarray(draws[f'{a} - {b}'])
    observed=point[a]-point[b]
    # Centered bootstrap test for H0: delta = 0; two-sided finite-sample p.
    p=(1+np.count_nonzero(np.abs(delta-observed)>=abs(observed)))/(len(delta)+1)
    rows.append(dict(comparison=f'{a} vs {b}',reference_pr_auc=point[b],h1_pr_auc=point[a],delta_pr_auc=observed,ci_low=np.quantile(delta,.025),ci_high=np.quantile(delta,.975),bootstrap_p_two_sided=p,valid_bootstrap=len(delta)))
result=pd.DataFrame(rows)
order=np.argsort(result.bootstrap_p_two_sided.to_numpy())
ps=result.bootstrap_p_two_sided.to_numpy()
adjusted=np.zeros(len(ps))
maxp=0
for rank,index in enumerate(order):
    maxp=max(maxp,min(1,(len(ps)-rank)*ps[index]))
    adjusted[index]=maxp
result['holm_adjusted_p']=adjusted
result.to_csv(ROOT/'h1_e2_paired_bootstrap.csv',index=False)
pd.DataFrame({'model':list(point),'pr_auc':[point[c] for c in point]}).to_csv(ROOT/'h1_e2_reference_scores.csv',index=False)
(ROOT/'h1_bootstrap_metadata.json').write_text(json.dumps({'bootstrap_samples_requested':N_BOOT,'seed':SEED,'unit':'commit_sha','e2_rows':len(y),'e2_commits':len(unique),'failures':int(y.sum()),'method':'paired commit-cluster bootstrap, percentile CI; centered two-sided bootstrap p; Holm across eight planned comparisons'},indent=2))
print(result.to_string(index=False),flush=True)
print('Saved:',ROOT/'h1_e2_paired_bootstrap.csv')
