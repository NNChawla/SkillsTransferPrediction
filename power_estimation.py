# Load results
from itertools import combinations
import pickle, os
import pandas as pd
import numpy as np
import pingouin as pg
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

C_low = "001"
C_high = "10"
K_low = "04"
K_high = "1"
P = "07"
SP = "07"
gt_score = "02"
window = "541"
threshold = "1.0"

# -------------  inputs from your ANOVA output -----------------
eta2   = 0.2817451097476204      # partial-η² for motion_combo
eps    = 0.3255009209861151      # GG epsilon  (motion row)
m      = 7          # 15 motion levels (within-subject)
n_subj = 5           # folds = subjects
alpha  = 0.05

dir = f"./results/results_pca_{C_low}-{C_high}C_{K_low}-{K_high}K_{P}P_{SP}SP_gt{gt_score}P_{window}_{threshold}"

files = sorted(os.listdir(dir))

files = [i for i in files if "final" in i]

results = []
for name in files:
    with open(f"{dir}/{name}", "rb") as f:
        results.append((name.replace(f"{window}_{threshold}_Step_", "").replace("_final_results.pkl", ""), pickle.load(f)))

results_dict = dict(results)

for key in results_dict.keys():
    results_dict[key] = results_dict[key][0]

anova_data = {}
for key in results_dict.keys():
    anova_data[key] = [i["mcc"] for i in results_dict[key]["kfold_results"]]

results = {}

device_types = ["Head", "RightHand", "LeftHand"]
motion_types = ["linvel", "linacc", "angvel", "angacc"]

dvc_cmbs = []
for i in range(1, len(device_types)+1):
    dvc_cmbs.extend(list(combinations(device_types, i)))

mt_cmbs = []
for i in range(1, len(motion_types)+1):
    mt_cmbs.extend(list(combinations(motion_types, i)))

all_cmbs = []
for i in dvc_cmbs:
    for j in mt_cmbs:
        all_cmbs.append((i, j))

for cmb in all_cmbs:
    key = "_".join(cmb[0])+"_"+"_".join(cmb[1])
    results[cmb] = anova_data[key]

# guard
n_folds_set = {len(v) for v in results.values()}
assert len(n_folds_set) == 1, "All combos must have the same #outer folds!"
n_folds = n_folds_set.pop()
print(f"Loaded {len(results)} combos × {n_folds} folds ({len(results)*n_folds} rows)")

# -------------------------------------------------------------
# 2. Make tidy long dataframe
# -------------------------------------------------------------
rows = []
for (devs, mots), mcc_vec in results.items():
    dev_name = "+".join(sorted(devs)) or "None"
    mot_name = "+".join(sorted(mots)) or "None"
    for fold, score in enumerate(mcc_vec):
        rows.append({"fold": fold,
                     "device_combo": dev_name,
                     "motion_combo": mot_name,
                     "score": score})
df = pd.DataFrame(rows)
df["device_combo"] = df["device_combo"].astype("category")
df["motion_combo"] = df["motion_combo"].astype("category")

print(f"Tidy table:\n{df.head()}")

# ---------- 1) observed (= post-hoc) power --------------------
# need an estimate of mean correlation among motion scores
# wide = (
#     df.groupby(['fold', 'motion_combo'], as_index=False)['score']
#       .mean()                               # or .median(), .max(), …
#       .pivot(index='fold',
#              columns='motion_combo',
#              values='score')
# )
# corrs = wide.corr().values[np.triu_indices(m, k=1)]
# mean_r = np.tanh(np.arctanh(corrs).mean())   # Fisher-z average
# print(f"mean r among motions ≈ {mean_r:.2f}")

mean_r = 0.1

obs_power = pg.power_rm_anova(eta_squared=eta2,
                              m=m,
                              n=n_subj,
                              alpha=alpha,
                              power=None,
                              corr=mean_r,
                              epsilon=eps)
print(f"Observed power for motion factor = {obs_power:.3f}")

# ---------- 2) sample-size (fold) planning for 80 % power -----
target_power = 0.80
needed_n = pg.power_rm_anova(eta_squared=eta2,
                             m=m,
                             n=None,              # ask for n
                             power=target_power,
                             alpha=alpha,
                             corr=mean_r,
                             epsilon=eps)
print(f"Folds (subjects) needed for 80 % power ≈ {np.ceil(needed_n)}")