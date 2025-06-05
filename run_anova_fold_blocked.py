#!/usr/bin/env python3
# anova_fold_blocked.py
# -------------------------------------------------------------
# Requirements:
#   pip install pandas pingouin statsmodels seaborn matplotlib

import pickle             # change to json / csv loader if needed
import pandas as pd
import numpy as np
import pingouin as pg
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------
# 1. LOAD outer-fold MCC results
# -------------------------------------------------------------
INPUT = Path("anova_data.pkl")          # <— UPDATE TO YOUR FILE
with INPUT.open("rb") as f:
    # dict { (device_tuple, motion_tuple) : [mcc_fold0 … mcc_foldN] }
    results = pickle.load(f)

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

# -------------------------------------------------------------
# 3. Group means + 95 % CIs and error-bar plot
# -------------------------------------------------------------
means = (df.groupby(["device_combo", "motion_combo"])
           .agg(mean=("score", "mean"),
                sd=("score", "std"),
                n=("score", "size")))
means["sem"] = means["sd"] / np.sqrt(means["n"])
means["ci95"] = 1.96 * means["sem"]
print("\nTop 10 combos by mean MCC:")
print(means.sort_values("mean", ascending=False).head(10))

# 95 %-CI plot for motion combos (collapsed over devices)
plt.figure(figsize=(10, 4))
sns.pointplot(data=df,
              x="motion_combo",
              y="score",
              ci=95,
              capsize=.1)
plt.title("Mean MCC ± 95 % CI by motion combo (all devices pooled)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("ci_plot_motion.png", dpi=300)
plt.close()
print("Saved 95-CI plot ➜ ci_plot_motion.png")

# -------------------------------------------------------------
# 4. Repeated-measures two-way ANOVA  (fold = subject)
# -------------------------------------------------------------
rm_aov = pg.rm_anova(dv="score",
                     within=["device_combo", "motion_combo"],
                     subject="fold",
                     data=df,
                     detailed=True,
                     effsize="np2")   # partial-η²
print("\nRepeated-measures two-way ANOVA (Type-II SS):")
print(rm_aov.round(4))

# -------------------------------------------------------------
# 5. Alternative: linear mixed-effects model  (optional)
# -------------------------------------------------------------
# mixed = smf.mixedlm("score ~ C(device_combo)*C(motion_combo)",
#                     data=df,
#                     groups=df["fold"]).fit(reml=False)

# print("\nMixed-effects model (fixed-effect Wald tests):")
# print(anova_lm(mixed, typ=2).round(4))

# -------------------------------------------------------------
# 6. Save ANOVA table
# -------------------------------------------------------------
rm_aov.to_csv("rm_anova_table.csv", index=False)
print("Saved rm_anova_table.csv")
