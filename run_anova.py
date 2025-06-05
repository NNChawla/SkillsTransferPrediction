# two_way_anova_mcc.py
# --------------------------------------------------------------
import itertools
import pickle                         # or json / csv
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --------------------------------------------------------------
# 1. LOAD the nested-CV results
# --------------------------------------------------------------
with open("./anova_data.pkl", "rb") as f:           # <— UPDATE PATH
    results = pickle.load(f)                        # { (devs, mots): [mccs] }

# sanity check
n_folds = {len(v) for v in results.values()}
assert len(n_folds) == 1, "All combos must have the same #folds"
n_folds = n_folds.pop()

# --------------------------------------------------------------
# 2. tidy dataframe  (one row = one outer-fold replicate)
# --------------------------------------------------------------
rows = []
for (devs, mots), fold_scores in results.items():
    dev_name = "+".join(sorted(devs))               # e.g. "Head+LeftHand"
    mot_name = "+".join(sorted(mots))               # e.g. "LinVel+AngAcc"
    for fold_idx, mcc in enumerate(fold_scores):
        rows.append({
            "device_combo": dev_name,
            "motion_combo": mot_name,
            "fold": fold_idx,
            "score": mcc,
        })

df = pd.DataFrame(rows)
df["device_combo"] = df["device_combo"].astype("category")
df["motion_combo"] = df["motion_combo"].astype("category")

print(f"Tidy table: {df.shape[0]} rows, "
      f"{df['device_combo'].nunique()} device levels × "
      f"{df['motion_combo'].nunique()} motion levels")

# --------------------------------------------------------------
# 3. Two–way factorial ANOVA
#    score  ~  device + motion + device:motion
# --------------------------------------------------------------
formula = "score ~ C(device_combo) + C(motion_combo) + C(device_combo):C(motion_combo)"
model   = smf.ols(formula, data=df).fit()

anova   = anova_lm(model, typ=2)          # Type-II SS
print("\nANOVA table (Type-II):")
print(anova)

# --------------------------------------------------------------
# 4. OPTIONAL: post-hoc Tukey HSD on main effects
# --------------------------------------------------------------
print("\nPost-hoc Tukey — device combos")
print(pairwise_tukeyhsd(endog=df["score"],
                        groups=df["device_combo"],
                        alpha=0.05))

print("\nPost-hoc Tukey — motion combos")
print(pairwise_tukeyhsd(endog=df["score"],
                        groups=df["motion_combo"],
                        alpha=0.05))

# --------------------------------------------------------------
# 5. Save ANOVA table for the report
# --------------------------------------------------------------

anova["eta_sq"] = anova["sum_sq"] / (anova["sum_sq"].sum() + model.ssr)
print(anova.loc["C(motion_combo)","eta_sq"])

anova.to_csv("./anova_table.csv")
print("\nSaved anova_table.csv")