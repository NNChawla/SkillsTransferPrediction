import os, pickle
import numpy as np
import pandas as pd
from scipy import stats

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
ROOT_DIR = "./results/central_hypopt_shift"
PATTERN  = "results_pca_001-100C_05-1K_07P_07SP_gt02P_450_1.0"

# ── 2. LOAD BEST RUNS ─────────────────────────────────────────────────────────
dirpath = os.path.join(ROOT_DIR, PATTERN)
files = [f for f in os.listdir(dirpath) if f.endswith("_final_results.pkl")]

def parse_combo(combo_str):
    """
    Parse a device-motion combination string into separate device and motion tuples.
    
    Args:
        combo_str (str): String in format 'Device1_Device2_motion1_motion2'
        
    Returns:
        tuple: (devices_tuple, motions_tuple)
    """
    # Split the string by underscores
    parts = combo_str.split('_')
    
    # Known devices and motions
    devices = {'LeftHand', 'RightHand', 'Head'}
    motions = {'linvel', 'linacc', 'angvel', 'angacc'}
    
    # Separate devices and motions
    device_parts = []
    motion_parts = []
    
    for part in parts:
        if part in devices:
            device_parts.append(part)
        elif part in motions:
            motion_parts.append(part)
    
    return (tuple(device_parts), tuple(motion_parts))

results_dict = {}
for fname in files:
    combo = fname.replace("_final_results.pkl", "")\
                 .replace("450_1.0_Step_", "")
    combo = parse_combo(combo)
    with open(os.path.join(dirpath, fname), "rb") as f:
        runs = pickle.load(f)
    # pick run with highest MCC
    results_dict[combo] = max(runs, key=lambda x: x["mcc"])

# ── 3. BUILD LONG TABLE OF 0/1 CORRECTNESS ───────────────────────────────────
rows = []
for combo, best_run in results_dict.items():
    devices, motions = combo
    for fold_res in best_run["kfold_results"]:
        pids   = np.unique(fold_res["PID_test"])
        pids   = np.concatenate([pids, pids])
        true_A, pred_A = fold_res["true_A"], fold_res["pred_A"]
        true_B, pred_B = fold_res["true_B"], fold_res["pred_B"]
        truths = np.concatenate([true_A, true_B])
        preds  = np.concatenate([pred_A, pred_B])
        for pid, y, yh in zip(pids, truths, preds):
            rows.append({
                "participant_id": pid,
                "device_combo":   "_".join(devices),
                "motion_combo":   "_".join(motions),
                "correct":        int(y == yh)
            })

df = pd.DataFrame(rows)

# ── 4. AGGREGATE TO ONE BINARY PER TRIPLE ────────────────────────────────────
agg = (
    df
    .groupby(["participant_id", "device_combo", "motion_combo"], as_index=False)
    ["correct"]
    .mean()
)
agg["correct_bin"] = (agg["correct"] >= 0.5).astype(int)

# ── 5. PREPARE DEVICE‐WIDE AND MOTION‐WIDE TABLES ────────────────────────────
# 5a) Device: collapse across motion_combo
dev_df = (
    agg
    .groupby(["participant_id", "device_combo"], as_index=False)["correct_bin"]
    .mean()
)
dev_df["correct_dev"] = (dev_df["correct_bin"] >= 0.5).astype(int)
dev_wide = dev_df.pivot(index="participant_id",
                        columns="device_combo",
                        values="correct_dev")

# 5b) Motion: collapse across device_combo
mot_df = (
    agg
    .groupby(["participant_id", "motion_combo"], as_index=False)["correct_bin"]
    .mean()
)
mot_df["correct_mot"] = (mot_df["correct_bin"] >= 0.5).astype(int)
mot_wide = mot_df.pivot(index="participant_id",
                        columns="motion_combo",
                        values="correct_mot")

# ── 6. COCHRAN’S Q FUNCTION ─────────────────────────────────────────────────
def cochran_q(wide_df):
    w = wide_df.dropna().astype(int)
    N, k = w.shape
    Tj = w.sum(axis=0).values
    Si = w.sum(axis=1).values
    T  = Tj.sum()
    num = (k - 1) * (k * np.sum(Tj**2) - T**2)
    den = k * T - np.sum(Si**2)
    Q = num / den
    p = stats.chi2.sf(Q, k - 1)
    return Q, p

# ── 7. RUN AND PRINT TESTS ───────────────────────────────────────────────────
Q_dev, p_dev = cochran_q(dev_wide)
print(f"Device‐combo Cochran’s Q({dev_wide.shape[1]-1}) = {Q_dev:.2f}, p = {p_dev:.4f}")

Q_mot, p_mot = cochran_q(mot_wide)
print(f"Motion‐combo Cochran’s Q({mot_wide.shape[1]-1}) = {Q_mot:.2f}, p = {p_mot:.4f}")

# ------------------------------------------------------------------
# Pairwise McNemar with safe handling of n01+n10 == 0
# ------------------------------------------------------------------
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import itertools

motion_cols = mot_wide.columns.tolist()
records = []

for a, b in itertools.combinations(motion_cols, 2):
    ab = mot_wide[[a, b]].dropna().astype(int)
    tbl = pd.crosstab(ab[a], ab[b]).reindex(index=[0, 1],
                                            columns=[0, 1],
                                            fill_value=0)
    n01, n10 = int(tbl.loc[0, 1]), int(tbl.loc[1, 0])

    if n01 + n10 == 0:                 # identical performance
        chi2, p_unc = np.nan, 1.0      # or just skip these pairs
    else:
        res = mcnemar(tbl, exact=False, correction=True)
        chi2, p_unc = res.statistic, res.pvalue

    records.append((a, b, n01, n10, chi2, p_unc))

mc_df = pd.DataFrame(records,
                     columns=["A", "B", "n01", "n10", "chi2", "p_unc"])

# Holm FWER correction
mc_df["p_corr"] = multipletests(mc_df["p_unc"], method="holm")[1]

sig = mc_df.query("p_corr < 0.05")

print("\nHolm-corrected McNemar contrasts (motion combos)")
print(sig[["A", "B", "n01", "n10", "chi2", "p_corr"]])

# save if you like
mc_df.to_csv("motion_mcnemar_all.csv", index=False)
sig.to_csv("motion_mcnemar_SIG.csv", index=False)