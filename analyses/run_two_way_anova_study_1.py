import pandas as pd
import pingouin as pg
import math, os, pickle
import numpy as np
from scipy import stats

useIBT = 'noIBT'
rel_thr = 0.15
red_thr = 0.70

def parse_combo(combo_str):
    mapping = {
        "pos":                              (("Position",),               ("Linear",)),
        "quat":                             (("Position",),               ("Rotational",)),
        "pos_quat":                         (("Position",),               ("Linear", "Rotational")),
        "linvel":                           (("Velocity",),               ("Linear",)),
        "angvel":                           (("Velocity",),               ("Rotational",)),
        "linvel_angvel":                    (("Velocity",),               ("Linear", "Rotational")),
        "pos_linvel":                       (("Position", "Velocity"),    ("Linear",)),
        "quat_angvel":                      (("Position", "Velocity"),    ("Rotational",)),
        "pos_quat_linvel_angvel":           (("Position", "Velocity"),    ("Linear", "Rotational")),
    }
    try:
        return mapping[combo_str]
    except KeyError:
        raise ValueError(f"Unknown combo string: {combo_str}")

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
ROOT_DIR = f"./results/study_1_{useIBT}_rel_{rel_thr}_red_{red_thr}"
PATTERN  = f"results_relevance_{rel_thr}_redundancy_{red_thr}_study_1_{useIBT}"

# ── 2. LOAD BEST RUNS ─────────────────────────────────────────────────────────
dirpath = os.path.join(ROOT_DIR, PATTERN)
files = [f for f in os.listdir(dirpath) if f.endswith("_nested_metrics.pkl")]

results_dict = {}
for fname in files:
    combo = fname.replace(f"_{useIBT}_nested_metrics.pkl", "").replace("Head_LeftHand_RightHand_", "")
    combo = parse_combo(combo)
    with open(os.path.join(dirpath, fname), "rb") as f:
        run = pickle.load(f)
    results_dict[combo] = run

# ── 3. LOAD SCORE CALCULATE IMBALANCED SCORE ──────────────────────────────────
with open("/srv/STP/experimentData/step_score.pkl", "rb") as f:
    score_df = pickle.load(f)

score_df_A = score_df[score_df['task_id'] == "A"]
score_df_B = score_df[score_df['task_id'] == "B"]
class_presences_A = score_df_A['score'].apply(lambda x: 0 if x > 0 else 1).value_counts().to_dict()
class_presences_B = score_df_B['score'].apply(lambda x: 0 if x > 0 else 1).value_counts().to_dict()

for key, value in class_presences_A.items():
    class_presences_A[key] = 1 / (value / len(score_df_A))
for key, value in class_presences_B.items():
    class_presences_B[key] = 1 / (value / len(score_df_B))

class_presences_A = {0: 0, 1: 1}
class_presences_B = {0: 0, 1: 1}

# ── 3. BUILD LONG TABLE OF 0/1 CORRECTNESS ───────────────────────────────────
rows = []
for combo, best_run in results_dict.items():
    motions, spatials = combo
    for fold in best_run["outer_results"]:
        pids   = np.unique(fold["PID_test"])
        pids   = np.concatenate([pids, pids])
        true_A, pred_A = fold["true_A"], fold["pred_A"]
        true_B, pred_B = fold["true_B"], fold["pred_B"]

        true_A = [class_presences_A[int(i)] for i in true_A]
        true_B = [class_presences_B[int(i)] for i in true_B]
        pred_A = [class_presences_A[int(i)] for i in pred_A]
        pred_B = [class_presences_B[int(i)] for i in pred_B]
        
        truths = np.concatenate([true_A, true_B])
        preds  = np.concatenate([pred_A, pred_B])
        for pid, y, yh in zip(pids, truths, preds):
            score = 1 if math.isclose(y, yh) else 0
            score *= y

            rows.append({
                "participant_id": pid,
                "motion_combo":   "_".join(motions),
                "spatial_combo":  "_".join(spatials),
                "score":        score
            })

df = pd.DataFrame(rows)

# ── 4. AGGREGATE TO ONE BINARY PER TRIPLE ────────────────────────────────────
long = (
    df
    .groupby(["participant_id", "motion_combo", "spatial_combo"], as_index=False)
    ["score"]
    .mean()
)
print("Tidy head:\n", long.head())

aov = pg.rm_anova(
        dv="score",
        within=["motion_combo", "spatial_combo"],
        subject="participant_id",
        data=long,
        detailed=True,
        effsize="np2")       # partial-η²
print("\nRepeated-measures two-way ANOVA")
print(aov.round(4))

# ------------------------------------------------------------------
# 4.  Optional post-hoc contrasts (Holm-corrected) -----------------------------
#     – differences between motion classes, collapsed over spatial types
# ------------------------------------------------------------------
p_motion = (pg.pairwise_tests(dv="score",
                              within="motion_combo",
                              subject="participant_id",
                              padjust="bonf",
                              data=long)
            .loc[:, ["A", "B", "T", "dof", "p-unc", "p-corr"]]
            .round(4))
print("\nMotion post-hocs (Holm-corrected)")
print(p_motion.query("`p-corr` < 0.05"))

# same idea for spatial-combo contrasts:
p_spatial = (pg.pairwise_tests(dv="score",
                              within="spatial_combo",
                              subject="participant_id",
                              padjust="bonf",
                              data=long)
            .loc[:, ["A", "B", "T", "dof", "p-unc", "p-corr"]]
            .round(4))
print("\nSpatial post-hocs (Holm-corrected)")
print(p_spatial.query("`p-corr` < 0.05"))

aov.to_csv(f"study_1_two_way_anova_results_{useIBT}_rel_{rel_thr}_red_{red_thr}.csv", index=False)
p_motion.to_csv(f"study_1_motion_post_hocs_{useIBT}_rel_{rel_thr}_red_{red_thr}.csv", index=False)
p_spatial.to_csv(f"study_1_spatial_post_hocs_{useIBT}_rel_{rel_thr}_red_{red_thr}.csv", index=False)