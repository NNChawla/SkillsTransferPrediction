import pandas as pd
import os
import pickle
import numpy as np
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import warnings

# --- SETUP ---
# Suppress convergence warnings, which can be common during model fitting
warnings.filterwarnings("ignore", category=UserWarning)

useIBT = 'noIBT'
rel_thr = 0.15
red_thr = 0.70

def parse_combo(combo_str: str):
    mapping = {
        "pos":                              (("Position",),               ("Linear",)),
        "quat":                             (("Position",),               ("Angular",)),
        "pos_quat":                         (("Position",),               ("Linear", "Angular")),
        "linvel":                           (("Velocity",),               ("Linear",)),
        "angvel":                           (("Velocity",),               ("Angular",)),
        "linvel_angvel":                    (("Velocity",),               ("Linear", "Angular")),
        "linacc":                           (("Acceleration",),           ("Linear",)),
        "angacc":                           (("Acceleration",),           ("Angular",)),
        "linacc_angacc":                    (("Acceleration",),           ("Linear", "Angular")),
        "pos_linvel":                       (("Position", "Velocity"),    ("Linear",)),
        "quat_angvel":                      (("Position", "Velocity"),    ("Angular",)),
        "pos_quat_linvel_angvel":           (("Position", "Velocity"),    ("Linear", "Angular")),
        "pos_linacc":                       (("Position", "Acceleration"), ("Linear",)),
        "quat_angacc":                      (("Position", "Acceleration"), ("Angular",)),
        "pos_quat_linacc_angacc":           (("Position", "Acceleration"), ("Linear", "Angular")),
        "linvel_linacc":                    (("Velocity", "Acceleration"), ("Linear",)),
        "angvel_angacc":                    (("Velocity", "Acceleration"), ("Angular",)),
        "linvel_angvel_linacc_angacc":      (("Velocity", "Acceleration"), ("Linear", "Angular")),
        "pos_linvel_linacc":                (("Position", "Velocity", "Acceleration"), ("Linear",)),
        "quat_angvel_angacc":               (("Position", "Velocity", "Acceleration"), ("Angular",)),
        "pos_quat_linvel_angvel_linacc_angacc": (("Position", "Velocity", "Acceleration"), ("Linear", "Angular")),
    }
    try:
        return mapping[combo_str]
    except KeyError:
        raise ValueError(f"Unknown combo string: {combo_str}")

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
ROOT_DIR = f"./results/study_2_{useIBT}_rel_{rel_thr}_red_{red_thr}"
PATTERN  = f"results_relevance_{rel_thr}_redundancy_{red_thr}_study_2_{useIBT}"

# ── 2. LOAD BEST RUNS ─────────────────────────────────────────────────────────

dirpath = os.path.join(ROOT_DIR, PATTERN)
# Check if directory exists to prevent errors
if not os.path.isdir(dirpath):
    raise FileNotFoundError(f"The specified directory does not exist: {dirpath}")

files = [f for f in os.listdir(dirpath) if f.endswith("_final_results.pkl")]

results_dict = {}
for fname in files:
    combo = fname.replace(f"_{useIBT}_final_results.pkl", "")\
                 .replace("Head_LeftHand_RightHand_", "")
    combo = parse_combo(combo)
    with open(os.path.join(dirpath, fname), "rb") as f:
        runs = pickle.load(f)
    # pick run with highest MCC
    results_dict[combo] = max(runs, key=lambda x: x["mcc"])

# --- 2. BUILD THE TRIAL-LEVEL LONG DATAFRAME ---
# This is the crucial part. We do NOT average the results.
# We add the true participant class ('y') for each trial to the dataframe.
rows = []
for combo, best_run in results_dict.items():
    motions, spatials = combo
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
                "motion_combo":   "_".join(motions),
                "spatial_combo":  "_".join(spatials),
                "participant_class": int(y), # The TRUE class label for the trial
                "correct":        int(y == yh) # Whether the prediction was correct
            })

df = pd.DataFrame(rows)

# Convert all relevant columns to categorical type for statsmodels
for col in ["participant_id", "motion_combo", "spatial_combo", "participant_class"]:
    df[col] = df[col].astype("category")

print("--- Tidy trial-level dataframe head ---")
print(df.head())

# --- 3. DEFINE AND FIT THE GENERALIZED LINEAR MIXED-EFFECTS MODEL (GLMM) ---
# This section replaces the ANOVA.
# We model 'correct' as a function of our predictors, including the participant's true class.
# The interaction with 'participant_class' is key to understanding if a model is only
# good for the majority or minority class.

# Define the model formula
formula = "correct ~ C(motion_combo) * C(spatial_combo)"#  * C(participant_class)"

# Define the random effects structure (a random intercept for each participant)
vc_formula = {"participant": "0 + C(participant_id)"}

print("\n--- Fitting GLMM ---")
print(f"Formula: {formula}")

# Initialize and fit the Bayesian GLMM
# This is well-suited for complex models that might have convergence issues in frequentist frameworks
glmm_model = BinomialBayesMixedGLM.from_formula(
    formula,
    vc_formula,
    data=df
)

# Fit the model using Variational Bayes, which is efficient
glmm_results = glmm_model.fit_vb()

# --- 4. DISPLAY AND SAVE RESULTS ---
print("\n--- GLMM Results Summary ---")
print(glmm_results.summary())

# Save the full results summary to a text file for later review
with open(f"glmm_study_2_results_summary_{useIBT}.txt", "w") as f:
    f.write(str(glmm_results.summary()))

# Save the model object itself
with open(f"glmm_study_2_model_{useIBT}.pkl", "wb") as f:
    pickle.dump(glmm_results, f)

print(f"\nAnalysis complete. Results saved to glmm_study_2_results_summary_{useIBT}.txt")

# # --- HOW TO INTERPRET THE RESULTS ---
# print("""
# --- How to Interpret the GLMM Output ---
# 1.  Look at the 'Post. Mean' for each term. This is the estimated effect size. A positive value means it increases the log-odds of a correct prediction.
# 2.  Check the 95% credible interval (approximated by 'Post. Mean' +/- 1.96 * 'Post. SD'). If this interval does NOT contain 0, the effect is statistically significant.
# 3.  **Crucial Interaction Term**: Look for the three-way interaction, e.g., 'C(motion_combo)[T.Position_Velocity]:C(spatial_combo)[T.Linear_Angular]:C(participant_class)[T.1]'.
#     * If this term is significant and positive, it means that specific model combination is **especially good** at classifying the minority class (Class 1). THIS IS A GREAT RESULT!
#     * If this term is significant and negative, it means that model is good for the majority class but performs **worse** on the minority class than you would otherwise expect. This indicates a model that hasn't properly learned to distinguish the classes.
# """)