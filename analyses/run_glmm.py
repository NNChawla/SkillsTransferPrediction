import os
import pickle
from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import warnings

# Suppress potential warnings from statsmodels during fitting
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- 1. CONFIGURATION ---
# These settings remain the same.
ROOT_DIR = "./results/central_hypopt_shift"
PATTERN  = "results_pca_001-100C_05-1K_07P_07SP_gt02P_450_1.0"

# --- 2. LOAD AND PARSE BEST RUNS ---
# This section for loading data is also unchanged.
dirpath = os.path.join(ROOT_DIR, PATTERN)
# A check to ensure the directory exists
if not os.path.isdir(dirpath):
    raise ValueError(f"Error: Directory not found at {dirpath}")
    print(f"Error: Directory not found at {dirpath}")
    # Create dummy data to allow the script to run for demonstration
    print("Creating dummy data for demonstration purposes.")
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # This part is just to make the script runnable if the data doesn't exist.
    # In your actual use, the real data files should be present.
    dummy_files = [
        "450_1.0_Step_Head_linvel_final_results.pkl",
        "450_1.0_Step_LeftHand_RightHand_angacc_final_results.pkl",
        "450_1.0_Step_Head_linacc_final_results.pkl",
    ]
    for fname in dummy_files:
        with open(os.path.join(dirpath, fname), "wb") as f:
            dummy_run = {
                "mcc": np.random.rand(),
                "kfold_results": [{
                    "PID_test": np.arange(1, 6),
                    "true_A": np.random.randint(0, 2, 5), "pred_A": np.random.randint(0, 2, 5),
                    "true_B": np.random.randint(0, 2, 5), "pred_B": np.random.randint(0, 2, 5),
                }]
            }
            pickle.dump([dummy_run], f)

files = [f for f in os.listdir(dirpath) if f.endswith("_final_results.pkl")]

def parse_combo(combo_str):
    """Parses a device-motion string into separate device and motion parts."""
    parts = combo_str.split('_')
    devices = {'LeftHand', 'RightHand', 'Head'}
    motions = {'linvel', 'linacc', 'angvel', 'angacc'}
    device_parts = tuple(sorted([p for p in parts if p in devices]))
    motion_parts = tuple(sorted([p for p in parts if p in motions]))
    return (device_parts, motion_parts)

results_dict = {}
for fname in files:
    combo_str = fname.replace("_final_results.pkl", "").replace("450_1.0_Step_", "")
    combo_key = parse_combo(combo_str)
    with open(os.path.join(dirpath, fname), "rb") as f:
        runs = pickle.load(f)
    results_dict[combo_key] = max(runs, key=lambda x: x["mcc"])

print(len(results_dict))

# --- 3. BUILD LONG TABLE WITH BINARY INDICATORS ---
# This section is updated to create binary columns for each device/motion type.
rows = []
for combo, best_run in results_dict.items():
    devices, motions = combo
    # Use all k-fold results for a complete dataset
    for fold_res in best_run["kfold_results"]:
        # Ensure correct participant ID mapping for two trials
        pids_base = np.unique(fold_res["PID_test"])
        pids = np.concatenate([pids_base, pids_base])

        truths = np.concatenate([fold_res["true_A"], fold_res["true_B"]])
        preds  = np.concatenate([fold_res["pred_A"], fold_res["pred_B"]])

        for pid, y, yh in zip(pids, truths, preds):
            rows.append({
                "participant_id": f"P{pid}", # Make it a categorical-friendly string
                "device_combo": "_".join(devices),
                "motion_combo": "_".join(motions),
                "correct": int(y == yh)
            })

df = pd.DataFrame(rows)

# Create binary indicator variables from the combo strings
# This is the key change for the new model formula
all_devices = ['Head', 'LeftHand', 'RightHand']
all_motions = ['linvel', 'linacc', 'angvel', 'angacc']

for device in all_devices:
    df[f'Has_{device}'] = df['device_combo'].str.contains(device).astype(int)

for motion in all_motions:
    df[f'Has_{motion}'] = df['motion_combo'].str.contains(motion).astype(int)

# Convert categorical columns for statsmodels
df["participant_id"] = df["participant_id"].astype("category")
df["device_combo"] = df["device_combo"].astype("category")
df["motion_combo"] = df["motion_combo"].astype("category")

print("--- Dataframe Head with Indicator Variables ---")
print(df.head())
print("\n")


# --- 4. FIT AND INTERPRET THE MAIN GLMM ---
# The formula now uses the specific binary effects and their interactions.
# This model is more interpretable than the previous one.
main_effects_devices = " + ".join([f"Has_{d}" for d in all_devices])
main_effects_motions = " + ".join([f"Has_{m}" for m in all_motions])

formula = f"correct ~ ({main_effects_devices}) * ({main_effects_motions})"
vc_formula = {"participant": "0 + C(participant_id)"} # Random intercept for each participant

print(f"--- Fitting Main GLMM with formula: {formula} ---")
main_model = BinomialBayesMixedGLM.from_formula(
    formula,
    vc_formula,
    df
)
main_results = main_model.fit_vb()

print("\n--- Main GLMM Results Summary ---")
print(main_results.summary())
print("\nInterpretation Note: A factor or interaction is significant if its 95% credible interval [0.025, 0.975] does not contain 0.")
print("\n")


# --- 5. POST-HOC CONTRASTS FOR SPECIFIC COMPARISONS ---
# This section replaces the old post-hoc tests and adds robust error handling.
def run_motion_contrast(df, contrast_pair, vc_formula):
    """
    Runs a specific pairwise comparison for motion types by refitting a GLMM.
    This version includes robust error handling.

    Args:
        df (pd.DataFrame): The full analysis dataframe.
        contrast_pair (tuple): A tuple of two motion combo strings to compare.
        vc_formula (dict): The random effects formula for the model.

    Returns:
        pd.DataFrame: A DataFrame with the summary row for the contrast, or an empty one on failure.
    """
    level1, level2 = contrast_pair
    print(f"--- Running Contrast: '{level1}' vs. '{level2}' ---")

    # Create a new DataFrame containing only the data for the two levels in the contrast.
    # This prevents issues from other data and makes the model more stable.
    df_contrast = df[df['motion_combo'].isin([level1, level2])].copy()
    
    # Ensure there's enough data to run the contrast
    if df_contrast.shape[0] < 10 or len(df_contrast['motion_combo'].unique()) < 2:
        print(f"    [WARNING] Insufficient data for contrast '{level1}' vs '{level2}'. Skipping.")
        return pd.DataFrame()

    # Explicitly set the reference level for the comparison
    df_contrast['motion_combo'] = pd.Categorical(df_contrast['motion_combo'], categories=[level2, level1], ordered=True)

    # Use a simpler additive model for clear post-hoc interpretation
    contrast_formula = "correct ~ C(motion_combo) + C(device_combo)"

    try:
        model = BinomialBayesMixedGLM.from_formula(
            contrast_formula,
            vc_formula,
            data=df_contrast
        )
        results = model.fit_vb()
        
        # *** ERROR HANDLING FIX ***
        # Check if the summary has the expected coefficient table.
        # A failed model fit will result in fewer than 2 tables.
        if len(results.summary().tables) < 2:
            print(f"    [WARNING] Model for contrast '{level1}' vs '{level2}' failed to converge or produce a valid summary. Skipping.")
            return pd.DataFrame()  # Return an empty DataFrame on failure

        summary_df = results.summary().tables[1]
        
        # The coefficient for level1 is the contrast against level2 (the reference)
        contrast_row = summary_df[summary_df.index == f"C(motion_combo)[T.{level1}]"]
        
        if contrast_row.empty:
            print(f"    [WARNING] Could not find the specific coefficient for '{level1}' in the summary. Skipping.")
            return pd.DataFrame()

        return contrast_row

    except Exception as e:
        print(f"    [ERROR] A critical exception occurred while fitting contrast '{level1}' vs '{level2}': {e}")
        return pd.DataFrame() # Return an empty DataFrame on any other exception

# --- Main loop for running contrasts ---
motion_combos = df['motion_combo'].cat.categories
motion_contrasts_to_run = list(combinations(motion_combos, 2))

print("\n--- Post-Hoc Pairwise Motion Contrasts ---")
all_contrast_results = []
for pair in motion_contrasts_to_run:
    contrast_res = run_motion_contrast(df, pair, vc_formula)
    
    # *** ERROR HANDLING FIX ***
    # Check if the result is a non-empty DataFrame before trying to process it.
    if not contrast_res.empty:
        ci_low = contrast_res['[0.025'].iloc[0]
        ci_high = contrast_res['0.975]'].iloc[0]
        significant = "Yes" if ci_low * ci_high > 0 else "No" # Significant if CI does not cross zero
        
        all_contrast_results.append({
            "Contrast": f"{pair[0]} vs. {pair[1]}",
            "Coef.": contrast_res['Coef.'].iloc[0],
            "Std.Err.": contrast_res['Std.Err.'].iloc[0],
            "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "Significant": significant
        })

# Display all results in a clean table
if all_contrast_results:
    posthoc_df = pd.DataFrame(all_contrast_results)
    print(posthoc_df.to_string())
else:
    print("No valid contrasts were successfully run. Check your data and model specifications.")


# --- 6. SAVE ARTIFACTS ---
print("\n--- Saving analysis artifacts ---")
# Save the final dataframe with all indicator columns
df.to_csv("glmm_analysis_dataframe.csv", index=False)

# Save the main model results
with open("glmm_main_model_results.pickle", "wb") as f:
    pickle.dump(main_results, f)

# Save the post-hoc results table
if all_contrast_results:
    posthoc_df.to_csv("glmm_posthoc_motion_contrasts.csv", index=False)

print("\nAnalysis complete.")
