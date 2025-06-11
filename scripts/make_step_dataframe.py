import pandas as pd

# ------------------------------------------------------------------
# 0) Load the table you showed in the *first* screenshot
# ------------------------------------------------------------------
import pickle

with open("/srv/STP/shift450_tabulated_dataframe.pkl", "rb") as f:
    df_step = pickle.load(f)

# ------------------------------------------------------------------
# 1) Melt: columns  ➜  rows  (long format)
# ------------------------------------------------------------------
long = (
    df_step
      .melt(id_vars="PID", var_name="raw_feature", value_name="value")
)

# raw_feature looks like  "step0_Head_linvel_x_91_pause_0.25_duration_count_A"

# ------------------------------------------------------------------
# 2) Split off the trailing "_A" / "_B"
# ------------------------------------------------------------------
long[["feature", "task_id"]] = (
    long["raw_feature"]
        .str.rsplit("_", n=1, expand=True)   # split **once** from the right
)

long.drop(columns="raw_feature", inplace=True)

# ------------------------------------------------------------------
# 3) Pivot back to wide: one row ↔ (PID , task_id)
# ------------------------------------------------------------------
wide = (
    long
      .pivot_table(index=["PID", "task_id"],
                   columns="feature",
                   values="value")
      .reset_index()
      .sort_values(["PID", "task_id"])
)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
PID = wide["PID"]
task_id = wide["task_id"]
wide = wide.drop(columns=["PID", "task_id"])
features = [i for i in wide.columns.to_list() if "_pause_" in i]
wide = wide[features]
wide = pd.DataFrame(imputer.fit_transform(wide), columns=features)
wide.insert(0, "PID", PID)
wide.insert(1, "task_id", task_id)

with open("shift450_central_step_df.pkl", "wb") as f:
    pickle.dump(wide, f)

# The DataFrame now matches the second screenshot:
# • 210 rows  (105 PIDs × 2 task_id)
# • ~297 026 feature columns with names like
#   "step0_Head_angacc_magnitude_271_pause_0.25_duration_count"
#   (no trailing _A / _B, because task_id is its own column)