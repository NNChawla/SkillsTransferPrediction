import pickle, re, gc, time, pandas as pd

with open("./experimentData/std_findiff_tabulated_dataframe.pkl", "rb") as f:
    sfdf = pickle.load(f)

selected_features = sfdf.columns.to_list()
selected_features = [i for i in selected_features if ("pause" in i)]

start_time = time.time()

for window in [9, 271, 361, 451]:
    for threshold in [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
        sub_sf = [i for i in selected_features if ((f"_{window}_" in i) and (f"_{threshold}_" in i))]
        ndf = sfdf[["PID"] + sub_sf]
        value_vars = [c for c in ndf.columns if c != "PID"]

        t = time.time()
        print(f"Processing {window} {threshold}")
        df_long = (
            ndf
            .melt(id_vars="PID",
                    value_vars=value_vars,
                    var_name="raw_feature",          # temporarily holds the long col-name
                    value_name="value")
        )
        print(f"Long df created: {time.time() - t}")

        t = time.time()
        print(f"Grouping {window} {threshold}")
        pat = re.compile(r"step(?P<step_id>\d+)_(?P<feat>.+)_(?P<task_id>[AB])$")
        df_long[["step_id", "feature", "task_id"]] = (
            df_long["raw_feature"]
            .apply(lambda s: pd.Series(pat.match(s).groupdict()))
        )
        df_long["step_id"]       = df_long["step_id"].astype(int)
        df_long["task_id"]       = df_long["task_id"].astype("category")
        df_long.drop(columns="raw_feature", inplace=True)
        print(f"Grouping done: {time.time() - t}")

        t = time.time()
        print(f"Pivoting {window} {threshold}")
        df_tidy = (
            df_long
            .pivot_table(index=["PID", "task_id", "step_id"],
                        columns="feature",
                        values="value")
            .reset_index()
            .sort_values(["PID", "task_id", "step_id"])
        )
        print(f"Pivoting done: {time.time() - t}")

        t = time.time()
        print(f"Saving {window} {threshold}")
        with open(f"./experimentData/step_df_{window}_{threshold}.pkl", "wb") as f:
            pickle.dump(df_tidy, f)
        print(f"Saving done: {time.time() - t}")

        del df_tidy, df_long, ndf
        gc.collect()

print(f"Total time taken: {time.time() - start_time}")

# import pickle, pandas as pd

# combined_df = None

# for window in [9, 271, 361, 451]:
#     for threshold in [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
#         print(f"Processing window {window} and threshold {threshold}")
#         with open(f"./experimentData/step_df_{window}_{threshold}.pkl", "rb") as f:
#             step_0_11_df = pickle.load(f)
#         with open(f"./experimentData/substep_df_{window}_{threshold}.pkl", "rb") as f:
#             step_12_df = pickle.load(f)
#         step_0_12_df = pd.merge(step_0_11_df, step_12_df, on=step_0_11_df.columns.to_list(), how="outer")
#         if combined_df is None:
#             combined_df = step_0_12_df
#         else:
#             combined_df = pd.merge(combined_df, step_0_12_df, on=["PID", "task_id", "step_id"], how="outer")

