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
C_high = "100"
K_low = "05"
K_high = "1"
P = "07"
SP = "07"
gt_score = "02"

window = [450] # [9, 91, 181, 271, 361, 451, 541]
threshold = [1.0] # [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

for w in window:
    for t in threshold:
        dir = f"./results/central_hypopt_shift_notime/results_pca_{C_low}-{C_high}C_{K_low}-{K_high}K_{P}P_{SP}SP_gt{gt_score}P_{w}_{t}"

        files = sorted(os.listdir(dir))

        files = [i for i in files if "final" in i]

        results = []
        for name in files:
            with open(f"{dir}/{name}", "rb") as f:
                results.append((name.replace(f"{w}_{t}_Step_", "").replace("_final_results.pkl", ""), pickle.load(f)))

        results_dict = dict(results)

        for key in results_dict.keys():
            scores = [i['mcc'] for i in results_dict[key]]
            best_idx = scores.index(max(scores))
            results_dict[key] = results_dict[key][best_idx]

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
        assert len(n_folds_set) == 1, "All combos must have the same # outer folds!"
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
        # df.to_csv(f"./window_451_std_0.75_foldData.csv", index=False)

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
        plt.savefig(f"./anova/ci_plot_motion_{C_low}-{C_high}C_{K_low}-{K_high}K_{P}P_{SP}SP_gt{gt_score}P_{w}_{t}.png", dpi=300)
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
        # 6. Save ANOVA table
        # -------------------------------------------------------------
        rm_aov.to_csv(f"./anova/rm_anova_table_{C_low}-{C_high}C_{K_low}-{K_high}K_{P}P_{SP}SP_gt{gt_score}P_{w}_{t}.csv", index=False)
        print("Saved rm_anova_table.csv")

        # -------------------------------------------------------------
        # 5. Post-hoc paired contrasts for MOTION ONLY  (Holm-corrected)
        # -------------------------------------------------------------
        post_motion = pg.pairwise_tests(
            dv='score',
            within='motion_combo',
            subject='fold',
            padjust='holm',          # 'fdr_bh' if you prefer FDR
            effsize='cohen',         # Cohen dz for paired samples
            alternative='two-sided',
            data=df
        )

        sig_motion = post_motion.query('`p-corr` < 0.05')

        print("\nHolm–corrected pairwise comparisons for motion_combo:")
        print(post_motion[['A','B','T','dof','p-unc','p-corr','cohen']].head())

        if sig_motion.empty:
            print("No motion-combo pairs remain significant after correction.")
        else:
            print("\nSignificant pairs (p-corr < 0.05):")
            print(sig_motion[['A','B','T','dof','p-corr','cohen']])

        # -------------------------------------------------------------
        # 5. Post-hoc paired contrasts for MOTION ONLY  (Holm-corrected)
        # -------------------------------------------------------------
        post_device = pg.pairwise_tests(
            dv='score',
            within='device_combo',
            subject='fold',
            padjust='holm',          # 'fdr_bh' if you prefer FDR
            effsize='cohen',         # Cohen dz for paired samples
            alternative='two-sided',
            data=df
        )

        sig_device = post_device.query('`p-corr` < 0.05')

        print("\nHolm–corrected pairwise comparisons for device_combo:")
        print(post_device[['A','B','T','dof','p-unc','p-corr','cohen']].head())

        if sig_device.empty:
            print("No device-combo pairs remain significant after correction.")
        else:
            print("\nSignificant pairs (p-corr < 0.05):")
            print(sig_device[['A','B','T','dof','p-corr','cohen']])

        # -------------------------------------------------------------
        # 6. Save post-hoc tables
        # -------------------------------------------------------------
        fname_base = f"{C_low}-{C_high}C_{K_low}-{K_high}K_{P}P_{SP}SP_gt{gt_score}P_{w}_{t}"
        post_motion.to_csv(f"./anova/pairwise_motion_{fname_base}.csv", index=False)
        sig_motion.to_csv(f"./anova/pairwise_motion_SIG_{fname_base}.csv", index=False)
        post_device.to_csv(f"./anova/pairwise_device_{fname_base}.csv", index=False)
        sig_device.to_csv(f"./anova/pairwise_device_SIG_{fname_base}.csv", index=False)
        print(f"Saved pairwise_motion_{fname_base}.csv and pairwise_device_{fname_base}.csv")