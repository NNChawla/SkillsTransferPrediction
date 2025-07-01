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
df = df[~df.motion_combo.str.contains("Acceleration")]