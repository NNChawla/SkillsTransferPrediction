rows = []
for combo, best_run in results_dict.items():
    motions, spatials = combo
    rows.append({
        "motion_combo":   "_".join(motions),
        "spatial_combo":  "_".join(spatials),
        "mcc":       best_run['mcc'],
        "mcc_std":   best_run['mcc_std'],
        "accuracy":  best_run['accuracy'],
        "accuracy_std": best_run['accuracy_std'],
        "balanced_accuracy": best_run['balanced_accuracy'],
        "balanced_accuracy_std": best_run['balanced_accuracy_std'],
        "f1":        best_run['f1'],
        "f1_std":    best_run['f1_std'],
        "precision": best_run['precision'],
        "precision_std": best_run['precision_std'],
        "recall":    best_run['recall'],
        "recall_std": best_run['recall_std'],
        "sensitivity": best_run['sensitivity'],
        "sensitivity_std": best_run['sensitivity_std'],
        "specificity": best_run['specificity'],
        "specificity_std": best_run['specificity_std'],
    })