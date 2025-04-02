import pickle, optuna, os
feature_levels = ["session"]
feature_objects = ["RightHand"]
feature_types = ["quat"]
feature_subtypes = ["inter_extrema_intervals_base", "inter_extrema_intervals_low",
                      "inter_extrema_intervals_mid", "inter_extrema_intervals_high"]
all_subset_feature_terms = []
for feature_level in feature_levels:
    for feature_object in feature_objects:
        for feature_type in feature_types:
            for feature_subtype in feature_subtypes:
                all_subset_feature_terms.append([feature_level, feature_object, feature_type, feature_subtype])

train_sets = ["A", "B"]
cross_tasks = [True]
feature_selections_models = [("rfe", "linsvc")]
clf_funcs = [lambda x: 1 if x > 0 else 0]

labels_quat_linsvc = []
for subset_feature_terms in all_subset_feature_terms:
    for train_set in train_sets:
        for cross_task in cross_tasks:
            for feature_selection, model_name in feature_selections_models:
                for clf_func in clf_funcs:
                    is_cross_task = "cross_task" if cross_task else "within_task"
                    label = f"{'_'.join(subset_feature_terms)}_{feature_selection}_{model_name}_{train_set}_{is_cross_task}"
                    labels_quat_linsvc.append(label)

quat_linsvc = {}
for i in labels_quat_linsvc:
    with open(f"./results/{i}_avg_metrics.pkl", "rb") as f:
        quat_linsvc[i] = pickle.load(f)

for i, j in quat_linsvc.items():
    print(f"{i}: {j['feature_selection_measure']}")

studies_quat_linsvc = {}
for i in labels_quat_linsvc:
    with open(f"./results/{i}_study.pkl", "rb") as f:
        studies_quat_linsvc[i] = pickle.load(f)

for i, j in studies_quat_linsvc.items():
    print(f"{i}: {j.best_trial.params}")

optuna.visualization.plot_contour(studies_quat_linsvc[labels_quat_linsvc[0]])
optuna.visualization.plot_contour(studies_quat_linsvc[labels_quat_linsvc[0]], params=["model__C", "fs__n_features_to_select"])

axis_linsvc_studies = [i for i in os.listdir('/srv/STP/results') if ('study' in i) and ('multiobjective' in i) and ('linsvc' in i)]
for i in axis_linsvc_studies:
    print(i)
    with open(f'/srv/STP/results/{i}', 'rb') as f:
        study = pickle.load(f)
        diffs = [i.values[1] - i.values[0] for i in study.best_trials]
        print(diffs)
        avgs = [(i.values[0] + i.values[1])/2 for i in study.best_trials]
        print(avgs)
        print([i.values for i in study.best_trials])

axis_linsvc = [i for i in os.listdir('/srv/STP/results') if ('metrics' in i) and ('multiobjective' in i) and ('linsvc' in i)]
for i in axis_linsvc:
    print(i)
    with open(f'/srv/STP/results/{i}', 'rb') as f:
        print(pickle.load(f)['feature_selection_measure'])
    
for i in axis_linsvc:
    print(i)
    with open(f'/srv/STP/results/{i}', 'rb') as f:
        print(pickle.load(f)['feature_selection_measure'])