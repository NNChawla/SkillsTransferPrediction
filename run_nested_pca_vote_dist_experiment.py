from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import hmean
from sklearn.feature_selection import VarianceThreshold
import pickle, time, gc, sys, os, json, math, random
from optuna.pruners import MedianPruner
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import optuna

def rel_red_filter(df_wide: pd.DataFrame,
                   feature_cols: list[str],
                   score_col,
                   score_thr: float = 0.20,
                   pearson_thr: float = 0.95,
                   spearman_thr: float = 0.95) -> list[str]:
    """
    1) Keep features whose |Pearson corr| with the score >= score_thr.
    2) From those, drop redundant columns if either
       |Pearson| >= pearson_thr  OR  |Spearman| >= spearman_thr.

    Returns
    -------
    kept_cols : list[str]
    """
    # ---------- 1. relevance ---------------------------------------
    corr_with_score = df_wide[feature_cols].corrwith(
                          score_col, method="pearson").abs()
    relevant_cols   = corr_with_score[corr_with_score >= score_thr].index.tolist()
    print(f"Relevance filter: kept {len(relevant_cols)} / {len(feature_cols)} "
          f"features with |r| ≥ {score_thr}")

    if len(relevant_cols) <= 1:
        return relevant_cols                     # nothing left to de-correlate

    # ---------- 2. redundancy --------------------------------------
    corr_p = df_wide[relevant_cols].corr(method="pearson").abs()
    corr_s = df_wide[relevant_cols].corr(method="spearman").abs()

    upper = np.triu_indices_from(corr_p, k=1)
    mask_high = (corr_p.values[upper] > pearson_thr) | \
                (corr_s.values[upper] > spearman_thr)

    to_drop, kept = set(), set()
    for (i, j, hi) in zip(upper[0], upper[1], mask_high):
        if not hi:
            continue
        col_i, col_j = corr_p.columns[i], corr_p.columns[j]
        # precedence: keep col_i if not dropped yet
        if col_j in to_drop or col_i in kept:
            to_drop.add(col_j)
        elif col_i in to_drop or col_j in kept:
            to_drop.add(col_i)
        else:
            kept.add(col_i)
            to_drop.add(col_j)

    final_cols = [c for c in relevant_cols if c not in to_drop]
    print(f"Redundancy filter: dropped {len(to_drop)} correlated features "
          f"({len(final_cols)} remain).")
    return final_cols

def harmonic_mean(x, y):
    if x <= 0 or y <= 0:
        return min(x, y)
    return 2 / ((1 / x) + (1 / y))

def evaluate_predictions(true_A, pred_A, true_B, pred_B):
    accuracy_A, accuracy_B = accuracy_score(true_A, pred_A), accuracy_score(true_B, pred_B)
    accuracy = harmonic_mean(accuracy_A, accuracy_B)
    balanced_accuracy_A, balanced_accuracy_B = balanced_accuracy_score(true_A, pred_A), balanced_accuracy_score(true_B, pred_B)
    balanced_accuracy = harmonic_mean(balanced_accuracy_A, balanced_accuracy_B)
    f1_A, f1_B = f1_score(true_A, pred_A, average='weighted', zero_division=0), f1_score(true_B, pred_B, average='weighted', zero_division=0)
    f1 = harmonic_mean(f1_A, f1_B)
    precision_A, precision_B = precision_score(true_A, pred_A, average='weighted', zero_division=0), precision_score(true_B, pred_B, average='weighted', zero_division=0)
    precision = harmonic_mean(precision_A, precision_B)
    recall_A, recall_B = recall_score(true_A, pred_A, average='weighted', zero_division=0), recall_score(true_B, pred_B, average='weighted', zero_division=0)
    recall = harmonic_mean(recall_A, recall_B)
    sensitivity_A, sensitivity_B = recall_score(true_A, pred_A, average='binary', pos_label=1, zero_division=0), recall_score(true_B, pred_B, average='binary', pos_label=1, zero_division=0)
    sensitivity = harmonic_mean(sensitivity_A, sensitivity_B)
    specificity_A, specificity_B = recall_score(true_A, pred_A, average='binary', pos_label=0, zero_division=0), recall_score(true_B, pred_B, average='binary', pos_label=0, zero_division=0)
    specificity = harmonic_mean(specificity_A, specificity_B)
    mcc_A, mcc_B = matthews_corrcoef(true_A, pred_A), matthews_corrcoef(true_B, pred_B)
    mcc = harmonic_mean(mcc_A, mcc_B)
    cm_A = confusion_matrix(true_A, pred_A, labels=[0, 1])
    cm_B = confusion_matrix(true_B, pred_B, labels=[0, 1])
    normalized_matrices = []
    for cm in [cm_A, cm_B]:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        normalized_matrices.append(cm_norm)
    cm = hmean(normalized_matrices, axis=0)

    result = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc,
        'confusion_matrix': cm,
        'true_A': true_A,
        'pred_A': pred_A,
        'true_B': true_B,
        'pred_B': pred_B
    }
    return result

def distance_weighted_vote(step_labels: np.ndarray,
                           margins: np.ndarray) -> int:
    """Distance-weighted majority vote (paper’s rule)."""
    classes = np.unique(step_labels)
    conf = {c: np.sum(np.abs(margins[step_labels == c]))
            for c in classes}
    return max(conf, key=conf.get)

# ------------------------------------------------------------------
# Choose one of the three strategies
VOTE_STRATEGY = "majority"        # "majority", "prob_sum", "tanh_margin"
TANH_ALPHA    = None              # auto-scale if None

results_path = "./results_vote_pca_001-10C_04-1K"
os.makedirs(results_path, exist_ok=True)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# Preprocessing
RANDOM_STATE = 26
random.seed(RANDOM_STATE)

# Optuna Parameters
trial_num = 100

# Parallelization Parameters
study_n_jobs = 1

# Experiment Parameters
num_experiment_runs = 1

step_path = "./experimentData/meta_feature_df.pkl"
md_path = "./experimentData/meta_feature_score.pkl"
with open(step_path, "rb") as f:
    step_df = pickle.load(f)
with open(md_path, "rb") as f:
    score_df = pickle.load(f)

clf_func = lambda x: 0 if x > 0 else 1
step_df = step_df.merge(score_df, on=["PID", "task_id"], how="left")

X_full = step_df.iloc[:,3:].drop(["score"], axis=1)
y_pd = step_df["score"].apply(clf_func)
y_full = y_pd.to_numpy()

PID = step_df["PID"].to_numpy()
task_ids = step_df["task_id"].to_numpy()

inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def _run_nested_cross_validation(X_sub):
    outer_splits = list(outer_cv.split(X_sub, y_full, PID))
    outer_results = []
    counter = 0

    for train_idx, test_idx in tqdm(outer_splits, desc="Outer CV Splits"):
        split_time = time.time()
        # print(f"Starting outer split {counter} of {len(outer_splits)}")
        counter += 1

        # Prepare training data
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        task_train, task_test = task_ids[train_idx], task_ids[test_idx]
        PID_train, PID_test = PID[train_idx], PID[test_idx]

        def suggest_svc_params(trial, prefix="model__"):
            # ── kernel is always sampled ───────────────────────────────────────────────
            kernel = trial.suggest_categorical(f"{prefix}kernel", ["rbf"])

            # ── parameters that exist for *all* kernels ────────────────────────────────
            C = trial.suggest_float(f"{prefix}C", 0.0001, 0.1, log=True)
            max_iter     = trial.suggest_categorical("model__max_iter", [-1])

            # ── conditionals -----------------------------------------------------------
            if kernel in {"rbf", "sigmoid"}:
                gamma = trial.suggest_float(f"{prefix}gamma", 0.0001, 0.01, log=True)
            else:
                gamma = "scale"

            tol = trial.suggest_categorical(f"{prefix}tol", [1e-3])
            class_weight = trial.suggest_categorical(f"{prefix}class_weight", ["balanced"]) # {0: 0.555532870559412, 1: 5.001838235294118}])

            if VOTE_STRATEGY == "prob_sum":
                probability = trial.suggest_categorical(f"{prefix}probability", [True])
            else:
                probability = trial.suggest_categorical(f"{prefix}probability", [False])

            return dict(kernel=kernel, C=C, gamma=gamma, max_iter=max_iter, tol=tol, class_weight=class_weight, probability=probability)
            
        def objective(trial):
            pca_k = trial.suggest_float(f"pca_k", 0.4, 1.0, log=True)
            svc_params   = suggest_svc_params(trial, prefix="model__")

            var_thresh = VarianceThreshold(threshold=0.01)
            scaler = StandardScaler()
            pca = PCA(n_components=pca_k, random_state=RANDOM_STATE, svd_solver="full")
            clf = SVC(**svc_params, random_state=RANDOM_STATE)
                
            inner_splits = list(inner_cv.split(X_train, y_train, PID_train))
            inner_results = []

            for inner_step, (inner_train_idx, inner_test_idx) in tqdm(enumerate(inner_splits), desc="Hypopt CV Splits"):

                X_inner_train = X_train[inner_train_idx]
                X_inner_test = X_train[inner_test_idx]
                y_inner_train = y_train[inner_train_idx]
                y_inner_test = y_train[inner_test_idx]
                PID_inner_train = PID_train[inner_train_idx]
                PID_inner_test = PID_train[inner_test_idx]
                task_inner_train = task_train[inner_train_idx]
                task_inner_test = task_train[inner_test_idx]

                X_inner_train = var_thresh.fit_transform(X_inner_train)
                X_inner_test = var_thresh.transform(X_inner_test)

                X_inner_train_scaled = scaler.fit_transform(X_inner_train)
                X_inner_train_pca = pca.fit_transform(X_inner_train_scaled)
                clf.fit(X_inner_train_pca, y_inner_train)

                preds_A, preds_B = [], []
                true_A,  true_B  = [], []

                uniq_pids = np.unique(PID_inner_test)
                for pid in uniq_pids:
                    # Task A
                    mask_A = (PID_inner_test == pid) & (task_inner_test == "A")
                    if mask_A.any():
                        Xa = scaler.transform(X_inner_test[mask_A])
                        Za = pca.transform(Xa)
                        margins = clf.decision_function(Za)
                        step_lbls = clf.classes_[(margins > 0).astype(int)]

                        pred_A   = distance_weighted_vote(step_lbls, margins)
                        preds_A.append(pred_A)
                        true_A.append(y_inner_test[mask_A][0])  # same label for all steps of pid

                    # Task B
                    mask_B = (PID_inner_test == pid) & (task_inner_test == "B")
                    if mask_B.any():
                        Xb = scaler.transform(X_inner_test[mask_B])
                        Zb = pca.transform(Xb)
                        margins = clf.decision_function(Zb)
                        step_lbls = clf.classes_[(margins > 0).astype(int)]

                        pred_B   = distance_weighted_vote(step_lbls, margins)
                        preds_B.append(pred_B)
                        true_B.append(y_inner_test[mask_B][0])

                result = evaluate_predictions(true_A, preds_A, true_B, preds_B)
                result['PID_inner_train'] = PID_inner_train
                result['PID_inner_test'] = PID_inner_test
                inner_results.append(result)

                trial.report(result['mcc'], inner_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            trial.set_user_attr("inner_results", inner_results)

            cv_scores = [result['mcc'] for result in inner_results]
            return np.mean(cv_scores) - 0.5 * np.std(cv_scores)

        # Run Optuna study
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=2), sampler=sampler)
        study.optimize(objective, n_trials=trial_num, n_jobs=study_n_jobs)

        best_trial = study.best_trial
        best_params = best_trial.params
        inner_results = best_trial.user_attrs.get("inner_results", None)

        if inner_results is None:
            raise ValueError("No results found")

        def build_pipeline_from_svc_params(params):
            kernel = params["model__kernel"]
            C = params["model__C"]
            if kernel in {"rbf", "sigmoid"}:
                gamma = params["model__gamma"]

            max_iter = params["model__max_iter"]
            tol = params["model__tol"]
            class_weight = params["model__class_weight"]
            probability = params["model__probability"]
            model_instance = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight,
                                        random_state=RANDOM_STATE, max_iter=max_iter, tol=tol, probability=probability)
            return model_instance

        var_thresh = VarianceThreshold(threshold=0.01)
        scaler = StandardScaler()
        pca = PCA(n_components=best_params["pca_k"], random_state=RANDOM_STATE, svd_solver="full")
        clf = build_pipeline_from_svc_params(best_params)

        X_train = var_thresh.fit_transform(X_train)
        X_test = var_thresh.transform(X_test)

        X_train_scaled  = scaler.fit_transform(X_train)
        X_train_pca = pca.fit_transform(X_train_scaled)
        clf.fit(X_train_pca, y_train)

        preds_A, preds_B = [], []
        true_A,  true_B  = [], []
        for pid in np.unique(PID_test):
            # Task A
            mask_A = (PID_test == pid) & (task_test == "A")
            if mask_A.any():
                Xa = scaler.transform(X_test[mask_A])
                Za = pca.transform(Xa)
                margins = clf.decision_function(Za)
                step_lbls = clf.classes_[(margins > 0).astype(int)]

                pred_A   = distance_weighted_vote(step_lbls, margins)
                preds_A.append(pred_A)
                true_A.append(y_test[mask_A][0])  # same label for all steps of pid
            # Task B
            mask_B = (PID_test == pid) & (task_test == "B")
            if mask_B.any():
                Xb = scaler.transform(X_test[mask_B])
                Zb = pca.transform(Xb)
                margins = clf.decision_function(Zb)
                step_lbls = clf.classes_[(margins > 0).astype(int)]

                pred_B   = distance_weighted_vote(step_lbls, margins)
                preds_B.append(pred_B)
                true_B.append(y_test[mask_B][0])                
        result = evaluate_predictions(true_A, preds_A, true_B, preds_B)
        result['PID_train'] = PID_train
        result['PID_test'] = PID_test        
        result['best_params'] = best_params
        result['inner_results'] = inner_results
        outer_results.append(result)
        print(f"Nested CV Outer Fold Performance: {result['mcc']} | best_params: {best_params}")
        print(f"Split time: {time.time() - split_time}")

    # Aggregate CV metrics
    outer_cv_metrics = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'sensitivity': [],
        'specificity': [],
        'mcc': [],
        'confusion_matrices': [],
        'best_params': []
    }

    for result in outer_results:
        outer_cv_metrics['accuracy'].append(result['accuracy'])
        outer_cv_metrics['balanced_accuracy'].append(result['balanced_accuracy'])
        outer_cv_metrics['f1'].append(result['f1'])
        outer_cv_metrics['precision'].append(result['precision'])
        outer_cv_metrics['recall'].append(result['recall'])
        outer_cv_metrics['sensitivity'].append(result['sensitivity'])
        outer_cv_metrics['specificity'].append(result['specificity'])
        outer_cv_metrics['mcc'].append(result['mcc'])
        outer_cv_metrics['confusion_matrices'].append(result['confusion_matrix'])
        outer_cv_metrics['best_params'].append(result['best_params'])

    # param_counts = {}

    # for params in outer_cv_metrics['best_params']:
    #     for param, value in params.items():
    #         if param not in param_counts:
    #             param_counts[param] = {}
    #         param_counts[param][value] = param_counts[param].get(value, 0) + 1

    normalized_matrices = []
    for cm in outer_cv_metrics['confusion_matrices']:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        normalized_matrices.append(cm_norm)
    avg_cm = np.mean(normalized_matrices, axis=0)
    std_cm = np.std(normalized_matrices, axis=0)

    avg_metrics = {
        'accuracy': np.mean(outer_cv_metrics['accuracy']),
        'accuracy_std': np.std(outer_cv_metrics['accuracy']),
        'balanced_accuracy': np.mean(outer_cv_metrics['balanced_accuracy']),
        'balanced_accuracy_std': np.std(outer_cv_metrics['balanced_accuracy']),
        'f1': np.mean(outer_cv_metrics['f1']),
        'f1_std': np.std(outer_cv_metrics['f1']),
        'precision': np.mean(outer_cv_metrics['precision']),
        'precision_std': np.std(outer_cv_metrics['precision']),
        'recall': np.mean(outer_cv_metrics['recall']),
        'recall_std': np.std(outer_cv_metrics['recall']),
        'sensitivity': np.mean(outer_cv_metrics['sensitivity']),
        'sensitivity_std': np.std(outer_cv_metrics['sensitivity']),
        'specificity': np.mean(outer_cv_metrics['specificity']),
        'specificity_std': np.std(outer_cv_metrics['specificity']),
        'mcc': np.mean(outer_cv_metrics['mcc']),
        'mcc_std': np.std(outer_cv_metrics['mcc']),
        'confusion_matrices': outer_cv_metrics['confusion_matrices'],
        # 'hyperparameter_tuning': param_counts,
        'avg_confusion_matrix': avg_cm,
        'std_confusion_matrix': std_cm,
        'outer_results': outer_results
    }
    
    return avg_metrics

def _run_final_cross_validation(X_sub, outer_results):

    outer_splits = list(outer_cv.split(X_sub, y_full, PID))    
    final_results = []
    for outer_result_idx, outer_result in enumerate(outer_results):
        candidate_params = outer_result['best_params']

        cv_results = []
        for train_idx, test_idx in outer_splits:

            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]
            task_train, task_test = task_ids[train_idx], task_ids[test_idx]
            PID_train, PID_test = PID[train_idx], PID[test_idx]
            
            def build_pipeline_from_svc_params(params):
                kernel = params["model__kernel"]
                C = params["model__C"]
                if kernel in {"rbf", "sigmoid"}:
                    gamma = params["model__gamma"]

                max_iter = params["model__max_iter"]
                tol = params["model__tol"]
                class_weight = params["model__class_weight"]
                probability = params["model__probability"]
                model_instance = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight,
                                            random_state=RANDOM_STATE, max_iter=max_iter, tol=tol, probability=probability)
                return model_instance

            var_thresh = VarianceThreshold(threshold=0.01)
            scaler = StandardScaler()
            pca = PCA(n_components=candidate_params["pca_k"], random_state=RANDOM_STATE, svd_solver="full")
            clf = build_pipeline_from_svc_params(candidate_params)

            X_train = var_thresh.fit_transform(X_train)
            X_test = var_thresh.transform(X_test)

            X_train_scaled  = scaler.fit_transform(X_train)
            X_train_pca = pca.fit_transform(X_train_scaled)
            clf.fit(X_train_pca, y_train)

            preds_A, preds_B = [], []
            true_A,  true_B  = [], []
            for pid in np.unique(PID_test):
                # Task A
                mask_A = (PID_test == pid) & (task_test == "A")
                if mask_A.any():
                    Xa = scaler.transform(X_test[mask_A])
                    Za = pca.transform(Xa)
                    margins = clf.decision_function(Za)
                    step_lbls = clf.classes_[(margins > 0).astype(int)]

                    pred_A   = distance_weighted_vote(step_lbls, margins)
                    preds_A.append(pred_A)
                    true_A.append(y_test[mask_A][0])  # same label for all steps of pid

                # Task B
                mask_B = (PID_test == pid) & (task_test == "B")
                if mask_B.any():
                    Xb = scaler.transform(X_test[mask_B])
                    Zb = pca.transform(Xb)
                    margins = clf.decision_function(Zb)
                    step_lbls = clf.classes_[(margins > 0).astype(int)]

                    pred_B   = distance_weighted_vote(step_lbls, margins)
                    preds_B.append(pred_B)
                    true_B.append(y_test[mask_B][0])

            result = evaluate_predictions(true_A, preds_A, true_B, preds_B)
            result['PID_train'] = PID_train
            result['PID_test'] = PID_test        
            result['candidate_params'] = candidate_params
            cv_results.append(result)

        # Aggregate CV metrics
        cv_metrics = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'sensitivity': [],
            'specificity': [],
            'mcc': [],
            'confusion_matrices': []
        }

        for result in cv_results:
            cv_metrics['accuracy'].append(result['accuracy'])
            cv_metrics['balanced_accuracy'].append(result['balanced_accuracy'])
            cv_metrics['f1'].append(result['f1'])
            cv_metrics['precision'].append(result['precision'])
            cv_metrics['recall'].append(result['recall'])
            cv_metrics['sensitivity'].append(result['sensitivity'])
            cv_metrics['specificity'].append(result['specificity'])
            cv_metrics['mcc'].append(result['mcc'])
            cv_metrics['confusion_matrices'].append(result['confusion_matrix'])

        normalized_matrices = []
        for cm in cv_metrics['confusion_matrices']:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            normalized_matrices.append(cm_norm)
        avg_cm = np.mean(normalized_matrices, axis=0)
        std_cm = np.std(normalized_matrices, axis=0)

        kfold_metrics = {
            'accuracy': np.mean(cv_metrics['accuracy']),
            'accuracy_std': np.std(cv_metrics['accuracy']),
            'balanced_accuracy': np.mean(cv_metrics['balanced_accuracy']),
            'balanced_accuracy_std': np.std(cv_metrics['balanced_accuracy']),
            'f1': np.mean(cv_metrics['f1']),
            'f1_std': np.std(cv_metrics['f1']),
            'precision': np.mean(cv_metrics['precision']),
            'precision_std': np.std(cv_metrics['precision']),
            'recall': np.mean(cv_metrics['recall']),
            'recall_std': np.std(cv_metrics['recall']),
            'sensitivity': np.mean(cv_metrics['sensitivity']),
            'sensitivity_std': np.std(cv_metrics['sensitivity']),
            'specificity': np.mean(cv_metrics['specificity']),
            'specificity_std': np.std(cv_metrics['specificity']),
            'mcc': np.mean(cv_metrics['mcc']),
            'mcc_std': np.std(cv_metrics['mcc']),
            'confusion_matrices': cv_metrics['confusion_matrices'],
            'avg_confusion_matrix': avg_cm,
            'std_confusion_matrix': std_cm,
            'kfold_results': cv_results
        }

        final_results.append(kfold_metrics)
    
    return final_results

def _run_experiment(parameters, best_score):
    start_time = time.time()
    feature_set, label = parameters
    
    print(f"Running experiment for {label}")
    
    # optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_sub = X_full[feature_set]
    
    filtered_cols = rel_red_filter(X_sub,
                                    feature_set,
                                    score_col=y_pd,
                                    score_thr=0.035,
                                    pearson_thr=0.7,
                                    spearman_thr=0.7)
    X_sub = X_sub[filtered_cols].to_numpy()    

    nested_metrics = _run_nested_cross_validation(X_sub)
    final_results = _run_final_cross_validation(X_sub, nested_metrics['outer_results'])

    final_results_metrics = []
    for i, result in enumerate(final_results):
        final_results_metrics.append(result['mcc'])
        final_params = result['kfold_results'][0]['candidate_params']
        print(f"Final K-Fold {i} MCC: {result['mcc']} | Params: {final_params}")

    print(f"Final Best MCC: {max(final_results_metrics)}")
    print(f"Final Best Params: {final_results[np.argmax(final_results_metrics)]['kfold_results'][0]['candidate_params']}")

    if max(final_results_metrics) > best_score:
        best_score = max(final_results_metrics)
        with open(f"{results_path}/{label}_nested_metrics.pkl", "wb") as f:
            pickle.dump(nested_metrics, f)
        with open(f"{results_path}/{label}_final_results.pkl", "wb") as f:
            pickle.dump(final_results, f)
    print(f"Experiment run time: {time.time() - start_time}")
    return best_score

config_path = "./experimentConfigs"
experiment_configs = sorted(os.listdir(config_path))
feature_sets = []
feature_labels = []
for file_name in experiment_configs:
    if "!" in file_name:
        continue
    with open(os.path.join(config_path, file_name), "r") as f:
        config = json.load(f)
    
    feature_labels.append(file_name.replace(".json", ""))
    feature_sets.append(config)

parameter_list = []
for feature_set, feature_label in zip(feature_sets, feature_labels):
    parameter_list.append([feature_set, feature_label])

for parameters in parameter_list:
    print(f"Running {parameters[1]}")
    best_score = -np.inf
    for experiment_num in range(num_experiment_runs):
        best_score = _run_experiment(parameters, best_score)
        print(f"{parameters[1]} | Experiment {experiment_num} best score: {best_score}")