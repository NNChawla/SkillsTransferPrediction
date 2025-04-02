import pandas as pd
import numpy as np
import warnings
import pickle, time, gc, sys, os
from tqdm import tqdm
#from tqdm_joblib import tqdm_joblib  # no longer needed for Optuna
import optuna

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix, matthews_corrcoef,
                             make_scorer)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# Preprocessing
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
random_state = 25

tab_path = "./experimentData/tabulated_dataframe.pkl"
md_path = "./experimentData/metadata.csv"
with open(tab_path, "rb") as f:
    tabulated_dataframe = pickle.load(f)
all_features = tabulated_dataframe.columns
metadata_dataframe = pd.read_csv(md_path)

class SFSSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, k_features, forward=True, floating=True, scoring=None, cv=5, n_jobs=1):
        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.sfs_ = SFS(estimator=self.estimator,
                        k_features=self.k_features,
                        forward=self.forward,
                        floating=self.floating,
                        scoring=self.scoring,
                        cv=self.cv,
                        n_jobs=self.n_jobs,
                        verbose=0)
        self.sfs_.fit(X, y)
        self.k_feature_idx_ = self.sfs_.k_feature_idx_
        return self

    def transform(self, X):
        # Convert to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, list(self.k_feature_idx_)]
        return X[:, self.k_feature_idx_]

def get_strings_with_substrings(string_list, target_strings):
    valid_strings = []
    for target_string in target_strings:
        if all(s in target_string for s in string_list):
            valid_strings.append(target_string)
    valid_strings = sorted(list(set(["_".join(i.split("_")[:-1]) for i in valid_strings])))
    return valid_strings

def get_scoring_function(measure):
    if measure == 'specificity':
        return lambda y_true, y_pred: recall_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
    elif measure == 'f1':
        return lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
    elif measure == 'accuracy':
        return accuracy_score
    elif measure == 'balanced_accuracy':
        return balanced_accuracy_score
    elif measure == 'mcc':
        return matthews_corrcoef
    else:
        raise ValueError(f"Invalid feature selection measure: {measure}")

def evaluate_predictions(y_true, y_pred, score_fn):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    sensitivity = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
    feature_selection_measure = score_fn(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    result = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'feature_selection_measure': feature_selection_measure,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }

    return result

def drop_high_na_columns(input_df, na_threshold):
    df = input_df.copy()
    na_counts = df.isna().sum()
    columns_to_drop = na_counts[na_counts > na_threshold].index
    return df.drop(columns=columns_to_drop)

def variance_filter(input_df):
    df = input_df.copy()
    variances = df.var()
    low_variance_features = variances[variances < 0.01].index
    na_variance_features = variances[variances.isna()].index
    df = df.drop(columns=low_variance_features)
    df = df.drop(columns=na_variance_features)
    return df

def scale_impute_df(input_df, scaler, imputer):
    df = input_df.copy()
    df = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(df)), columns=df.columns)
    return df

def _init_subset_data(search_terms, use_metadata_features=False):
    subset_features = get_strings_with_substrings(search_terms, all_features)
    subset_features_A = [f"{i}_A" for i in subset_features]
    subset_features_A = [i for i in subset_features_A if i in all_features]
    tabulated_dataframe_subset_A = tabulated_dataframe[["PID"] + subset_features_A].copy()
    subset_features_B = [f"{i}_B" for i in subset_features]
    subset_features_B = [i for i in subset_features_B if i in all_features]
    tabulated_dataframe_subset_B = tabulated_dataframe[["PID"] + subset_features_B].copy()

    metadata_df_A = None
    metadata_df_B = None
    if use_metadata_features:
        include_set_feature_A = ["A_RW_Build_Time"]
        include_set_feature_B = ["B_RW_Build_Time"]
        exclude_set_feature_A = ["B_RW_Build_Time"]
        exclude_set_feature_B = ["A_RW_Build_Time"]
        drop_columns = ["Score_A", "Score_B", "Score_A_Linear", "Score_B_Linear"]
        drop_columns_A = drop_columns + exclude_set_feature_A
        drop_columns_B = drop_columns + exclude_set_feature_B
        scalable_metadata_features = ["hour_of_day", "time_sin", "time_cos", "TotalDuration"]
        scalable_metadata_features_A = scalable_metadata_features + include_set_feature_A
        scalable_metadata_features_B = scalable_metadata_features + include_set_feature_B

        metadata_df_A = metadata_dataframe.copy()
        metadata_df_B = metadata_dataframe.copy()
        metadata_df_A = metadata_df_A.drop(columns=drop_columns_A)
        metadata_df_B = metadata_df_B.drop(columns=drop_columns_B)
        metadata_df_A = metadata_df_A.rename(columns={"ID": "PID"})
        metadata_df_B = metadata_df_B.rename(columns={"ID": "PID"})
        scalable_metadata_dataframe_A = metadata_df_A[scalable_metadata_features_A + ["PID"]].copy()
        scalable_metadata_dataframe_B = metadata_df_B[scalable_metadata_features_B + ["PID"]].copy()
        tabulated_dataframe_subset_A = pd.merge(tabulated_dataframe_subset_A, scalable_metadata_dataframe_A, on="PID", how="inner")
        tabulated_dataframe_subset_B = pd.merge(tabulated_dataframe_subset_B, scalable_metadata_dataframe_B, on="PID", how="inner")

        metadata_PID = metadata_df_A["PID"]
        metadata_df_A = metadata_df_A.drop(columns=scalable_metadata_features_A + ["PID"])
        metadata_df_A = variance_filter(metadata_df_A)
        metadata_df_A = pd.concat([metadata_PID, metadata_df_A], axis=1)
        metadata_PID = metadata_df_B["PID"]
        metadata_df_B = metadata_df_B.drop(columns=scalable_metadata_features_B + ["PID"])
        metadata_df_B = variance_filter(metadata_df_B)
        metadata_df_B = pd.concat([metadata_PID, metadata_df_B], axis=1)

    tabulated_PID = tabulated_dataframe_subset_A["PID"]
    tabulated_dataframe_subset_A = tabulated_dataframe_subset_A.drop(columns=["PID"])
    tabulated_dataframe_subset_A = drop_high_na_columns(tabulated_dataframe_subset_A, int(tabulated_dataframe_subset_A.shape[0] * 0.2))
    tabulated_dataframe_subset_A = variance_filter(tabulated_dataframe_subset_A)
    tabulated_dataframe_subset_A = pd.concat([tabulated_PID, tabulated_dataframe_subset_A], axis=1)

    tabulated_PID = tabulated_dataframe_subset_B["PID"]
    tabulated_dataframe_subset_B = tabulated_dataframe_subset_B.drop(columns=["PID"])
    tabulated_dataframe_subset_B = drop_high_na_columns(tabulated_dataframe_subset_B, int(tabulated_dataframe_subset_B.shape[0] * 0.2))
    tabulated_dataframe_subset_B = variance_filter(tabulated_dataframe_subset_B)
    tabulated_dataframe_subset_B = pd.concat([tabulated_PID, tabulated_dataframe_subset_B], axis=1)

    return tabulated_dataframe_subset_A, tabulated_dataframe_subset_B, metadata_df_A, metadata_df_B, subset_features_A, subset_features_B

def _init_cross_validation(model_name, run_cross_task, feature_selection_method,
                           scoring_function_to_use, clf_function_to_use, seed):
    if run_cross_task:
        score_A = metadata_dataframe["Score_B_Linear"].apply(clf_function_to_use)
        score_B = metadata_dataframe["Score_A_Linear"].apply(clf_function_to_use)
    else:
        score_A = metadata_dataframe["Score_A_Linear"].apply(clf_function_to_use)
        score_B = metadata_dataframe["Score_B_Linear"].apply(clf_function_to_use)

    inner_cv = (RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed) # revert to 3 repeats when testing is done
                if run_cross_task 
                else StratifiedKFold(n_splits=1, shuffle=True, random_state=seed)) # revert to 4 splits when testing is done
    # We'll record which feature selection method is used:
    pipeline_template = feature_selection_method  # "sfs", "rfe", or None
    return score_A, score_B, inner_cv, pipeline_template

def _run_cross_validation(subset_df_A, subset_df_B, metadata_df_A, metadata_df_B, score_A, score_B, run_cross_task,
                          feature_selection_method, scoring_function_to_use,
                          label_desc, subset_features_A, subset_features_B, inner_cv, outer_cv,
                          pipeline_template, model_name, seed):
    # Outer CV splits (if run_cross_task then use all data; otherwise use outer_cv splits)
    col_indices = np.arange(len(subset_df_A))
    splits = [(col_indices, col_indices)] if run_cross_task else list(outer_cv.split(subset_df_A, score_A))

    results = []
    for train_idx, test_idx in splits:
        # Prepare training data
        train_data_A = subset_df_A.iloc[train_idx].copy()
        train_PID = train_data_A["PID"]
        train_data_A = train_data_A.drop(columns=["PID"])
        train_data_A = scale_impute_df(train_data_A, scaler, imputer)
        if metadata_df_A is not None:
            train_data_A = pd.concat([train_PID, train_data_A], axis=1)
            train_data_A = pd.merge(train_data_A, metadata_df_A, on="PID", how="inner")
            train_data_A = train_data_A.drop(columns=["PID"])
        train_score_A = score_A.iloc[train_idx].copy()

        train_data_B = subset_df_B.iloc[train_idx].copy()
        train_PID = train_data_B["PID"]
        train_data_B = train_data_B.drop(columns=["PID"])
        train_data_B = scale_impute_df(train_data_B, scaler, imputer)
        if metadata_df_B is not None:
            train_data_B = pd.concat([train_PID, train_data_B], axis=1)
            train_data_B = pd.merge(train_data_B, metadata_df_B, on="PID", how="inner")
            train_data_B = train_data_B.drop(columns=["PID"])
        train_score_B = score_B.iloc[train_idx].copy()

        # Prepare testing data
        test_data_A = subset_df_A.iloc[test_idx].copy()
        test_PID = test_data_A["PID"]
        test_data_A = test_data_A.drop(columns=["PID"])
        test_data_A = scale_impute_df(test_data_A, scaler, imputer)
        if metadata_df_A is not None:
            test_data_A = pd.concat([test_PID, test_data_A], axis=1)
            test_data_A = pd.merge(test_data_A, metadata_df_A, on="PID", how="inner")
            test_data_A = test_data_A.drop(columns=["PID"])
        test_score_A = score_A.iloc[test_idx].copy()

        test_data_B = subset_df_B.iloc[test_idx].copy()
        test_PID = test_data_B["PID"]
        test_data_B = test_data_B.drop(columns=["PID"])
        test_data_B = scale_impute_df(test_data_B, scaler, imputer)
        if metadata_df_B is not None:
            test_data_B = pd.concat([test_PID, test_data_B], axis=1)
            test_data_B = pd.merge(test_data_B, metadata_df_B, on="PID", how="inner")
            test_data_B = test_data_B.drop(columns=["PID"])
        test_score_B = score_B.iloc[test_idx].copy()

        # Define the objective function for Optuna
        def objective(trial):
            # Build model based on model_name
            if model_name == "linsvc":
                penalty = trial.suggest_categorical("model__penalty", ['l1', 'l2'])
                C = trial.suggest_loguniform("model__C", 0.001, 1000)
                max_iter = trial.suggest_categorical("model__max_iter", [1000, 2000, 5000, 10000])
                model_instance = LinearSVC(penalty=penalty, C=C, class_weight='balanced',
                                           random_state=seed, max_iter=max_iter)
            elif model_name == "logreg":
                penalty = trial.suggest_categorical("model__penalty", ['l1', 'l2'])
                C = trial.suggest_loguniform("model__C", 0.001, 1000)
                max_iter = trial.suggest_categorical("model__max_iter", [1000, 2000, 5000, 10000])
                solver = trial.suggest_categorical("model__solver", ['liblinear'])
                model_instance = LogisticRegression(penalty=penalty, C=C, class_weight='balanced',
                                                    random_state=seed, max_iter=max_iter, n_jobs=1, solver=solver)
            elif model_name == "knn":
                n_neighbors = trial.suggest_int("model__n_neighbors", 3, 21, step=2)
                metric = trial.suggest_categorical("model__metric", ['euclidean', 'manhattan'])
                weights = trial.suggest_categorical("model__weights", ['uniform', 'distance'])
                model_instance = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
            else:
                raise ValueError("Unknown model name")

            steps = []
            if feature_selection_method == "sfs":
                floating = trial.suggest_categorical("fs__floating", [False])
                forward = trial.suggest_categorical("fs__forward", [True])
                steps.append(('fs', SFSSelector(estimator=model_instance,
                                                k_features="best",
                                                forward=forward,
                                                floating=floating,
                                                scoring=make_scorer(scoring_function_to_use),
                                                cv=inner_cv,
                                                n_jobs=1)))
                steps.append(('model', model_instance))
            elif feature_selection_method == "rfe":
                n_features_to_select = trial.suggest_int("fs__n_features_to_select", 1, 20)
                step = trial.suggest_categorical("fs__step", [1, 2, 5, 7, 10])
                rfe = RFE(estimator=model_instance, n_features_to_select=n_features_to_select, step=step)
                steps.append(('fs', rfe))
                steps.append(('model', model_instance))
            else:
                steps.append(('model', model_instance))
            pipeline = Pipeline(steps)
            # Evaluate with inner CV
            cv_scores_A = cross_val_score(pipeline, train_data_A, train_score_A, cv=inner_cv,
                                        scoring=make_scorer(scoring_function_to_use), n_jobs=-1)
            cv_scores_B = cross_val_score(pipeline, train_data_B, train_score_B, cv=inner_cv,
                                        scoring=make_scorer(scoring_function_to_use), n_jobs=-1)
            return np.mean(cv_scores_A), np.mean(cv_scores_B)

        # Run Optuna study
        study = optuna.create_study(directions=["maximize", "maximize"])
        study.optimize(objective, n_trials=None, timeout=3600)
        best_params = study.best_trials[np.argmax([np.mean(trial.values) for trial in study.best_trials])].params

        with open(f"./results_server/{label_desc}_study.pkl", "wb") as f:
            pickle.dump(study, f)

        # Helper to rebuild pipeline from best_params
        def build_pipeline_from_params(params):
            if model_name == "linsvc":
                penalty = params["model__penalty"]
                C = params["model__C"]
                max_iter = params["model__max_iter"]
                model_instance = LinearSVC(penalty=penalty, C=C, class_weight='balanced',
                                           random_state=seed, max_iter=max_iter)
            elif model_name == "logreg":
                penalty = params["model__penalty"]
                C = params["model__C"]
                max_iter = params["model__max_iter"]
                solver = params["model__solver"]
                model_instance = LogisticRegression(penalty=penalty, C=C, class_weight='balanced',
                                                    random_state=seed, max_iter=max_iter, n_jobs=1, solver=solver)
            elif model_name == "knn":
                n_neighbors = params["model__n_neighbors"]
                metric = params["model__metric"]
                weights = params["model__weights"]
                model_instance = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
            else:
                raise ValueError("Unknown model name")
            steps = []
            if feature_selection_method == "sfs":
                floating = params["fs__floating"]
                forward = params["fs__forward"]
                steps.append(('fs', SFSSelector(estimator=model_instance,
                                                k_features="best",
                                                forward=forward,
                                                floating=floating,
                                                scoring=make_scorer(scoring_function_to_use),
                                                cv=inner_cv,
                                                n_jobs=1)))
                steps.append(('model', model_instance))
            elif feature_selection_method == "rfe":
                n_features_to_select = params["fs__n_features_to_select"]
                step = params["fs__step"]
                rfe = RFE(estimator=model_instance, n_features_to_select=n_features_to_select, step=step)
                steps.append(('fs', rfe))
                steps.append(('model', model_instance))
            else:
                steps.append(('model', model_instance))
            return Pipeline(steps)

        best_pipeline_A = build_pipeline_from_params(best_params)
        # Retrain the best pipeline on the entire training fold
        best_pipeline_A.fit(train_data_A, train_score_A)
        pred_score_A = best_pipeline_A.predict(test_data_A)
        best_pipeline_B = build_pipeline_from_params(best_params)
        best_pipeline_B.fit(train_data_B, train_score_B)
        pred_score_B = best_pipeline_B.predict(test_data_B)

        # If feature selection was used, try to extract the selected features.
        selected_features_A = None
        selected_features_B = None
        if feature_selection_method in ["sfs", "rfe"]:
            # For SFSSelector the attribute is "k_feature_idx_"
            if hasattr(best_pipeline_A.named_steps['fs'], 'k_feature_idx_'):
                selected_mask = best_pipeline_A.named_steps['fs'].k_feature_idx_
                selected_features_A = np.array(train_data_A.columns)[selected_mask]
            elif hasattr(best_pipeline_A.named_steps['fs'], 'support_'):
                selected_mask = best_pipeline_A.named_steps['fs'].support_
                selected_features_A = np.array(train_data_A.columns)[selected_mask]

            if hasattr(best_pipeline_B.named_steps['fs'], 'k_feature_idx_'):
                selected_mask = best_pipeline_B.named_steps['fs'].k_feature_idx_
                selected_features_B = np.array(train_data_B.columns)[selected_mask]
            elif hasattr(best_pipeline_B.named_steps['fs'], 'support_'):
                selected_mask = best_pipeline_B.named_steps['fs'].support_
                selected_features_B = np.array(train_data_B.columns)[selected_mask]

        result = evaluate_predictions(test_score_A, pred_score_A, scoring_function_to_use)
        result_B = evaluate_predictions(test_score_B, pred_score_B, scoring_function_to_use)
        result['accuracy'] = (result['accuracy'] + result_B['accuracy']) / 2
        result['balanced_accuracy'] = (result['balanced_accuracy'] + result_B['balanced_accuracy']) / 2
        result['f1'] = (result['f1'] + result_B['f1']) / 2
        result['precision'] = (result['precision'] + result_B['precision']) / 2
        result['recall'] = (result['recall'] + result_B['recall']) / 2
        result['sensitivity'] = (result['sensitivity'] + result_B['sensitivity']) / 2
        result['specificity'] = (result['specificity'] + result_B['specificity']) / 2
        result['feature_selection_measure'] = (result['feature_selection_measure'] + result_B['feature_selection_measure']) / 2
        
        result['best_params'] = best_params
        result['selected_features'] = np.append(selected_features_A, selected_features_B)
        print(f"Performance: {result['feature_selection_measure']} | best_params: {best_params}")
        results.append(result)

    # Aggregate CV metrics
    cv_metrics = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'sensitivity': [],
        'specificity': [],
        'feature_selection_measure': [],
        'confusion_matrices': [],
        'selected_features': [],
        'best_params': []
    }

    for result in results:
        cv_metrics['accuracy'].append(result['accuracy'])
        cv_metrics['balanced_accuracy'].append(result['balanced_accuracy'])
        cv_metrics['f1'].append(result['f1'])
        cv_metrics['precision'].append(result['precision'])
        cv_metrics['recall'].append(result['recall'])
        cv_metrics['sensitivity'].append(result['sensitivity'])
        cv_metrics['specificity'].append(result['specificity'])
        cv_metrics['feature_selection_measure'].append(result['feature_selection_measure'])
        
        cv_metrics['selected_features'].append(result['selected_features'])
        cv_metrics['best_params'].append(result['best_params'])

    feature_counts = {}
    param_counts = {}

    for params in cv_metrics['best_params']:
        for param, value in params.items():
            if param not in param_counts:
                param_counts[param] = {}
            param_counts[param][value] = param_counts[param].get(value, 0) + 1

    for features in cv_metrics['selected_features']:
        if features is not None:
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

    avg_metrics = {
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
        'feature_selection_measure': np.mean(cv_metrics['feature_selection_measure']),
        'feature_selection_measure_std': np.std(cv_metrics['feature_selection_measure']),
        'feature_selection': sorted([(i, j) for i, j in feature_counts.items()], key=lambda x: x[1], reverse=True),
        'feature_counts': feature_counts,
        'hyperparameter_tuning': param_counts
    }

    with open(f"./results_server/{label_desc}_avg_metrics.pkl", "wb") as f:
        pickle.dump(avg_metrics, f)

def _run_experiment(parameters):
    subset_feature_terms, cross_task, feature_selection, model_name, clf_func, label, scoring_function = parameters
    subset_df_A, subset_df_B, metadata_df_A, metadata_df_B, subset_features_A, subset_features_B = _init_subset_data(subset_feature_terms, use_metadata_features=False)
    if len(subset_df_A.columns) < 2 or len(subset_df_B.columns) < 2:
        print(f"Skipping {label} because it has less than 2 features")
        return
    # Use the new cross-validation initialization (grid_search replaced by Optuna)
    score_A, score_B, inner_cv, pipeline_template = _init_cross_validation(model_name, cross_task,
                                                                   feature_selection, scoring_function, clf_func, random_state)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    _run_cross_validation(subset_df_A, subset_df_B, metadata_df_A, metadata_df_B, score_A, score_B, cross_task, feature_selection, scoring_function,
                          label, subset_features_A, subset_features_B, inner_cv, outer_cv, pipeline_template, model_name, random_state)

feature_levels = ["session"]
feature_objects = ["RightHand"]
feature_types = ["axis"]
feature_subtypes = ["inter_extrema_intervals_base", "inter_extrema_intervals_low",
                      "inter_extrema_intervals_mid", "inter_extrema_intervals_high"]
all_subset_feature_terms = []
for feature_level in feature_levels:
    for feature_object in feature_objects:
        for feature_type in feature_types:
            for feature_subtype in feature_subtypes:
                all_subset_feature_terms.append([feature_level, feature_object, feature_type, feature_subtype])

cross_tasks = [False]
feature_selections_models = [("rfe", "linsvc")]# [("rfe", "logreg")]#, ("sfs", "linsvc"), ("sfs", "knn")]
clf_funcs = [lambda x: 1 if x > 0 else 0]

parameter_list = []
for subset_feature_terms in all_subset_feature_terms:
    for cross_task in cross_tasks:
        for feature_selection, model_name in feature_selections_models:
            for clf_func in clf_funcs:
                is_cross_task = "cross_task" if cross_task else "within_task"
                label = f"{'_'.join(subset_feature_terms)}_{feature_selection}_{model_name}_multiobjective_{is_cross_task}"
                scoring_function = get_scoring_function("mcc")
                parameter_list.append([subset_feature_terms, cross_task, feature_selection,
                                        model_name, clf_func, label, scoring_function])

_ = [
    _run_experiment(parameters) 
    for parameters in 
    tqdm(parameter_list, desc="Running All Experiments")
]