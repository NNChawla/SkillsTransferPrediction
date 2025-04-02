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

def _init_model(model_name, seed):
    # This helper is still used for default grid parameters.
    if model_name == "linsvc":
        param_grid = {
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'model__class_weight': ['balanced'],
            'model__random_state': [seed],
            'model__max_iter': [1000, 2000, 5000, 10000]
        }
        model = LinearSVC()
    elif model_name == "logreg":
        param_grid = {
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'model__class_weight': ['balanced'],
            'model__random_state': [seed],
            'model__max_iter': [1000, 2000, 5000, 10000],
            'model__n_jobs': [1],
            'model__solver': ['liblinear']
        }
        model = LogisticRegression()
    elif model_name == "knn":
        param_grid = {
            'model__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'model__metric': ['euclidean', 'manhattan'],
            'model__weights': ['uniform', 'distance'],
        }
        model = KNeighborsClassifier()
    return param_grid, model

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

def _init_subset_data(training_set, search_terms, use_metadata_features=False):
    subset_features = get_strings_with_substrings(search_terms, all_features)
    subset_features = [f"{i}_{training_set}" for i in subset_features]
    subset_features = [i for i in subset_features if i in all_features]
    tabulated_dataframe_subset = tabulated_dataframe.copy()
    tabulated_dataframe_subset = tabulated_dataframe_subset[["PID"] + subset_features].copy()

    metadata_df = None
    if use_metadata_features:
        include_set_feature = ["A_RW_Build_Time"] if training_set == 'A' else ["B_RW_Build_Time"]
        exclude_set_feature = ["B_RW_Build_Time"] if training_set == 'A' else ["A_RW_Build_Time"]
        drop_columns = ["Score_A", "Score_B", "Score_A_Linear", "Score_B_Linear"] + exclude_set_feature
        scalable_metadata_features = ["hour_of_day", "time_sin", "time_cos", "TotalDuration"] + include_set_feature

        metadata_df = metadata_dataframe.copy()
        metadata_df = metadata_df.drop(columns=drop_columns)
        metadata_df = metadata_df.rename(columns={"ID": "PID"})
        scalable_metadata_dataframe = metadata_df[scalable_metadata_features + ["PID"]].copy()
        tabulated_dataframe_subset = pd.merge(tabulated_dataframe_subset, scalable_metadata_dataframe, on="PID", how="inner")

        metadata_PID = metadata_df["PID"]
        metadata_df = metadata_df.drop(columns=scalable_metadata_features + ["PID"])
        metadata_df = variance_filter(metadata_df)
        metadata_df = pd.concat([metadata_PID, metadata_df], axis=1)

    tabulated_PID = tabulated_dataframe_subset["PID"]
    tabulated_dataframe_subset = tabulated_dataframe_subset.drop(columns=["PID"])
    tabulated_dataframe_subset = drop_high_na_columns(tabulated_dataframe_subset, int(tabulated_dataframe_subset.shape[0] * 0.2))
    tabulated_dataframe_subset = variance_filter(tabulated_dataframe_subset)
    tabulated_dataframe_subset = pd.concat([tabulated_PID, tabulated_dataframe_subset], axis=1)

    return tabulated_dataframe_subset, metadata_df, subset_features

def _init_cross_validation(model_name, training_set, run_cross_task, feature_selection_method,
                           scoring_function_to_use, clf_function_to_use, seed):
    score_A = metadata_dataframe["Score_A_Linear"].apply(clf_function_to_use)
    score_B = metadata_dataframe["Score_B_Linear"].apply(clf_function_to_use)

    score_to_use = (score_A if training_set == 'B' else score_B) if run_cross_task else (score_A if training_set == 'A' else score_B)
    inner_cv = (RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed) 
                if run_cross_task 
                else StratifiedKFold(n_splits=4, shuffle=True, random_state=seed))
    # We'll record which feature selection method is used:
    pipeline_template = feature_selection_method  # "sfs", "rfe", or None
    return score_to_use, inner_cv, pipeline_template

def _run_cross_validation(subset_df, metadata_df, score, run_cross_task,
                          feature_selection_method, scoring_function_to_use,
                          label_desc, available_features, inner_cv, outer_cv,
                          pipeline_template, model_name, seed):
    # Outer CV splits (if run_cross_task then use all data; otherwise use outer_cv splits)
    col_indices = np.arange(len(subset_df))
    splits = [(col_indices, col_indices)] if run_cross_task else list(outer_cv.split(subset_df, score))

    results = []
    for train_idx, test_idx in splits:
        # Prepare training data
        train_data = subset_df.iloc[train_idx].copy()
        train_PID = train_data["PID"]
        train_data = train_data.drop(columns=["PID"])
        train_data = scale_impute_df(train_data, scaler, imputer)
        if metadata_df is not None:
            train_data = pd.concat([train_PID, train_data], axis=1)
            train_data = pd.merge(train_data, metadata_df, on="PID", how="inner")
            train_data = train_data.drop(columns=["PID"])
        train_score = score.iloc[train_idx].copy()

        # Prepare testing data
        try:
            test_data = subset_df.iloc[test_idx].copy()
            test_PID = test_data["PID"]
            test_data = test_data.drop(columns=["PID"])
            test_data = scale_impute_df(test_data, scaler, imputer)
            if metadata_df is not None:
                test_data = pd.concat([test_PID, test_data], axis=1)
                test_data = pd.merge(test_data, metadata_df, on="PID", how="inner")
                test_data = test_data.drop(columns=["PID"])
            test_score = score.iloc[test_idx].copy()
        except Exception as e:
            print(f"Error in _run_cross_validation: {e}")
            with open(f"./results/{label_desc}_error1.pkl", "wb") as f:
                pickle.dump(test_data, f)
            with open(f"./results/{label_desc}_error2.pkl", "wb") as f:
                pickle.dump(train_data, f)
            raise e

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
            cv_scores = cross_val_score(pipeline, train_data, train_score, cv=inner_cv,
                                        scoring=make_scorer(scoring_function_to_use), n_jobs=-1)
            return np.mean(cv_scores)

        # Run Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        best_params = study.best_trial.params

        with open(f"./results/{label_desc}_study.pkl", "wb") as f:
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

        best_pipeline = build_pipeline_from_params(best_params)
        # Retrain the best pipeline on the entire training fold
        best_pipeline.fit(train_data, train_score)
        pred_score = best_pipeline.predict(test_data)

        # If feature selection was used, try to extract the selected features.
        selected_features = None
        if feature_selection_method in ["sfs", "rfe"]:
            # For SFSSelector the attribute is "k_feature_idx_"
            if hasattr(best_pipeline.named_steps['fs'], 'k_feature_idx_'):
                selected_mask = best_pipeline.named_steps['fs'].k_feature_idx_
                selected_features = np.array(train_data.columns)[selected_mask]
            elif hasattr(best_pipeline.named_steps['fs'], 'support_'):
                selected_mask = best_pipeline.named_steps['fs'].support_
                selected_features = np.array(train_data.columns)[selected_mask]

        result = evaluate_predictions(test_score, pred_score, scoring_function_to_use)
        result['best_params'] = best_params
        result['selected_features'] = selected_features
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
        cv_metrics['confusion_matrices'].append(result['confusion_matrix'])
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

    normalized_matrices = []
    for cm in cv_metrics['confusion_matrices']:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        normalized_matrices.append(cm_norm)
    avg_cm = np.mean(normalized_matrices, axis=0)
    std_cm = np.std(normalized_matrices, axis=0)

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
        'confusion_matrices': cv_metrics['confusion_matrices'],
        'feature_selection': sorted([(i, j) for i, j in feature_counts.items()], key=lambda x: x[1], reverse=True),
        'feature_counts': feature_counts,
        'hyperparameter_tuning': param_counts,
        'avg_confusion_matrix': avg_cm,
        'std_confusion_matrix': std_cm
    }

    with open(f"./results/{label_desc}_avg_metrics.pkl", "wb") as f:
        pickle.dump(avg_metrics, f)

def _run_experiment(parameters):
    subset_feature_terms, train_set, cross_task, feature_selection, model_name, clf_func, label, scoring_function = parameters
    subset_df, metadata_df, subset_features = _init_subset_data(train_set, subset_feature_terms, use_metadata_features=False)
    if len(subset_df.columns) < 2:
        print(f"Skipping {label} because it has less than 2 features")
        return
    # Use the new cross-validation initialization (grid_search replaced by Optuna)
    score, inner_cv, pipeline_template = _init_cross_validation(model_name, train_set, cross_task,
                                                                   feature_selection, scoring_function, clf_func, random_state)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    _run_cross_validation(subset_df, metadata_df, score, cross_task, feature_selection, scoring_function,
                          label, subset_features, inner_cv, outer_cv, pipeline_template, model_name, random_state)

feature_levels = ["session"]
feature_objects = ["RightHand"]
feature_types = ["axis", "quat"]
feature_subtypes = ["inter_extrema_intervals_base", "inter_extrema_intervals_low",
                      "inter_extrema_intervals_mid", "inter_extrema_intervals_high"]
all_subset_feature_terms = []
for feature_level in feature_levels:
    for feature_object in feature_objects:
        for feature_type in feature_types:
            for feature_subtype in feature_subtypes:
                all_subset_feature_terms.append([feature_level, feature_object, feature_type, feature_subtype])

train_sets = ["A", "B"]
cross_tasks = [True] #, False]
feature_selections_models = [("rfe", "linsvc")]# [("rfe", "logreg")]#, ("sfs", "linsvc"), ("sfs", "knn")]
clf_funcs = [lambda x: 1 if x > 0 else 0]

parameter_list = []
for subset_feature_terms in all_subset_feature_terms:
    for train_set in train_sets:
        for cross_task in cross_tasks:
            for feature_selection, model_name in feature_selections_models:
                for clf_func in clf_funcs:
                    is_cross_task = "cross_task" if cross_task else "within_task"
                    label = f"{'_'.join(subset_feature_terms)}_{feature_selection}_{model_name}_{train_set}_{is_cross_task}"
                    scoring_function = get_scoring_function("mcc")
                    parameter_list.append([subset_feature_terms, train_set, cross_task, feature_selection,
                                            model_name, clf_func, label, scoring_function])

_ = [
    _run_experiment(parameters) 
    for parameters in 
    tqdm(parameter_list, desc="Running All Experiments")
]