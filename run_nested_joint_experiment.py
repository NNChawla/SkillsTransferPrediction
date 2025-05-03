from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from greedy_coordinate_descent import group_feature_search
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     StratifiedShuffleSplit, cross_val_score)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from wrappers import *
import pickle, time, gc, sys, os, json, math, random
from itertools import combinations, product
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import optuna

os.makedirs("./results_server", exist_ok=True)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

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

# Preprocessing
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
random_state = 26

# Optuna Parameters
trial_num = 5

# Feature Reduction Parameters
na_threshold = 0.2

# Parallelization Parameters
study_n_jobs = -1

# GCD Parameters
gcd_n_trials = 1

# Manual Feature Search Parameters
override_feature_removal = True
manual_feature_search_method = "greedy"
num_experiment_runs = 20

tab_path = "./experimentData/tabulated_dataframe.pkl"
md_path = "./experimentData/metadata.csv"
with open(tab_path, "rb") as f:
    tabulated_dataframe = pickle.load(f)
metadata_dataframe = pd.read_csv(md_path)

include_set_feature_A, include_set_feature_B = ["RW_Build_Time_A"], ["RW_Build_Time_B"]
exclude_set_feature_A, exclude_set_feature_B = ["RW_Build_Time_B"], ["RW_Build_Time_A"]
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
smdf_A = metadata_df_A[scalable_metadata_features_A].copy()
smdf_B = metadata_df_B[scalable_metadata_features_B].copy()
mdf_A = metadata_df_A.drop(columns=scalable_metadata_features_A + ["PID"])
mdf_B = metadata_df_B.drop(columns=scalable_metadata_features_B + ["PID"])

metadata_binary_features = [i for i in mdf_A.columns if ("-" in i)]
metadata_binary_features = list(set([i.replace("A-", "X-").replace("B-", "X-").replace("_A", "_X").replace("_B", "_X") for i in metadata_binary_features]))
metadata_unique_features = [i for i in mdf_A.columns if not ("-" in i)]
metadata_all_unique_features = metadata_unique_features + metadata_binary_features
metadata_A_features, metadata_B_features = list(zip(*[(i.replace("X-", "A-").replace("_X", "_A"), i.replace("X-", "B-").replace("_X", "_B")) for i in metadata_binary_features]))
metadata_A_features = metadata_unique_features + list(metadata_A_features)
metadata_B_features = metadata_unique_features + list(metadata_B_features)
mdf_A = mdf_A[metadata_A_features]
mdf_B = mdf_B[metadata_B_features]

PID = tabulated_dataframe["PID"]
tdf = tabulated_dataframe.drop(columns=["PID"])
tdf = drop_high_na_columns(tdf, int(tdf.shape[0] * na_threshold))
tdf = variance_filter(tdf)
tdf = pd.concat([PID, tdf], axis=1)
tdf = tdf.iloc[:,1:]
all_feature_names = tdf.columns.to_list()
all_feature_names_set = set(all_feature_names)
unique_features = sorted(list(set(["_".join(i.split('_')[:-1]) for i in all_feature_names])))
unique_features = [i for i in unique_features if ((f"{i}_A" in all_feature_names_set) and (f"{i}_B" in all_feature_names_set))]
features_A = [f"{i}_A" for i in unique_features]
features_B = [f"{i}_B" for i in unique_features]
tdf_A = tdf[features_A]
tdf_B = tdf[features_B]

def get_strings_with_substrings(string_list, target_strings):
    valid_strings = []
    for target_string in target_strings:
        if all(s in target_string for s in string_list):
            valid_strings.append(target_string)
    valid_strings = sorted(valid_strings)
    return valid_strings

def _init_subset_data(subset_features, feature_selection, use_metadata_features=False):
    invalid_features = [i for i in subset_features if (f"{i}_A" not in tdf_A.columns) or (f"{i}_B" not in tdf_B.columns)]
    if len(invalid_features) > 0 and feature_selection != "manual" and not override_feature_removal:
        invalid_features_str = '\n\t'.join(invalid_features)
        print(f"Removing invalid features: {invalid_features_str}")
        subset_features = [i for i in subset_features if (f"{i}_A" in tdf_A.columns) and (f"{i}_B" in tdf_B.columns)]
        subset_features_A = [f"{i}_A" for i in subset_features]
        subset_features_B = [f"{i}_B" for i in subset_features]
        tdf_subset_A = tdf_A[subset_features_A]
        tdf_subset_B = tdf_B[subset_features_B]
    elif len(invalid_features) > 0 and (feature_selection == "manual" or override_feature_removal):
        invalid_features_str = '\n\t'.join(invalid_features)
        print(f"Including invalid features: {invalid_features_str}")
        subset_features_A = [f"{i}_A" for i in subset_features]
        subset_features_B = [f"{i}_B" for i in subset_features]
        tdf_subset_A = tabulated_dataframe[subset_features_A]
        tdf_subset_B = tabulated_dataframe[subset_features_B]
    else:
        print(f"No invalid features found")
        subset_features_A = [f"{i}_A" for i in subset_features]
        subset_features_B = [f"{i}_B" for i in subset_features]
        tdf_subset_A = tdf_A[subset_features_A]
        tdf_subset_B = tdf_B[subset_features_B]

    metadata_df_A = None
    metadata_df_B = None
    if use_metadata_features:
        tdf_subset_A = pd.concat([tdf_subset_A, smdf_A], axis=1)
        tdf_subset_B = pd.concat([tdf_subset_B, smdf_B], axis=1)

        metadata_df_A = mdf_A
        metadata_df_B = mdf_B

    return tdf_subset_A, tdf_subset_B, metadata_df_A, metadata_df_B

def _init_cross_validation(clf_function_to_use, seed):
    score_A = metadata_dataframe["Score_A_Linear"].apply(clf_function_to_use)
    score_B = metadata_dataframe["Score_B_Linear"].apply(clf_function_to_use)

    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    return score_A, score_B, inner_cv, outer_cv

def _run_nested_cross_validation(subset_df_A, subset_df_B, metadata_df_A,
                          metadata_df_B, score_A, score_B,
                          scoring_function_to_use, inner_cv,
                          outer_cv, feature_set, seed):
    # Outer CV splits (if run_cross_task then use all data; otherwise use outer_cv splits)
    col_indices = np.arange(len(subset_df_A))
    outer_splits = list(outer_cv.split(subset_df_A, score_A))
    
    available_features = [i.replace("_A", "") for i in subset_df_A.columns]
    feature_mapping = {j: i for i, j in enumerate(available_features)}

    outer_results = []
    counter = 0
    for train_idx, test_idx in outer_splits:
        split_time = time.time()
        # print(f"Starting outer split {counter} of {len(outer_splits)}")
        counter += 1

        # Prepare training data
        train_data_A = subset_df_A.iloc[train_idx].copy()
        train_data_A = scale_impute_df(train_data_A, scaler, imputer)
        if metadata_df_A is not None:
            train_metadata_A = metadata_df_A.iloc[train_idx].copy()
            train_data_A = pd.concat([train_data_A, train_metadata_A], axis=1)
        train_score_A = score_A.iloc[train_idx].copy()

        train_data_B = subset_df_B.iloc[train_idx].copy()
        train_data_B = scale_impute_df(train_data_B, scaler, imputer)
        if metadata_df_B is not None:
            train_metadata_B = metadata_df_B.iloc[train_idx].copy()
            train_data_B = pd.concat([train_data_B, train_metadata_B], axis=1)
        train_score_B = score_B.iloc[train_idx].copy()

        # Prepare testing data
        test_data_A = subset_df_A.iloc[test_idx].copy()
        test_data_A = scale_impute_df(test_data_A, scaler, imputer)
        if metadata_df_A is not None:
            test_metadata_A = metadata_df_A.iloc[test_idx].copy()
            test_data_A = pd.concat([test_data_A, test_metadata_A], axis=1)
        test_score_A = score_A.iloc[test_idx].copy()

        test_data_B = subset_df_B.iloc[test_idx].copy()
        test_data_B = scale_impute_df(test_data_B, scaler, imputer)
        if metadata_df_B is not None:
            test_metadata_B = metadata_df_B.iloc[test_idx].copy()
            test_data_B = pd.concat([test_data_B, test_metadata_B], axis=1)
        test_score_B = score_B.iloc[test_idx].copy()

        train_data_A_array = train_data_A.to_numpy(copy=True)
        train_data_B_array = train_data_B.to_numpy(copy=True)
        train_score_A_array = train_score_A.to_numpy(copy=True)
        train_score_B_array = train_score_B.to_numpy(copy=True)
        test_data_A_array = test_data_A.to_numpy(copy=True)
        test_data_B_array = test_data_B.to_numpy(copy=True)
        test_score_A_array = test_score_A.to_numpy(copy=True)
        test_score_B_array = test_score_B.to_numpy(copy=True)

        def objective(trial):
            penalty = trial.suggest_categorical("model__penalty", ['l2'])
            C = trial.suggest_float("model__C", 0.001, 0.01, log=True)
            loss = trial.suggest_categorical("model__loss", ['squared_hinge'])
            max_iter = trial.suggest_categorical("model__max_iter", [10, 100, 1000, 5000])
            model_instance = JointEstimator(LinearSVC(penalty=penalty, C=C, class_weight='balanced',
                                        random_state=seed, max_iter=max_iter, loss=loss),
                                        train_data_A_array, train_data_B_array,
                                        train_score_A_array, train_score_B_array)

            steps = []
            steps.append(('tf', JointDummyTransformer()))
            steps.append(('model', model_instance))
            pipeline = Pipeline(steps)
                
            inner_splits = list(inner_cv.split(train_data_A, train_score_A))
            inner_results = []
            for inner_train_idx, inner_test_idx in inner_splits:

                inner_train_data_A = train_data_A.iloc[inner_train_idx]
                inner_train_data_B = train_data_B.iloc[inner_train_idx]
                inner_train_score_A = train_score_A.iloc[inner_train_idx]
                inner_train_score_B = train_score_B.iloc[inner_train_idx]
                inner_test_data_A = train_data_A.iloc[inner_test_idx]
                inner_test_data_B = train_data_B.iloc[inner_test_idx]
                inner_test_score_A = train_score_A.iloc[inner_test_idx]
                inner_test_score_B = train_score_B.iloc[inner_test_idx]

                gcd_features = group_feature_search(
                    pipeline,
                    inner_train_data_A,
                    inner_train_data_B,
                    inner_train_score_A,
                    inner_train_score_B,
                    scoring_function_to_use,
                    inner_cv,
                    feature_set,
                    feature_mapping,
                    n_trials=gcd_n_trials,
                    mode=manual_feature_search_method
                )

                inner_train_data_A = inner_train_data_A[[f"{feat}_A" for feat in gcd_features]]
                inner_train_data_B = inner_train_data_B[[f"{feat}_B" for feat in gcd_features]]
                inner_train_data_A_array = inner_train_data_A.to_numpy()
                inner_train_data_B_array = inner_train_data_B.to_numpy()
                inner_train_score_A_array = inner_train_score_A.to_numpy()
                inner_train_score_B_array = inner_train_score_B.to_numpy()
                inner_test_data_A = inner_test_data_A[[f"{feat}_A" for feat in gcd_features]]
                inner_test_data_B = inner_test_data_B[[f"{feat}_B" for feat in gcd_features]]
                inner_test_data_A_array = inner_test_data_A.to_numpy()
                inner_test_data_B_array = inner_test_data_B.to_numpy()
                inner_test_score_A_array = inner_test_score_A.to_numpy()
                inner_test_score_B_array = inner_test_score_B.to_numpy()

                pipeline.steps[1][1].set_data(inner_train_data_A_array,
                                            inner_train_data_B_array,
                                            inner_train_score_A_array,
                                            inner_train_score_B_array)
                pipeline.fit(inner_train_data_A, inner_train_score_A)

                pipeline.steps[1][1].set_data(inner_test_data_A_array,
                                            inner_test_data_B_array,
                                            inner_test_score_A_array,
                                            inner_test_score_B_array)
                y_pred = pipeline.predict(inner_test_data_A)

                result = evaluate_predictions(None, y_pred, scoring_function_to_use)
                result['y_true_A'] = test_score_A
                result['y_pred_A'] = y_pred
                result['y_true_B'] = test_score_B
                result['y_pred_B'] = y_pred
                result['train_PIDs'] = PID[train_idx]
                result['test_PIDs'] = PID[test_idx]
                inner_results.append(result)

            trial.set_user_attr("inner_results", inner_results)

            return np.mean([result['joint_mcc'] for result in inner_results])

        # Run Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trial_num, n_jobs=study_n_jobs)

        best_trial = study.best_trial
        best_params = best_trial.params
        inner_results = best_trial.user_attrs.get("inner_results", None)

        if inner_results is None:
            raise ValueError("No results found")

        def build_pipeline_from_params(params):
            penalty = params["model__penalty"]
            C = params["model__C"]
            loss = params["model__loss"]
            max_iter = params["model__max_iter"]                
            model_instance = JointEstimator(LinearSVC(penalty=penalty, C=C, class_weight='balanced',
                                        random_state=seed, max_iter=max_iter, loss=loss),
                                        train_data_A_array, train_data_B_array,
                                        train_score_A_array, train_score_B_array)
           
            steps = []
            steps.append(('tf', JointDummyTransformer()))
            steps.append(('model', model_instance))
            return Pipeline(steps)

        best_pipeline = build_pipeline_from_params(best_params)

        gcd_features = group_feature_search(
            best_pipeline,
            train_data_A,
            train_data_B,
            train_score_A,
            train_score_B,
            scoring_function_to_use,
            inner_cv,
            feature_set,
            feature_mapping,
            n_trials=gcd_n_trials,
            mode=manual_feature_search_method
        )

        selected_train_data_A = train_data_A[[f"{feat}_A" for feat in gcd_features]]
        selected_train_data_B = train_data_B[[f"{feat}_B" for feat in gcd_features]]
        selected_train_data_A_array = selected_train_data_A.to_numpy()
        selected_train_data_B_array = selected_train_data_B.to_numpy()
        train_score_A_array = train_score_A.to_numpy()
        train_score_B_array = train_score_B.to_numpy()
        selected_test_data_A = test_data_A[[f"{feat}_A" for feat in gcd_features]]
        selected_test_data_B = test_data_B[[f"{feat}_B" for feat in gcd_features]]
        selected_test_data_A_array = selected_test_data_A.to_numpy()
        selected_test_data_B_array = selected_test_data_B.to_numpy()
        test_score_A_array = test_score_A.to_numpy()
        test_score_B_array = test_score_B.to_numpy()

        best_pipeline.steps[1][1].set_data(selected_train_data_A_array,
                                    selected_train_data_B_array,
                                    train_score_A_array,
                                    train_score_B_array)
        best_pipeline.fit(selected_train_data_A, train_score_A)
        best_pipeline.steps[1][1].set_data(selected_test_data_A_array,
                                    selected_test_data_B_array,
                                    test_score_A_array,
                                    test_score_B_array)
        y_pred = best_pipeline.predict(selected_test_data_A)

        result = evaluate_predictions(None, y_pred, scoring_function_to_use)
        result['best_params'] = best_params
        result['selected_features'] = gcd_features
        result['y_true_A'] = test_score_A
        result['y_pred_A'] = y_pred
        result['y_true_B'] = test_score_B
        result['y_pred_B'] = y_pred
        result['train_PIDs'] = PID[train_idx]
        result['test_PIDs'] = PID[test_idx]
        result['hypopt_results'] = inner_results
        outer_results.append(result)

        gcd_features_map = '_'.join(str(feature_mapping[feat]) for feat in gcd_features)
        print(f"Nested CV Outer Fold Performance: {result['joint_mcc']} | best_params: {best_params} | selected_features: {gcd_features_map}")
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
        'joint_mcc': [],
        'confusion_matrices': [],
        'selected_features': [],
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
        outer_cv_metrics['joint_mcc'].append(result['joint_mcc'])
        outer_cv_metrics['confusion_matrices'].append(result['confusion_matrix'])
        outer_cv_metrics['selected_features'].append(result['selected_features'])
        outer_cv_metrics['best_params'].append(result['best_params'])

    feature_counts = {}
    param_counts = {}

    for params in outer_cv_metrics['best_params']:
        for param, value in params.items():
            if param not in param_counts:
                param_counts[param] = {}
            param_counts[param][value] = param_counts[param].get(value, 0) + 1

    for features in outer_cv_metrics['selected_features']:
        if features is not None:
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

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
        'joint_mcc': np.mean(outer_cv_metrics['joint_mcc']),
        'joint_mcc_std': np.std(outer_cv_metrics['joint_mcc']),
        'confusion_matrices': outer_cv_metrics['confusion_matrices'],
        'feature_selection': sorted([(i, j) for i, j in feature_counts.items()], key=lambda x: x[1], reverse=True),
        'feature_counts': feature_counts,
        'hyperparameter_tuning': param_counts,
        'avg_confusion_matrix': avg_cm,
        'std_confusion_matrix': std_cm,
        'nested_results': outer_results
    }
    
    return avg_metrics

def _run_final_cross_validation(subset_df_A, subset_df_B, metadata_df_A, metadata_df_B,
                               score_A, score_B, scoring_function, outer_cv,
                               nested_results, random_state):

    col_indices = np.arange(len(subset_df_A))
    outer_splits = list(outer_cv.split(subset_df_A, score_A))
    
    available_features = [i.replace("_A", "") for i in subset_df_A.columns]
    feature_mapping = {j: i for i, j in enumerate(available_features)}

    final_results = []

    for nested_result_idx, nested_result in enumerate(nested_results):
        hypopt_params = nested_result['best_params']
        fs_features = nested_result['selected_features']

        cv_results = []
        for train_idx, test_idx in outer_splits:

            # Prepare training data
            train_data_A = subset_df_A[[f"{feat}_A" for feat in fs_features]].iloc[train_idx].copy()
            train_data_A = scale_impute_df(train_data_A, scaler, imputer)
            if metadata_df_A is not None:
                train_metadata_A = metadata_df_A.iloc[train_idx].copy()
                train_data_A = pd.concat([train_data_A, train_metadata_A], axis=1)
            train_score_A = score_A.iloc[train_idx].copy()

            train_data_B = subset_df_B[[f"{feat}_B" for feat in fs_features]].iloc[train_idx].copy()
            train_data_B = scale_impute_df(train_data_B, scaler, imputer)
            if metadata_df_B is not None:
                train_metadata_B = metadata_df_B.iloc[train_idx].copy()
                train_data_B = pd.concat([train_data_B, train_metadata_B], axis=1)
            train_score_B = score_B.iloc[train_idx].copy()

            # Prepare testing data
            test_data_A = subset_df_A[[f"{feat}_A" for feat in fs_features]].iloc[test_idx].copy()
            test_data_A = scale_impute_df(test_data_A, scaler, imputer)
            if metadata_df_A is not None:
                test_metadata_A = metadata_df_A.iloc[test_idx].copy()
                test_data_A = pd.concat([test_data_A, test_metadata_A], axis=1)
            test_score_A = score_A.iloc[test_idx].copy()

            test_data_B = subset_df_B[[f"{feat}_B" for feat in fs_features]].iloc[test_idx].copy()
            test_data_B = scale_impute_df(test_data_B, scaler, imputer)
            if metadata_df_B is not None:
                test_metadata_B = metadata_df_B.iloc[test_idx].copy()
                test_data_B = pd.concat([test_data_B, test_metadata_B], axis=1)
            test_score_B = score_B.iloc[test_idx].copy()

            train_data_A_array = train_data_A.to_numpy(copy=True)
            train_data_B_array = train_data_B.to_numpy(copy=True)
            train_score_A_array = train_score_A.to_numpy(copy=True)
            train_score_B_array = train_score_B.to_numpy(copy=True)
            test_data_A_array = test_data_A.to_numpy(copy=True)
            test_data_B_array = test_data_B.to_numpy(copy=True)
            test_score_A_array = test_score_A.to_numpy(copy=True)
            test_score_B_array = test_score_B.to_numpy(copy=True)

            def build_pipeline_from_params(params):
                penalty = params["model__penalty"]
                C = params["model__C"]
                loss = params["model__loss"]
                max_iter = params["model__max_iter"]                
                model_instance = JointEstimator(LinearSVC(penalty=penalty, C=C, class_weight='balanced',
                                            random_state=random_state, max_iter=max_iter, loss=loss),
                                            train_data_A_array, train_data_B_array,
                                            train_score_A_array, train_score_B_array)
            
                steps = []
                steps.append(('tf', JointDummyTransformer()))
                steps.append(('model', model_instance))
                return Pipeline(steps)

            best_pipeline = build_pipeline_from_params(hypopt_params)

            best_pipeline.steps[1][1].set_data(train_data_A_array,
                                        train_data_B_array,
                                        train_score_A_array,
                                        train_score_B_array)
            best_pipeline.fit(train_data_A, train_score_A)
            best_pipeline.steps[1][1].set_data(test_data_A_array,
                                        test_data_B_array,
                                        test_score_A_array,
                                        test_score_B_array)
            y_pred = best_pipeline.predict(test_data_A)

            result = evaluate_predictions(None, y_pred, scoring_function)
            result['selected_params'] = hypopt_params
            result['selected_features'] = fs_features
            result['selected_features_map'] = '_'.join(str(feature_mapping[feat]) for feat in fs_features)
            result['y_true_A'] = test_score_A
            result['y_pred_A'] = y_pred
            result['y_true_B'] = test_score_B
            result['y_pred_B'] = y_pred
            result['train_PIDs'] = PID[train_idx]
            result['test_PIDs'] = PID[test_idx]
            cv_results.append(result)
        
        # print(f"K-Fold CV Performance: {result['joint_mcc']} | selected_params: {hypopt_params} | selected_features: {fs_features}")
        # print(f"Split time: {time.time() - split_time}")

        # Aggregate CV metrics
        cv_metrics = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'sensitivity': [],
            'specificity': [],
            'joint_mcc': [],
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
            cv_metrics['joint_mcc'].append(result['joint_mcc'])
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
            'joint_mcc': np.mean(cv_metrics['joint_mcc']),
            'joint_mcc_std': np.std(cv_metrics['joint_mcc']),
            'confusion_matrices': cv_metrics['confusion_matrices'],
            'avg_confusion_matrix': avg_cm,
            'std_confusion_matrix': std_cm,
            'kfold_results': cv_results
        }

        final_results.append(kfold_metrics)
    
    return final_results

def _run_experiment(parameters, best_score):
    start_time = time.time()
    feature_set, cross_task, feature_selection, model_name, clf_func, label, scoring_function = parameters

    combined_feature_set = [item for sublist in feature_set for item in sublist] if all(isinstance(item, list) for item in feature_set) else feature_set

    print(f"Running experiment for {label}")
    subset_df_A, subset_df_B, metadata_df_A, metadata_df_B = _init_subset_data(combined_feature_set, feature_selection, use_metadata_features=False)    
    score_A, score_B, inner_cv, outer_cv = _init_cross_validation(clf_func, random_state)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    nested_metrics = _run_nested_cross_validation(subset_df_A, subset_df_B, metadata_df_A, metadata_df_B,
                        score_A, score_B, scoring_function, inner_cv, outer_cv, feature_set, random_state)

    final_results = _run_final_cross_validation(subset_df_A, subset_df_B, metadata_df_A, metadata_df_B,
                        score_A, score_B, scoring_function, outer_cv, nested_metrics['nested_results'], random_state)

    final_results_metrics = []
    for i, result in enumerate(final_results):
        final_results_metrics.append(result['joint_mcc'])
        final_features_map = result['kfold_results'][0]['selected_features_map']
        final_params = result['kfold_results'][0]['selected_params']
        print(f"Final K-Fold {i} MCC: {result['joint_mcc']} | Features: {final_features_map} | Params: {final_params}")

    print(f"Final Best MCC: {max(final_results_metrics)}")
    print(f"Final Best Params: {final_results[np.argmax(final_results_metrics)]['kfold_results'][0]['selected_params']}")
    print(f"Final Best Features: {final_results[np.argmax(final_results_metrics)]['kfold_results'][0]['selected_features']}")

    if max(final_results_metrics) > best_score:
        best_score = max(final_results_metrics)
        with open(f"./results_server/{label}_nested_metrics.pkl", "wb") as f:
            pickle.dump(nested_metrics, f)
        with open(f"./results_server/{label}_final_results.pkl", "wb") as f:
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

    experiment_features = []
    for line in config:
        if type(line) == list:
            if line[0] == "prod":
                line.pop(0)
                experiment_features.append(line)
            else:
                experiment_features.extend(get_strings_with_substrings(line, unique_features))
        elif type(line) == str:
            experiment_features.append(line)

    feature_sets.append(experiment_features)

cross_tasks = [False]
feature_selections_models = [("manual", "linsvc")]
clf_funcs = [lambda x: 0 if x > 0 else 1]
scoring_function = joint_scorer

parameter_list = []
for feature_set, feature_label in zip(feature_sets, feature_labels):
    for cross_task in cross_tasks:
        for feature_selection, model_name in feature_selections_models:
            for clf_func in clf_funcs:
                is_cross_task = "cross_task" if cross_task else "within_task"
                label = f"{feature_label}_{feature_selection}_{model_name}_{is_cross_task}"
                parameter_list.append([feature_set, cross_task, feature_selection,
                                        model_name, clf_func, label, scoring_function])

for parameters in parameter_list:
    print(f"Running {parameters[5]}")
    best_score = -np.inf
    for experiment_num in range(num_experiment_runs):
        best_score = _run_experiment(parameters, best_score)
        print(f"{parameters[5]} | Experiment {experiment_num} best score: {best_score}")