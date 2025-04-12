from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
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
random_state = 25

# Optuna Parameters
total_run_time = 60
trial_run_time = None
trial_num = 5

# Feature Reduction Parameters
na_threshold = 0.2

# SFS Parameters
num_features = 5

# Parallelization Parameters
sfs_n_jobs = 1 # fine set to 1, coarse set to -1
study_n_jobs = 1 # fine set to -1, coarse set to 1

# Manual Feature Search Parameters
override_feature_removal = True
manual_feature_combination = True
manual_feature_search_method = "greedy_fixed_length" # "greedy"
manual_greedy_floating_step = False
manual_max_combination_size = None

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

def _init_cross_validation(run_cross_task, clf_function_to_use, seed):
    if run_cross_task:
        score_A = metadata_dataframe["Score_B_Linear"].apply(clf_function_to_use)
        score_B = metadata_dataframe["Score_A_Linear"].apply(clf_function_to_use)
    else:
        score_A = metadata_dataframe["Score_A_Linear"].apply(clf_function_to_use)
        score_B = metadata_dataframe["Score_B_Linear"].apply(clf_function_to_use)

    inner_cv = (RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
                if run_cross_task 
                else StratifiedKFold(n_splits=4, shuffle=True, random_state=seed))
    
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

    if run_cross_task:
        n_splits = inner_cv.n_splits * inner_cv.n_repeats
    else:
        n_splits = inner_cv.n_splits * outer_cv.n_splits
    trial_run_time = total_run_time / n_splits
    if trial_num is None:
        print(f"Using per trial run time: {trial_run_time}")    
    
    return score_A, score_B, inner_cv, outer_cv

def _run_cross_validation(subset_df_A, subset_df_B, metadata_df_A,
                          metadata_df_B, score_A, score_B,
                          run_cross_task, feature_selection_method,
                          scoring_function_to_use, label_desc, inner_cv,
                          outer_cv, model_name, seed):    
    # Outer CV splits (if run_cross_task then use all data; otherwise use outer_cv splits)
    col_indices = np.arange(len(subset_df_A))
    splits = [(col_indices, col_indices)] if run_cross_task else list(outer_cv.split(subset_df_A, score_A))

    results = []
    split_num = 0
    for train_idx, test_idx in splits:
        split_time = time.time()

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

        imbalance_ratio = np.sum(train_score_A_array == 0) / np.sum(train_score_A_array == 1)

        def objective(trial):            
            if model_name == "linsvc":
                penalty = trial.suggest_categorical("model__penalty", ['l2'])
                C = trial.suggest_float("model__C", 0.001, 0.1, log=True)
                max_iter = trial.suggest_categorical("model__max_iter", [10, 100, 1000, 5000, 10000])
                model_instance = JointEstimator(LinearSVC(penalty=penalty, C=C, class_weight='balanced',
                                           random_state=seed, max_iter=max_iter),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "logreg":
                penalty = trial.suggest_categorical("model__penalty", ['l2'])
                C = trial.suggest_float("model__C", 0.001, 0.1, log=True)
                max_iter = trial.suggest_categorical("model__max_iter", [10, 100, 1000, 5000, 10000])
                solver = trial.suggest_categorical("model__solver", ['liblinear'])
                model_instance = JointEstimator(LogisticRegression(penalty=penalty, C=C, class_weight='balanced',
                                                    random_state=seed, max_iter=max_iter, n_jobs=1, solver=solver),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "knn":
                n_neighbors = trial.suggest_int("model__n_neighbors", 3, 21, step=2)
                metric = trial.suggest_categorical("model__metric", ['euclidean', 'manhattan'])
                weights = trial.suggest_categorical("model__weights", ['uniform', 'distance'])
                model_instance = JointEstimator(KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "random_forest":
                n_estimators = trial.suggest_int("model__n_estimators", 10, 1000, step=10)
                max_depth = trial.suggest_int("model__max_depth", 1, 100)
                model_instance = JointEstimator(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "extra_trees":
                n_estimators = trial.suggest_int("model__n_estimators", 10, 1000, step=10)
                max_depth = trial.suggest_int("model__max_depth", 1, 100)
                min_samples_split = trial.suggest_int("model__min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("model__min_samples_leaf", 1, 5)
                class_weight = trial.suggest_categorical("model__class_weight", ['balanced', None])
                model_instance = JointEstimator(ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                                                     min_samples_leaf=min_samples_leaf, class_weight=class_weight, random_state=seed),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "xgboost":
                n_estimators = trial.suggest_int("model__n_estimators", 10, 1000, step=10)
                max_depth = trial.suggest_int("model__max_depth", 1, 100)
                learning_rate = trial.suggest_float("model__learning_rate", 0.001, 1, log=True)
                # scale_pos_weight = trial.suggest_categorical("model__scale_pos_weight", [imbalance_ratio, imbalance_ratio * 0.5, imbalance_ratio * 0.1, imbalance_ratio * 2, imbalance_ratio * 10])
                scale_pos_weight = trial.suggest_float("model__scale_pos_weight", 0.001, 1000)

                model_instance = JointEstimator(
                    XGBClassifier(
                        n_estimators=n_estimators, max_depth=max_depth,
                        random_state=seed, learning_rate=learning_rate,
                        scale_pos_weight=imbalance_ratio * scale_pos_weight
                    ),
                    train_data_A_array, train_data_B_array,
                    train_score_A_array, train_score_B_array
                )
            else:
                raise ValueError("Unknown model name")

            steps = []
            if feature_selection_method == "sfs":
                floating = trial.suggest_categorical("fs__floating", [False])
                forward = trial.suggest_categorical("fs__forward", [True])
                steps.append(('fs', JointSFSSelector(estimator=model_instance,
                                                k_features=num_features,
                                                forward=forward,
                                                floating=floating,
                                                scoring=make_scorer(scoring_function_to_use),
                                                cv=inner_cv,
                                                fixed_features=None,
                                                feature_groups=None,
                                                n_jobs=sfs_n_jobs)))
                steps.append(('model', model_instance))
            elif feature_selection_method == "rfe":
                n_features_to_select = trial.suggest_int("fs__n_features_to_select", 1, num_features)
                step = trial.suggest_categorical("fs__step", [1, 2, 5, 7, 10])
                rfe = RFE(estimator=model_instance, n_features_to_select=n_features_to_select, step=step)
                steps.append(('fs', rfe))
                steps.append(('model', model_instance))
            elif feature_selection_method == "manual":
                steps.append(('tf', JointDummyTransformer()))
                steps.append(('model', model_instance))
            else:
                raise ValueError("Unknown feature selection method")
            pipeline = Pipeline(steps)

            # Handle RFE later
            if feature_selection_method in ["sfs"]:
                pipeline.fit(train_data_A, train_score_A)
                selected_mask = [i for i in pipeline.named_steps['fs'].k_feature_idx_]
                trial.set_user_attr("selected_features", selected_mask)
                jdtf = JointDummyTransformer()
                jdtf.set_k_feature_idx(selected_mask)
                pipeline.steps.pop(0)
                pipeline.steps.insert(0, ('tf', jdtf))
                
            inner_splits = list(inner_cv.split(train_data_A, train_score_A))
            cv_scores = []
            for inner_train_idx, inner_test_idx in inner_splits:
                inner_train_data_A = train_data_A.iloc[inner_train_idx].to_numpy()
                inner_train_data_B = train_data_B.iloc[inner_train_idx].to_numpy()
                inner_train_score_A = train_score_A.iloc[inner_train_idx].to_numpy()
                inner_train_score_B = train_score_B.iloc[inner_train_idx].to_numpy()
                inner_test_data_A = train_data_A.iloc[inner_test_idx].to_numpy()
                inner_test_data_B = train_data_B.iloc[inner_test_idx].to_numpy()
                inner_test_score_A = train_score_A.iloc[inner_test_idx].to_numpy()
                inner_test_score_B = train_score_B.iloc[inner_test_idx].to_numpy()
                pipeline.steps[1][1].set_data(inner_train_data_A,
                                            inner_train_data_B,
                                            inner_train_score_A,
                                            inner_train_score_B)
                pipeline.fit(train_data_A.iloc[inner_train_idx], train_score_A.iloc[inner_train_idx])
                pipeline.steps[1][1].set_data(inner_test_data_A,
                                            inner_test_data_B,
                                            inner_test_score_A,
                                            inner_test_score_B)
                y_pred = pipeline.predict(train_data_A.iloc[inner_test_idx])
                joint_mcc = scoring_function_to_use(inner_test_score_A, y_pred)
                cv_scores.append(joint_mcc)

            return np.mean(cv_scores)

        # Run Optuna study
        study = optuna.create_study(direction="maximize")
        if trial_num is not None:
            study.optimize(objective, n_trials=trial_num, n_jobs=study_n_jobs)
        else:
            study.optimize(objective, timeout=trial_run_time, n_jobs=study_n_jobs)

        best_trial = study.best_trial
        best_params = best_trial.params
        selected_feature_indices = best_trial.user_attrs.get("selected_features", None)
        if manual_feature_combination is False:
            with open(f"./results_server/{label_desc}_study_{split_num}.pkl", "wb") as f:
                pickle.dump(study, f)
            split_num += 1

        # Helper to rebuild pipeline from best_params
        def build_pipeline_from_params(params, selected_feature_indices=None):
                     
            if model_name == "linsvc":
                penalty = params["model__penalty"]
                C = params["model__C"]
                max_iter = params["model__max_iter"]                
                model_instance = JointEstimator(LinearSVC(penalty=penalty, C=C, class_weight='balanced',
                                           random_state=seed, max_iter=max_iter),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)

            elif model_name == "logreg":
                penalty = params["model__penalty"]
                C = params["model__C"]
                max_iter = params["model__max_iter"]
                solver = params["model__solver"]
                model_instance = JointEstimator(LogisticRegression(penalty=penalty, C=C, class_weight='balanced',
                                                    random_state=seed, max_iter=max_iter, n_jobs=1, solver=solver),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "knn":
                n_neighbors = params["model__n_neighbors"]
                metric = params["model__metric"]
                weights = params["model__weights"]
                model_instance = JointEstimator(KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "random_forest":
                n_estimators = params["model__n_estimators"]
                max_depth = params["model__max_depth"]
                model_instance = JointEstimator(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "extra_trees":
                n_estimators = params["model__n_estimators"]
                max_depth = params["model__max_depth"]
                min_samples_split = params["model__min_samples_split"]
                min_samples_leaf = params["model__min_samples_leaf"]
                class_weight = params["model__class_weight"]
                model_instance = JointEstimator(ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                                                     min_samples_leaf=min_samples_leaf, class_weight=class_weight, random_state=seed),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            elif model_name == "xgboost":
                n_estimators = params["model__n_estimators"]
                max_depth = params["model__max_depth"]
                learning_rate = params["model__learning_rate"]
                scale_pos_weight = params["model__scale_pos_weight"]
                model_instance = JointEstimator(XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                              random_state=seed, learning_rate=learning_rate,
                                                              scale_pos_weight=imbalance_ratio * scale_pos_weight),
                                           train_data_A_array, train_data_B_array,
                                           train_score_A_array, train_score_B_array)
            else:
                raise ValueError("Unknown model name")
            steps = []
            if feature_selection_method == "sfs":
                if selected_feature_indices is not None:
                    jdtf = JointDummyTransformer()
                    jdtf.set_k_feature_idx(selected_feature_indices)
                    steps.append(('tf', jdtf))
                else:
                    raise ValueError("Selected feature indices are not set") 
                steps.append(('model', model_instance))
            elif feature_selection_method == "rfe":
                n_features_to_select = params["fs__n_features_to_select"]
                step = params["fs__step"]
                rfe = RFE(estimator=model_instance, n_features_to_select=n_features_to_select, step=step)
                steps.append(('fs', rfe))
                steps.append(('model', model_instance))
            elif feature_selection_method == "manual":
                steps.append(('tf', JointDummyTransformer()))
                steps.append(('model', model_instance))
            else:
                raise ValueError("Unknown feature selection method")
            return Pipeline(steps)

        best_pipeline = build_pipeline_from_params(best_params, selected_feature_indices)

        unique_feature_names = np.array([i.replace("_A", "") for i in train_data_A.columns])
        selected_feature_names = None
        if selected_feature_indices is not None:
            selected_feature_names = unique_feature_names[selected_feature_indices]
        else:
            selected_feature_names = unique_feature_names

        if run_cross_task:
            raise NotImplementedError("Cross task prediction evaluation not implemented")
            inner_splits = list(inner_cv.split(train_data_A, train_score_A))
            y_preds = []
            for inner_train_idx, inner_test_idx in inner_splits:
                inner_train_data_A = train_data_A.iloc[inner_train_idx].to_numpy()
                inner_train_data_B = train_data_B.iloc[inner_train_idx].to_numpy()
                inner_train_score_A = train_score_A.iloc[inner_train_idx].to_numpy()
                inner_train_score_B = train_score_B.iloc[inner_train_idx].to_numpy()
                inner_test_data_A = train_data_A.iloc[inner_test_idx].to_numpy()
                inner_test_data_B = train_data_B.iloc[inner_test_idx].to_numpy()
                inner_test_score_A = train_score_A.iloc[inner_test_idx].to_numpy()
                inner_test_score_B = train_score_B.iloc[inner_test_idx].to_numpy()
                best_pipeline.steps[1][1].set_data(inner_train_data_A,
                                            inner_train_data_B,
                                            inner_train_score_A,
                                            inner_train_score_B)
                best_pipeline.fit(train_data_A.iloc[inner_train_idx], train_score_A.iloc[inner_train_idx])
                best_pipeline.steps[1][1].set_data(inner_test_data_A,
                                            inner_test_data_B,
                                            inner_test_score_A,
                                            inner_test_score_B)
                y_pred = best_pipeline.predict(train_data_A.iloc[inner_test_idx])
                y_preds.append(y_pred)
        else:
            best_pipeline.fit(train_data_A, train_score_A)
            best_pipeline.steps[1][1].set_data(test_data_A_array,
                                            test_data_B_array,
                                            test_score_A_array,
                                            test_score_B_array)
            y_pred = best_pipeline.predict(test_data_A)

        result = evaluate_predictions(None, y_pred, scoring_function_to_use)
        result['best_params'] = best_params
        result['selected_features'] = selected_feature_names
        result['y_true_A'] = test_score_A
        result['y_pred_A'] = y_pred
        result['y_true_B'] = test_score_B
        result['y_pred_B'] = y_pred
        result['train_PIDs'] = PID[train_idx]
        result['test_PIDs'] = PID[test_idx]
        if not manual_feature_combination:
            print(f"Performance: {result['joint_mcc']} | best_params: {best_params} | selected_features: {selected_feature_names}")
        results.append(result)
        if not manual_feature_combination:
            print(f"Split time: {time.time() - split_time}")

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
        cv_metrics['joint_mcc'].append(result['joint_mcc'])
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
        'joint_mcc': np.mean(cv_metrics['joint_mcc']),
        'joint_mcc_std': np.std(cv_metrics['joint_mcc']),
        'confusion_matrices': cv_metrics['confusion_matrices'],
        'feature_selection': sorted([(i, j) for i, j in feature_counts.items()], key=lambda x: x[1], reverse=True),
        'feature_counts': feature_counts,
        'hyperparameter_tuning': param_counts,
        'avg_confusion_matrix': avg_cm,
        'std_confusion_matrix': std_cm,
        'split_results': results
    }
    
    return avg_metrics

def _run_experiment(parameters):
    start_time = time.time()
    feature_set, cross_task, feature_selection, model_name, clf_func, label, scoring_function = parameters

    combined_feature_set = [item for sublist in feature_set for item in sublist] if all(isinstance(item, list) for item in feature_set) else feature_set

    print(f"Running experiment for {label}")
    subset_df_A, subset_df_B, metadata_df_A, metadata_df_B = _init_subset_data(combined_feature_set, feature_selection, use_metadata_features=False)    
    score_A, score_B, inner_cv, outer_cv = _init_cross_validation(cross_task, clf_func, random_state)

    if feature_selection == "manual" and manual_feature_combination:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        available_features = [i.replace("_A", "") for i in subset_df_A.columns]
        feature_mapping = {j: i for i, j in enumerate(available_features)}
        feature_combination_performance = {}

        max_combo_size = min(manual_max_combination_size, len(available_features)) if manual_max_combination_size is not None else None

        if manual_feature_search_method == "exhaustive":

            if max_combo_size is not None:
                L_range = [max_combo_size]
                num_combinations = math.comb(len(available_features), max_combo_size)
            elif type(feature_set[0]) == list:
                num_combinations = len(feature_set[0]) ** len(feature_set)
            else:
                L_range = list(range(1, len(available_features) + 1))
                num_combinations = 2 ** len(available_features) - 1

            combination_iterators = []
            if type(feature_set[0]) == list:
                combination_iterators = [product(*feature_set)]
            else:
                for L in L_range:
                    combination_iterators.append(combinations(available_features, L))

            pbar = tqdm(total=num_combinations, desc="Current: None -> 0.0 | Best: None -> 0.0")

            best_combo = {'joint_mcc': -float('inf'), 'combo_map': None}
            for combination_iterator in combination_iterators:
                for combo in combination_iterator:
                    combo_map = '_'.join(str(feature_mapping[feature]) for feature in combo)

                    subset_df_A_combo = subset_df_A[[f"{i}_A" for i in combo]]
                    subset_df_B_combo = subset_df_B[[f"{i}_B" for i in combo]]

                    # print(f"Running experiment for {combo_map}")
                    combo_avg_metrics = _run_cross_validation(subset_df_A_combo, subset_df_B_combo, metadata_df_A, metadata_df_B,
                                        score_A, score_B, cross_task, feature_selection,
                                        scoring_function, label, inner_cv, outer_cv, model_name, random_state)
                    
                    if combo_avg_metrics['joint_mcc'] > best_combo['joint_mcc']:
                        best_combo = combo_avg_metrics
                        best_combo['combo_map'] = combo_map

                    progress_bar_str = f"Current: {combo_map} -> {combo_avg_metrics['joint_mcc']} | Best: {best_combo['combo_map']} -> {best_combo['joint_mcc']}"
                    pbar.set_description(progress_bar_str)
                    feature_combination_performance[combo_map] = combo_avg_metrics
                    pbar.update(1)
            pbar.close()
        
        elif manual_feature_search_method == "greedy":
            
            best_overall_combo = []
            best_overall_metric = -float('inf')
            
            max_greedy_size = manual_max_combination_size if manual_max_combination_size is not None else len(available_features)

            if (type(feature_set[0]) == list):
                combos = product(*feature_set)
                max_greedy_size = len(feature_set)
                num_combos = len(feature_set[0]) ** len(feature_set)
            else:
                combos = [feature_set]
                num_combos = 1

            for subset in combos:
                current_features = []
                while len(current_features) < max_greedy_size:
                    candidate_feature = None
                    best_candidate_metric = -float('inf')

                    for candidate in subset:
                        if candidate in current_features:
                            continue

                        candidate_set = current_features + [candidate]
                        candidate_combo_map = '_'.join(str(feature_mapping[i]) for i in candidate_set)
                        
                        subset_df_A_combo = subset_df_A[[f"{i}_A" for i in candidate_set]]
                        subset_df_B_combo = subset_df_B[[f"{i}_B" for i in candidate_set]]
                        
                        combo_avg_metrics = _run_cross_validation(
                            subset_df_A_combo, subset_df_B_combo, metadata_df_A, metadata_df_B,
                            score_A, score_B, cross_task, feature_selection,
                            scoring_function, label, inner_cv, outer_cv, model_name, random_state
                        )
                        feature_combination_performance[candidate_combo_map] = combo_avg_metrics

                        if combo_avg_metrics['joint_mcc'] > best_candidate_metric:
                            best_candidate_metric = combo_avg_metrics['joint_mcc']
                            candidate_feature = candidate
                    
                    if candidate_feature is None:
                        print("No candidate feature available to add; stopping greedy search.")
                        break
                    
                    current_features.append(candidate_feature)
                    print(f"{len(current_features)}/{len(subset)} | Added feature {candidate_feature} with metric: {best_candidate_metric}")

                    # ===== Floating Step: Attempt to remove features if it improves performance =====
                    if manual_greedy_floating_step and len(current_features) > 1:
                        removal_improved = True
                        # Continue removal attempts until no single removal yields improvement.
                        while removal_improved:
                            removal_improved = False
                            # Work on a copy since we might modify current_features during iteration.
                            for feature in current_features.copy():
                                candidate_set = [f for f in current_features if f != feature]
                                candidate_combo_map = '_'.join(str(feature_mapping[i]) for i in candidate_set)
                                
                                subset_df_A_combo = subset_df_A[[f"{i}_A" for i in candidate_set]]
                                subset_df_B_combo = subset_df_B[[f"{i}_B" for i in candidate_set]]
                                
                                combo_avg_metrics = _run_cross_validation(
                                    subset_df_A_combo, subset_df_B_combo, metadata_df_A, metadata_df_B,
                                    score_A, score_B, cross_task, feature_selection,
                                    scoring_function, label, inner_cv, outer_cv, model_name, random_state
                                )
                                feature_combination_performance[candidate_combo_map] = combo_avg_metrics

                                # If removal improves the joint_mcc relative to the current candidate set performance...
                                if combo_avg_metrics['joint_mcc'] > best_candidate_metric:
                                    print(f"Floating removal: Removing feature {feature} improved joint_mcc from {best_candidate_metric} to {combo_avg_metrics['joint_mcc']}")
                                    current_features.remove(feature)
                                    best_candidate_metric = combo_avg_metrics['joint_mcc']
                                    removal_improved = True
                                    # Restart removal evaluation after any change.
                                    break

                    if best_candidate_metric > best_overall_metric:
                        best_overall_metric = best_candidate_metric
                        best_overall_combo = current_features.copy()

            best_combo = {
                'joint_mcc': best_overall_metric,
                'combo_map': '_'.join(str(feature_mapping[i]) for i in best_overall_combo)
            }
            print(f"Best combo found: {best_combo['combo_map']} -> {best_combo['joint_mcc']}")

        elif manual_feature_search_method == "greedy_fixed_length":
            # current_selection = [group[0] for group in feature_set]
            current_selection = [random.choice(group) for group in feature_set]
            best_overall_combo = current_selection.copy()
            best_overall_metric = -float('inf')
            
            # Evaluate the initial combination.
            candidate_combo_map = '_'.join(str(feature_mapping[feat]) for feat in current_selection)
            subset_df_A_combo = subset_df_A[[f"{feat}_A" for feat in current_selection]]
            subset_df_B_combo = subset_df_B[[f"{feat}_B" for feat in current_selection]]
            
            current_metrics = _run_cross_validation(
                subset_df_A_combo, subset_df_B_combo, metadata_df_A, metadata_df_B,
                score_A, score_B, cross_task, feature_selection,
                scoring_function, label, inner_cv, outer_cv, model_name, random_state
            )
            best_overall_metric = current_metrics['joint_mcc']
            print(f"Initial fixed-length combo: {candidate_combo_map} -> {best_overall_metric}")

            direction = 0

            # Greedy coordinate descent: iterate over groups and try to improve the metric by changing one group's feature.
            improved = True
            while improved:
                improved = False

                group_order = list(range(len(feature_set)))

                # If randomizing
                # random.shuffle(group_order)
                # If back to front
                # if direction == 1:
                    # group_order.reverse()
                    # direction = 0
                # else:
                    # direction = 1

                # Iterate over each group (each coordinate).
                for group_idx in group_order:
                    original_feature = current_selection[group_idx]
                    best_candidate_in_group = original_feature
                    # Start with the current overall performance as the baseline for this group's decision.
                    best_candidate_metric_in_group = best_overall_metric
                    
                    # For each candidate in the current group:
                    for candidate in feature_set[group_idx]:
                        if candidate == original_feature:
                            continue
                        
                        # Form a candidate selection by substituting the current group's feature.
                        candidate_selection = current_selection.copy()
                        candidate_selection[group_idx] = candidate
                        candidate_combo_map = '_'.join(str(feature_mapping[feat]) for feat in candidate_selection)
                        
                        # Evaluate the candidate feature set.
                        subset_df_A_combo = subset_df_A[[f"{feat}_A" for feat in candidate_selection]]
                        subset_df_B_combo = subset_df_B[[f"{feat}_B" for feat in candidate_selection]]
                        candidate_metrics = _run_cross_validation(
                            subset_df_A_combo, subset_df_B_combo, metadata_df_A, metadata_df_B,
                            score_A, score_B, cross_task, feature_selection,
                            scoring_function, label, inner_cv, outer_cv, model_name, random_state
                        )
                        
                        # If this candidate improves the metric for this group, update our best candidate.
                        if candidate_metrics['joint_mcc'] > best_candidate_metric_in_group:
                            best_candidate_metric_in_group = candidate_metrics['joint_mcc']
                            best_candidate_in_group = candidate
                    
                    # Update the current selection if a better candidate was found in this group.
                    if best_candidate_in_group != original_feature:
                        print(f"Group {group_idx}: Replacing {original_feature} with {best_candidate_in_group} improved metric to {best_candidate_metric_in_group}")
                        current_selection[group_idx] = best_candidate_in_group
                        if best_candidate_metric_in_group > best_overall_metric:
                            best_overall_metric = best_candidate_metric_in_group
                            best_overall_combo = current_selection.copy()
                        improved = True   # Mark that we have made an improvement.
            
            best_combo = {
                'joint_mcc': best_overall_metric,
                'combo_map': '_'.join(str(feature_mapping[feat]) for feat in best_overall_combo)
            }
            print(f"Best fixed-length combo found: {best_combo['combo_map']} -> {best_combo['joint_mcc']}")

        else:
            raise ValueError("search_method must be either 'exhaustive' or 'greedy'")
            
        inv_feature_mapping = {v: k for k, v in feature_mapping.items()}
        combo_metrics = {
            'map': inv_feature_mapping,
            'performance': feature_combination_performance
        }
        
        with open(f"./results_server/{label}_{manual_feature_search_method}_combo_metrics.pkl", "wb") as f:
            pickle.dump(combo_metrics, f)
    else:
        avg_metrics = _run_cross_validation(subset_df_A, subset_df_B, metadata_df_A, metadata_df_B,
                          score_A, score_B, cross_task, feature_selection,
                          scoring_function, label, inner_cv, outer_cv, model_name, random_state)
        print(f"Final Average MCC: {avg_metrics['joint_mcc']}")
        with open(f"./results_server/{label}_avg_metrics.pkl", "wb") as f:
            pickle.dump(avg_metrics, f)
    print(f"Total run time: {time.time() - start_time}")

config_path = "./experimentConfigs"
experiment_configs = os.listdir(config_path)
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
feature_selections_models = [("manual", "linsvc")] # [("sfs", "xgboost")]
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

_ = [
    _run_experiment(parameters) 
    for parameters in 
    tqdm(parameter_list, desc="Running All Experiments")
]