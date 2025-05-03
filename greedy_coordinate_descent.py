import random, math
from tqdm import tqdm
from itertools import product
import numpy as np

def group_feature_search(
    pipeline,
    train_data_A,
    train_data_B,
    train_score_A,
    train_score_B,
    scoring_function,
    inner_cv,
    feature_set,
    feature_mapping,
    n_trials = 1,
    mode = "greedy" # "greedy" or "exhaustive"
    ):

    best_score = -np.inf
    best_result = None

    if mode == "greedy":
        for trial_num in range(n_trials):
            best_score, best_result = greedy_coordinate_descent(pipeline, train_data_A, train_data_B, train_score_A, train_score_B, scoring_function, inner_cv, feature_set, feature_mapping, best_score, best_result)
        # print(f"Trial {trial_num} best score: {best_score}")
    elif mode == "exhaustive":
        best_score, best_result = exhaustive_feature_search(pipeline, train_data_A, train_data_B, train_score_A, train_score_B, scoring_function, inner_cv, feature_set, feature_mapping, best_score, best_result)
    return best_result

def greedy_coordinate_descent(
    pipeline,
    train_data_A,
    train_data_B,
    train_score_A,
    train_score_B,
    scoring_function,
    inner_cv,
    feature_set,
    feature_mapping,
    best_score=-np.inf,
    best_result=None
    ):

    def _run_inner_cv(subset_df_A, subset_df_B):
        inner_splits = list(inner_cv.split(subset_df_A, train_score_A))
        cv_scores = []
        for inner_train_idx, inner_test_idx in inner_splits:
            inner_train_data_A = subset_df_A.iloc[inner_train_idx].to_numpy()
            inner_train_data_B = subset_df_B.iloc[inner_train_idx].to_numpy()
            inner_train_score_A = train_score_A.iloc[inner_train_idx].to_numpy()
            inner_train_score_B = train_score_B.iloc[inner_train_idx].to_numpy()
            inner_test_data_A = subset_df_A.iloc[inner_test_idx].to_numpy()
            inner_test_data_B = subset_df_B.iloc[inner_test_idx].to_numpy()
            inner_test_score_A = train_score_A.iloc[inner_test_idx].to_numpy()
            inner_test_score_B = train_score_B.iloc[inner_test_idx].to_numpy()
            pipeline.steps[1][1].set_data(inner_train_data_A,
                                        inner_train_data_B,
                                        inner_train_score_A,
                                        inner_train_score_B)
            pipeline.fit(subset_df_A.iloc[inner_train_idx], train_score_A.iloc[inner_train_idx])
            pipeline.steps[1][1].set_data(inner_test_data_A,
                                        inner_test_data_B,
                                        inner_test_score_A,
                                        inner_test_score_B)
            y_pred = pipeline.predict(subset_df_A.iloc[inner_test_idx])
            joint_mcc = scoring_function(None, y_pred)
            cv_scores.append(joint_mcc)
        return np.mean(cv_scores)

    feature_combination_performance = {}

    current_selection = [random.choice(group) for group in feature_set]
    best_overall_combo = current_selection.copy()
    best_overall_metric = -float('inf')
    
    # Evaluate the initial combination.
    candidate_combo_map = '_'.join(str(feature_mapping[feat]) for feat in current_selection)
    train_data_A_combo = train_data_A[[f"{feat}_A" for feat in current_selection]]
    train_data_B_combo = train_data_B[[f"{feat}_B" for feat in current_selection]]
    best_overall_metric = _run_inner_cv(train_data_A_combo, train_data_B_combo)

    # print(f"Initial fixed-length combo: {candidate_combo_map} -> {best_overall_metric}")

    direction = 1
    
    # Greedy coordinate descent: iterate over groups and try to improve the metric by changing one group's feature.
    improved = True
    while improved:
        improved = False

        group_order = list(range(len(feature_set)))

        # If randomizing
        # random.shuffle(group_order)
        # If back to front
        # if direction == 1:
        #     group_order.reverse()
        #     direction = 0
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
                train_data_A_combo = train_data_A[[f"{feat}_A" for feat in candidate_selection]]
                train_data_B_combo = train_data_B[[f"{feat}_B" for feat in candidate_selection]]
                candidate_joint_mcc = _run_inner_cv(train_data_A_combo, train_data_B_combo)
                
                # If this candidate improves the metric for this group, update our best candidate.
                if candidate_joint_mcc > best_candidate_metric_in_group:
                    feature_combination_performance[candidate_combo_map] = candidate_joint_mcc
                    best_candidate_metric_in_group = candidate_joint_mcc
                    best_candidate_in_group = candidate
            
            # Update the current selection if a better candidate was found in this group.
            if best_candidate_in_group != original_feature:
                # print(f"Group {group_idx}: Replacing {original_feature} with {best_candidate_in_group} improved metric to {best_candidate_metric_in_group}")
                current_selection[group_idx] = best_candidate_in_group
                if best_candidate_metric_in_group > best_overall_metric:
                    best_overall_metric = best_candidate_metric_in_group
                    best_overall_combo = current_selection.copy()
                improved = True   # Mark that we have made an improvement.

    if best_overall_metric > best_score:
        best_score = best_overall_metric
        best_result = best_overall_combo
    
    return best_score, best_result

def exhaustive_feature_search(
    pipeline,
    train_data_A,
    train_data_B,
    train_score_A,
    train_score_B,
    scoring_function,
    inner_cv,
    feature_set,        # List[List[str]]: groups of feature names (without "_A"/"_B")
    feature_mapping,    # Dict[str, any]: maps feature → some ID for logging
    best_score=-np.inf,
    best_result=None
):
    """
    Exhaustively evaluate every combination picking one feature from each group.

    Returns:
        best_score (float): the highest CV score found
        best_result (List[str]): the feature names achieving that score
        all_performances (Dict[str, float]): mapping of combo_key -> CV score
    """
    # ---- Inner CV evaluator (identical to your greedy version) ----
    def _run_inner_cv(subset_df_A, subset_df_B):
        splits = list(inner_cv.split(subset_df_A, train_score_A))
        cv_scores = []
        for train_idx, test_idx in splits:
            # slice out the numpy arrays for your custom estimator
            XtrA = subset_df_A.iloc[train_idx].to_numpy()
            XtrB = subset_df_B.iloc[train_idx].to_numpy()
            ytrA = train_score_A.iloc[train_idx].to_numpy()
            ytrB = train_score_B.iloc[train_idx].to_numpy()
            XteA = subset_df_A.iloc[test_idx].to_numpy()
            XteB = subset_df_B.iloc[test_idx].to_numpy()
            yteA = train_score_A.iloc[test_idx].to_numpy()
            yteB = train_score_B.iloc[test_idx].to_numpy()

            # tell your custom step about the inner‐train data
            pipeline.steps[1][1].set_data(XtrA, XtrB, ytrA, ytrB)
            pipeline.fit(subset_df_A.iloc[train_idx], train_score_A.iloc[train_idx])

            # tell it about the inner‐test data & predict
            pipeline.steps[1][1].set_data(XteA, XteB, yteA, yteB)
            y_pred = pipeline.predict(subset_df_A.iloc[test_idx])

            # compute your joint score (e.g. MCC)
            cv_scores.append(scoring_function(None, y_pred))

        return float(np.mean(cv_scores))

    all_performances = {}
    best_score = best_score
    best_result = best_result

    group_sizes = [len(group) for group in feature_set]
    total_combos = math.prod(group_sizes)

    # ---- Enumerate EVERY combination ----
    for combo in tqdm(product(*feature_set), total=total_combos, desc="Exhaustive feature search"):
        # build a string key for logging ("1_4_2_..." or feature IDs)
        combo_key = "_".join(str(feature_mapping[f]) for f in combo)

        # slice the DataFrame to only the columns we need
        dfA = train_data_A[[f"{f}_A" for f in combo]]
        dfB = train_data_B[[f"{f}_B" for f in combo]]

        # evaluate
        score = _run_inner_cv(dfA, dfB)
        all_performances[combo_key] = score

        # update best
        if score > best_score:
            best_score = score
            best_result = list(combo)

    return best_score, best_result