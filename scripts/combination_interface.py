import pandas as pd
from tqdm import tqdm
from math import comb
from .base import time_method
from itertools import combinations
from multiprocessing import Pool, cpu_count
import os, yaml, logging
from .regressor import *
from .classifier import *

def generate_global_feature_combinations(global_features, required_features, combo_sizes):
    """Generate combinations of global features, always including required features"""
    if not global_features:
        return {'none': {}}
        
    # Ensure required features exist in feature_types
    feature_types = list(global_features.keys())
    required_features = [f for f in required_features if f in feature_types]
    
    # If there are required features, don't include 'none' option
    global_feature_sets = {} if required_features else {'none': {}}
    
    if not required_features:
        # If no required features, generate regular combinations
        for size in combo_sizes:
            if size <= len(feature_types):
                for combo in combinations(feature_types, size):
                    key_name = '_'.join(combo)
                    if len(combo) == len(feature_types):
                        key_name = 'all_global'
                    feature_dict = {feat_type: global_features[feat_type] for feat_type in combo}
                    global_feature_sets[key_name] = feature_dict
        return global_feature_sets
    
    # Get remaining features
    remaining_features = [f for f in feature_types if f not in required_features]
    
    # For each combination size
    for size in combo_sizes:
        if size < len(required_features):
            continue  # Skip if size is smaller than number of required features
            
        # Calculate how many additional features we need
        additional_needed = size - len(required_features)
        
        # Generate combinations of remaining features
        if additional_needed > 0 and remaining_features:
            for combo in combinations(remaining_features, additional_needed):
                # Combine with required features
                full_combo = list(required_features) + list(combo)
                key_name = '_'.join(full_combo)
                if len(full_combo) == len(feature_types):
                    key_name = 'all_global'
                feature_dict = {feat_type: global_features[feat_type] for feat_type in full_combo}
                global_feature_sets[key_name] = feature_dict
        elif additional_needed == 0:
            # If size exactly matches required features, add just those
            key_name = '_'.join(required_features)
            if len(required_features) == len(feature_types):
                key_name = 'all_global'
            feature_dict = {feat_type: global_features[feat_type] for feat_type in required_features}
            global_feature_sets[key_name] = feature_dict
            
    return global_feature_sets

def generate_metadata_combinations(metadata_fields, required_fields, combo_sizes):
    """Generate combinations of metadata fields, always including required fields"""
    if not metadata_fields:
        return {'none': []}
        
    # Ensure required fields exist in metadata_fields
    required_fields = [f for f in required_fields if f in metadata_fields]
    
    # If there are required fields, don't include 'none' option
    metadata_sets = {} if required_fields else {'none': []}
    
    if not required_fields:
        # If no required fields, generate regular combinations
        for size in combo_sizes:
            if size <= len(metadata_fields):
                for combo in combinations(metadata_fields, size):
                    key_name = '_'.join(field.lower() for field in combo)
                    if len(combo) == len(metadata_fields):
                        key_name = 'all'
                    metadata_sets[key_name] = list(combo)
        return metadata_sets
    
    # Get remaining fields
    remaining_fields = [f for f in metadata_fields if f not in required_fields]
    
    # For each combination size
    for size in combo_sizes:
        if size < len(required_fields):
            continue  # Skip if size is smaller than number of required fields
            
        # Calculate how many additional fields we need
        additional_needed = size - len(required_fields)
        
        # Generate combinations of remaining fields
        if additional_needed > 0 and remaining_fields:
            for combo in combinations(remaining_fields, additional_needed):
                # Combine with required fields
                full_combo = list(required_fields) + list(combo)
                key_name = '_'.join(field.lower() for field in full_combo)
                if len(full_combo) == len(metadata_fields):
                    key_name = 'all'
                metadata_sets[key_name] = full_combo
        elif additional_needed == 0:
            # If size exactly matches required fields, add just those
            key_name = '_'.join(field.lower() for field in required_fields)
            if len(required_fields) == len(metadata_fields):
                key_name = 'all'
            metadata_sets[key_name] = list(required_fields)
            
    return metadata_sets

def generate_combinations(config, run_mode):
    """Generate only valid feature combinations that include the required fields.

    This function separates the features into required and optional.
    For each desired combination size s (from config['combination_sizes']),
    it selects only combinations of optional features of size (s - number_of_required_features)
    and unions them with the required features.
    """
    # if run_mode == 'automatic':
    #     return {
    #         'metadata': {'all_metadata_all_global': config.get('metadata_fields', [])},
    #         'global_features': {'all_metadata_all_global': config.get('global_features', {})}
    #     }
    
    # Build the full feature list with type tags.
    # For metadata features, we simply use their name.
    # For global features, we record a tuple (feature_name, stat).
    metadata_fields = config.get('metadata_fields', [])
    global_features_config = config.get('global_features', {})
    
    all_features = []
    for field in metadata_fields:
        all_features.append(('metadata', field))
    for feature_name, feature_config in global_features_config.items():
        for stat in feature_config.get('statistics', []):
            all_features.append(('global', (feature_name, stat)))
            
    # Identify required features.
    required_metadata = set(config.get('required_metadata_fields', []))
    required_global_stats = set()
    for feat in config.get('required_global_features', []):
        if '_' in feat:
            feature_name, stat = feat.rsplit('_', 1)
            required_global_stats.add((feature_name, stat))
    
    # Build the set of required features in the same format as in all_features.
    required_set = set()
    for feature in all_features:
        if feature[0] == 'metadata' and feature[1] in required_metadata:
            required_set.add(feature)
        elif feature[0] == 'global' and feature[1] in required_global_stats:
            required_set.add(feature)


    # (Optional) Warn or raise an error if no required features are found.
    # if not required_set:
    #     raise ValueError("No required features found. Check that your 'required_metadata_fields' "
    #                      "or 'required_global_features' exist in the provided features.")
    
    # The optional features are those not required.
    optional_features = [f for f in all_features if f not in required_set]

    # Get the list of combination sizes from configuration.
    if run_mode == 'manual':
        combo_sizes = config.get('combination_sizes', [len(all_features)])
    else:
        combo_sizes = [len(all_features)]
    
    # Prepare a dictionary to hold the results.
    result_combinations = {'metadata': {}, 'global_features': {}}
    
    # Pre-calculate total combinations (for the progress bar) by summing over sizes.
    total_combinations = 0
    for size in combo_sizes:
        if size < len(required_set):
            continue  # Skip sizes smaller than the number of required features.
        k = size - len(required_set)
        if k > len(optional_features):
            continue
        total_combinations += comb(len(optional_features), k)
    
    # pbar = tqdm(total=total_combinations, desc="Generating feature combinations")
    
    # Generate only valid combinations.
    for size in combo_sizes:
        if size < len(required_set):
            continue  # Skip sizes that cannot include all required features.
        k = size - len(required_set)
        if k > len(optional_features):
            continue

        for optional_combo in combinations(optional_features, k):
            # The valid combination is the union of the required features and the chosen optional ones.
            combo = required_set.union(optional_combo)
            # pbar.update(1)
            
            # Process the combination into separate structures for metadata and global features.
            metadata_list = []
            global_feats = {}
            feature_names = []
            
            for feature in combo:
                if feature[0] == 'metadata':
                    metadata_list.append(feature[1])
                    feature_names.append(feature[1])
                else:  # 'global' feature
                    feature_name, stat = feature[1]
                    if feature_name not in global_feats:
                        global_feats[feature_name] = {
                            'features': [feature_name],
                            'statistics': []
                        }
                        if 'transform' in global_features_config.get(feature_name, {}):
                            global_feats[feature_name]['transform'] = global_features_config[feature_name]['transform']
                    global_feats[feature_name]['statistics'].append(stat)
                    feature_names.append(f"{feature_name}_{stat}")
            
            # Create a key from the sorted list of feature names.
            key = '_'.join(sorted(feature_names))
            result_combinations['metadata'][key] = metadata_list
            result_combinations['global_features'][key] = global_feats
            
    # pbar.close()
    
    if not result_combinations['metadata']:
        raise ValueError("No valid feature combinations were generated. Check your configuration.")
    
    # print(f"Generated {len(result_combinations['metadata'])} feature combinations for sizes {combo_sizes}")
    # print(f"Required metadata: {required_metadata}")
    # print(f"Required global features: {required_global_stats}")
    
    return result_combinations

def run_experiment_wrapper(params):
    """Wrapper function for parallel processing"""
    logging.basicConfig(
        filename='logs/score_prediction.log',
        level=logging.ERROR,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    predictor_type = params.pop('predictor_type')
    predictor_class = RegressionPredictor if predictor_type == 'regression' else ClassificationPredictor
    predictor = predictor_class()
    
    # Extract feature key
    feature_key = params.pop('feature_key')

    # Run the experiment
    result = predictor.run_feature_selection_experiment(**params)
    
    # Add the feature_key back to the result
    result['feature_key'] = feature_key
    
    return result

def generate_experiment_params(feature_config, parameter_config, predictor_type, run_mode):
    """Generator function for experiment parameters"""
    combinations = generate_combinations(feature_config, run_mode)
    
    # Debug print
    # print(f"\nConfig parameters:")
    # print(f"run_mode: {run_mode}")
    # print(f"nan_policy: {parameter_config['nan_policy']}")
    # print(f"train_policy: {parameter_config['train_policy']}")
    # print(f"sample_rates: {parameter_config['sample_rates']}")
    # print(f"segment_policy: {parameter_config['segment_policy']}")
    # print(f"Number of feature combinations: {len(combinations['metadata'])}")
    
    handle_nan = parameter_config['nan_policy']
    train_set = parameter_config['train_policy']
    sample_rate = parameter_config['sample_rates']
    segment_size = parameter_config['segment_policy']

    # for feature_key in combinations['metadata'].keys():
    #     print("Key", feature_key)
    # input()

    # Use the same key for both metadata and global features
    for feature_key in combinations['metadata'].keys():
        yield {
            'predictor_type': predictor_type,
            'handle_nan': handle_nan,
            'train_set': train_set,
            'sample_rate': sample_rate,
            'segment_size': segment_size,
            'feature_key': feature_key,
            'metadata_features': combinations['metadata'][feature_key],
            'global_features': combinations['global_features'][feature_key]
        }

# @time_method("Total Run Time")
def run_experiments():
    """Run all experiments with feature caching"""
    # Load configuration
    with open('parameter_config.yaml', 'r') as f:
        parameter_config = yaml.safe_load(f)

    with open('feature_config.yaml', 'r') as f:
        feature_config = yaml.safe_load(f)

    with open('model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)

    dataset = parameter_config['dataset']
    train_policy = parameter_config['train_policy']
    run_mode = parameter_config['run_mode']
    cross_task = parameter_config['cross_task']
    predictor_type = parameter_config['predictor_type']
    model_type = model_config['model']['type']
    use_resampling = model_config["model"]["use_resampling"]
    threshold = parameter_config['cls_score_threshold']
    inner_cv_k = parameter_config['inner_cv_k']
    outer_cv_k = parameter_config['outer_cv_k']

    tag = f'{dataset}_{train_policy}_{run_mode}_{cross_task}'
    tag += f'_{predictor_type}_{model_type}_{use_resampling}'
    tag += f'_{threshold}_{inner_cv_k}_{outer_cv_k}'

    # Initialize predictor to load data
    predictor_class = RegressionPredictor if predictor_type == 'regression' else ClassificationPredictor
    predictor = predictor_class()
    
    # Generate experiment parameters for feature selection
    experiment_params_list = list(generate_experiment_params(feature_config, parameter_config, predictor_type, run_mode))
    total_experiments = len(experiment_params_list)
    
    # Create results directory    
    results_dir = f'results/{tag}'

    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []

    if run_mode == 'automatic':
        for experiment_params in experiment_params_list:
            result = run_experiment_wrapper(experiment_params)
            all_results.append(result)
            
            # Save intermediate results after each experiment
            metrics_df = pd.DataFrame(all_results)
            metrics_df.to_csv(f'{results_dir}/{run_mode}_results.csv', index=False)
            
    elif run_mode == 'manual':
        pbar = tqdm(total=total_experiments, desc=f"Running {predictor_type} experiments")
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            for result in pool.imap_unordered(run_experiment_wrapper, experiment_params_list):
                all_results.append(result)

                # Save intermediate results after each experiment
                metrics_df = pd.DataFrame(all_results)
                metrics_df.to_csv(f'{results_dir}/{run_mode}_results.csv', index=False)

                pbar.update(1)
        
        pbar.close()

    else:
        raise ValueError(f"Invalid run mode: {run_mode}")
            
    predictor.clear_feature_cache()
    
    # Regular experiment flow for run_mode=False or True
    current_tables = initialize_tables(parameter_config, feature_config, model_config)
    
    if predictor_type == 'regression':
        metrics = ['rmse', 'mae', 'r2', 'mse']
    else:
        metrics = ['accuracy', 'accuracy_std', 'balanced_accuracy',
                   'balanced_accuracy_std', 'f1', 'f1_std', 'precision',
                    'precision_std', 'recall', 'recall_std',
                    'feature_selection_measure', 'feature_selection_measure_std',
                    'sensitivity', 'sensitivity_std', 'specificity', 'specificity_std']
        
    for result in all_results:
        for metric in metrics:
            current_tables[metric][result['feature_key']] = result[metric]
    
    # Save results
    save_results(current_tables, parameter_config, feature_config, model_config)
    return current_tables

def create_score_table(feature_config, parameter_config, model_config):
    """Create an empty DataFrame for storing results, generating only feature combinations
    that include the required fields to avoid combinatorial explosion.
    
    Parameters:
        config (dict): Configuration containing metadata_fields, global_features,
                       required_metadata_fields, required_global_features, and combination_sizes.
        segment_sizes (list): Column labels for the DataFrame.
        run_mode (bool): If True, only one combination (all features) is used.
        
    Returns:
        pd.DataFrame: A DataFrame with row labels corresponding to feature combinations.
    """
    # if run_mode == 'automatic':
    #     # For single run, only create one row with all features.
    #     row_label = "all_metadata_all_global"
    #     return pd.DataFrame(columns=segment_sizes, index=[row_label])
    
    # --- Step 1. Gather all features ---
    # Get metadata fields and global features from the config.
    metadata_fields = feature_config.get('metadata_fields', [])
    global_features_config = feature_config.get('global_features', {})
    
    # Build a list of all feature identifiers.
    # For metadata features, we keep the string as is.
    # For global features, we combine the feature name and statistic (e.g., "globalStat_mean").
    all_features = []
    if metadata_fields:
        all_features.extend(metadata_fields)
    if global_features_config:
        for feature_name, feature_config in global_features_config.items():
            for stat in feature_config.get('statistics', []):
                all_features.append(f"{feature_name}_{stat}")
    
    # --- Step 2. Separate required and optional features ---
    # Get required fields from config.
    required_metadata = set(feature_config.get('required_metadata_fields', []))
    required_global = set(feature_config.get('required_global_features', []))
    required_features = required_metadata.union(required_global)

    if len(required_features) == 0:
        required_features = all_features
    
    # Only include features that actually exist in our feature list.
    available_required = [feat for feat in all_features if feat in required_features]
    optional_features = [feat for feat in all_features if feat not in required_features]
    
    # --- Step 3. Generate only valid feature combinations ---
    # Retrieve the combination sizes from configuration.
    if parameter_config['run_mode'] == 'manual':
        combo_sizes = model_config.get('combination_sizes', [len(all_features)])
    else:
        combo_sizes = [len(all_features)]
    
    row_labels = []
    for size in combo_sizes:
        # The combination must at least include all required features.
        if size < len(available_required):
            continue  # Not enough slots to include all required features.
        additional_needed = size - len(available_required)
        # If the required extra optional features exceed what is available, skip this size.
        if additional_needed > len(optional_features):
            continue
        for optional_combo in combinations(optional_features, additional_needed):
            # Build the combination by uniting the required features and the chosen optional ones.
            combination = list(available_required) + list(optional_combo)
            # Sort for consistent ordering.
            combination_sorted = sorted(combination)
            # Create a unique key by joining feature names with an underscore.
            key = '_'.join(combination_sorted)
            row_labels.append(key)
    
    if not row_labels:
        raise ValueError("No features found for score table. Check your configuration.")
        
    # --- Step 4. Create and return the DataFrame ---
    return pd.DataFrame(index=row_labels)

def initialize_tables(parameter_config, feature_config, model_config):
    """Initialize result tables structure based on predictor type"""
    tables = {}
    if parameter_config['predictor_type'] == 'regression':
        metrics = {'rmse': create_score_table(feature_config, parameter_config, model_config),
                    'mae': create_score_table(feature_config, parameter_config, model_config),
                    'r2': create_score_table(feature_config, parameter_config, model_config),
                    'mse': create_score_table(feature_config, parameter_config, model_config)}
    else:  # classification
        metrics = {'accuracy': create_score_table(feature_config, parameter_config, model_config),
                    'accuracy_std': create_score_table(feature_config, parameter_config, model_config),
                    'balanced_accuracy': create_score_table(feature_config, parameter_config, model_config),
                    'balanced_accuracy_std': create_score_table(feature_config, parameter_config, model_config),
                    'f1': create_score_table(feature_config, parameter_config, model_config),
                    'f1_std': create_score_table(feature_config, parameter_config, model_config),
                    'precision': create_score_table(feature_config, parameter_config, model_config),
                    'precision_std': create_score_table(feature_config, parameter_config, model_config),
                    'recall': create_score_table(feature_config, parameter_config, model_config),
                    'recall_std': create_score_table(feature_config, parameter_config, model_config),
                    'feature_selection_measure': create_score_table(feature_config, parameter_config, model_config),
                    'feature_selection_measure_std': create_score_table(feature_config, parameter_config, model_config),
                    'sensitivity': create_score_table(feature_config, parameter_config, model_config),
                    'sensitivity_std': create_score_table(feature_config, parameter_config, model_config),
                    'specificity': create_score_table(feature_config, parameter_config, model_config),
                    'specificity_std': create_score_table(feature_config, parameter_config, model_config)}
        
    tables = metrics.copy()

    return tables

def save_results(tables, parameter_config, feature_config, model_config):
    """Save results to CSV files"""

    dataset = parameter_config['dataset']
    train_policy = parameter_config['train_policy']
    run_mode = parameter_config['run_mode']
    cross_task = parameter_config['cross_task']
    predictor_type = parameter_config['predictor_type']
    model_type = model_config['model']['type']
    use_resampling = model_config["model"]["use_resampling"]
    threshold = parameter_config['cls_score_threshold']
    inner_cv_k = parameter_config['inner_cv_k']
    outer_cv_k = parameter_config['outer_cv_k']

    tag = f'{dataset}_{train_policy}_{run_mode}_{cross_task}'
    tag += f'_{predictor_type}_{model_type}_{use_resampling}'
    tag += f'_{threshold}_{inner_cv_k}_{outer_cv_k}'

    results_dir = f'results/{tag}'
    
    os.makedirs(results_dir, exist_ok=True)

    metrics = ['rmse', 'mae', 'r2', 'mse'] if predictor_type == 'regression' \
                else ['accuracy', 'accuracy_std', 'balanced_accuracy', 'balanced_accuracy_std', 'f1',
                      'f1_std', 'precision', 'precision_std', 'recall', 'recall_std',
                      'feature_selection_measure', 'feature_selection_measure_std',
                      'sensitivity', 'sensitivity_std', 'specificity', 'specificity_std']
    
    for metric in metrics:
        filename = f'{results_dir}/{metric}_table.csv'
        tables[metric].to_csv(filename)