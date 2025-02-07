import os
import yaml
import time
import logging
import numpy as np
import pandas as pd
from math import comb
from itertools import product, combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from score_prediction import RegressionPredictor, ClassificationPredictor

def create_score_table(config, segment_sizes=['full'], single_run=False):
    """Create an empty DataFrame for storing results, generating only feature combinations
    that include the required fields to avoid combinatorial explosion.
    
    Parameters:
        config (dict): Configuration containing metadata_fields, global_features,
                       required_metadata_fields, required_global_features, and combination_sizes.
        segment_sizes (list): Column labels for the DataFrame.
        single_run (bool): If True, only one combination (all features) is used.
        
    Returns:
        pd.DataFrame: A DataFrame with row labels corresponding to feature combinations.
    """
    if single_run:
        # For single run, only create one row with all features.
        row_label = "all_metadata_all_global"
        return pd.DataFrame(columns=segment_sizes, index=[row_label])
    
    # --- Step 1. Gather all features ---
    # Get metadata fields and global features from the config.
    metadata_fields = config.get('metadata_fields', [])
    global_features_config = config.get('global_features', {})
    
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
    required_metadata = set(config.get('required_metadata_fields', []))
    required_global = set(config.get('required_global_features', []))
    required_features = required_metadata.union(required_global)
    
    # Only include features that actually exist in our feature list.
    available_required = [feat for feat in all_features if feat in required_features]
    optional_features = [feat for feat in all_features if feat not in required_features]
    
    # --- Step 3. Generate only valid feature combinations ---
    # Retrieve the combination sizes from configuration.
    combo_sizes = config.get('combination_sizes', [len(all_features)])
    
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
    return pd.DataFrame(columns=segment_sizes, index=row_labels)

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

def generate_combinations(config, single_run=False):
    """Generate only valid feature combinations that include the required fields.

    This function separates the features into required and optional.
    For each desired combination size s (from config['combination_sizes']),
    it selects only combinations of optional features of size (s - number_of_required_features)
    and unions them with the required features.
    """
    if single_run:
        return {
            'metadata': {'all_metadata_all_global': config.get('metadata_fields', [])},
            'global_features': {'all_metadata_all_global': config.get('global_features', {})}
        }
    
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
    if not required_set:
        raise ValueError("No required features found. Check that your 'required_metadata_fields' "
                         "or 'required_global_features' exist in the provided features.")
    
    # The optional features are those not required.
    optional_features = [f for f in all_features if f not in required_set]
    
    # Get the list of combination sizes from configuration.
    combo_sizes = config.get('combination_sizes', [len(all_features)])
    
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
    
    pbar = tqdm(total=total_combinations, desc="Generating feature combinations")
    
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
            pbar.update(1)
            
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
            
    pbar.close()
    
    if not result_combinations['metadata']:
        raise ValueError("No valid feature combinations were generated. Check your configuration.")
    
    print(f"Generated {len(result_combinations['metadata'])} feature combinations for sizes {combo_sizes}")
    print(f"Required metadata: {required_metadata}")
    print(f"Required global features: {required_global_stats}")
    
    return result_combinations

def initialize_tables(train_policy, sample_rates, config, predictor_type='regression', single_run=False):
    """Initialize result tables structure based on predictor type"""
    tables = {}
    for train_set in train_policy:
        tables[train_set] = {
            'impute': {},
            'drop': {}
        }
        for rate in sample_rates:
            if predictor_type == 'regression':
                metrics = {'rmse': create_score_table(config, single_run=single_run),
                          'mae': create_score_table(config, single_run=single_run),
                          'r2': create_score_table(config, single_run=single_run),
                          'mse': create_score_table(config, single_run=single_run)}
            else:  # classification
                metrics = {'accuracy': create_score_table(config, single_run=single_run),
                          'f1': create_score_table(config, single_run=single_run),
                          'precision': create_score_table(config, single_run=single_run),
                          'recall': create_score_table(config, single_run=single_run)}
                
            tables[train_set]['impute'][rate] = metrics.copy()
            tables[train_set]['drop'][rate] = metrics.copy()
    return tables

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
    result = predictor.run_single_experiment(**params)
    
    # Add the feature_key back to the result
    result['feature_key'] = feature_key
    
    return result

def generate_experiment_params(config, predictor_type, single_run):
    """Generator function for experiment parameters"""
    combinations = generate_combinations(config, single_run)
    
    # Debug print
    print(f"\nConfig parameters:")
    print(f"nan_policy: {config['nan_policy']}")
    print(f"train_policy: {config['train_policy']}")
    print(f"sample_rates: {config['sample_rates']}")
    print(f"segment_policy: {config['segment_policy']}")
    print(f"Number of feature combinations: {len(combinations['metadata'])}")
    
    # Generate experiments for each combination of parameters
    for params in product(
        config['nan_policy'], 
        config['train_policy'], 
        config['sample_rates'], 
        config['segment_policy']
    ):
        handle_nan, train_set, sample_rate, segment_size = params
        
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

def run_experiments(predictor_type='regression', single_run=False):
    """Run all experiments with feature caching"""
    # Load configuration
    with open('experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize predictor to load data
    predictor_class = RegressionPredictor if predictor_type == 'regression' else ClassificationPredictor
    predictor = predictor_class()
    
    # Check if we should run nested CV
    if single_run == 'nested_cv':
        # Generate experiment parameters for nested CV
        experiment_params_list = list(generate_experiment_params(config, predictor_type, True))
        
        # Create results directory
        results_dir = f'results/nested_cv_{predictor_type}_{config["model"]["type"]}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Run each nested CV experiment sequentially
        all_results = []
        for experiment_params in experiment_params_list:
            result = predictor.run_nested_cv_experiment(
                handle_nan=experiment_params['handle_nan'],
                train_set=experiment_params['train_set'],
                sample_rate=experiment_params['sample_rate'],
                segment_size=experiment_params['segment_size'],
                metadata_features=experiment_params['metadata_features'],
                global_features=experiment_params['global_features']
            )
            all_results.append(result)
            
            # Save intermediate results after each experiment
            metrics_df = pd.DataFrame(all_results)
            metrics_df.to_csv(f'{results_dir}/nested_cv_results.csv', index=False)
            
            logging.info(f"Completed nested CV experiment {len(all_results)}/{len(experiment_params_list)}")
            
            predictor.clear_feature_cache()
        
        logging.info("All nested CV experiments completed")
        return
    
    # Regular experiment flow for single_run=False or True
    tables = initialize_tables(config['train_policy'], config['sample_rates'], 
                             config, predictor_type, single_run)
    
    # Create experiment parameter generator
    experiment_params = list(generate_experiment_params(config, predictor_type, single_run))
    total_experiments = len(experiment_params)
    
    # Run experiments in parallel
    pbar = tqdm(total=total_experiments, desc=f"Running {predictor_type} experiments")
    
    try:
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            for result in pool.imap_unordered(run_experiment_wrapper, experiment_params):
                pbar.update(1)
                pbar.set_description(
                    f"NaN: {result['handle_nan']}, Train: {result['train_set']}, "
                    f"Rate: {result['sample_rate'] if result['sample_rate'] else 'original'}"
                )
                
                # Process result using single feature key
                current_tables = tables[result['train_set']][result['handle_nan']][result['sample_rate']]
                
                if predictor_type == 'regression':
                    metrics = ['rmse', 'mae', 'r2', 'mse']
                else:
                    metrics = ['accuracy', 'f1', 'precision', 'recall']
                    
                for metric in metrics:
                    current_tables[metric].loc[result['feature_key'], result['segment_size']] = result[metric]
    
    except Exception as e:
        logging.error(f"Error in parallel processing: {str(e)}")
        raise
    
    finally:
        pbar.close()
        # Clear feature cache after experiments
        predictor.clear_feature_cache()
    
    # Save results
    save_results(tables, config['train_policy'], config['nan_policy'], 
                config['sample_rates'], predictor_type)

def save_results(tables, train_policy, nan_policy, sample_rates, predictor_type):
    """Save results to CSV files"""
    # Load config to get model type
    with open('experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_type = config['model']['type']
    
    results_dir = f'results/results_{predictor_type}_{model_type}'
    os.makedirs(results_dir, exist_ok=True)
    
    for train_set in train_policy:
        for handle_nan in nan_policy:
            for sample_rate in sample_rates:
                rate_suffix = f"_{sample_rate}" if sample_rate else ""
                suffix = f"_train{train_set}{rate_suffix}"
                current_tables = tables[train_set][handle_nan][sample_rate]
                
                metrics = ['rmse', 'mae', 'r2', 'mse'] if predictor_type == 'regression' \
                         else ['accuracy', 'f1', 'precision', 'recall']
                
                for metric in metrics:
                    filename = f'{results_dir}/{metric}_table_{handle_nan}{suffix}.csv'
                    try:
                        current_tables[metric].to_csv(filename)
                        logging.info(f"Saved {metric} results to: {filename}")
                    except Exception as e:
                        logging.error(f"Error saving {metric} results: {str(e)}")

if __name__ == "__main__":
    #run_experiments('regression', single_run=False)
    #run_experiments('classification', single_run=False)
    start_time = time.time()
    run_experiments('classification', single_run='nested_cv')
    end_time = time.time()
    print(f"Total Run Time: {end_time - start_time} seconds")