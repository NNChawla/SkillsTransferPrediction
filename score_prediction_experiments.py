import os
import yaml
import logging
import numpy as np
import pandas as pd
from itertools import product, combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from score_prediction import RegressionPredictor, ClassificationPredictor

def create_score_table(config, segment_sizes=['full'], single_run=False):
    """Create empty DataFrame for storing results using config values"""
    if single_run:
        # For single run, only create one row with all features
        row_label = "all_metadata_all_global"
        return pd.DataFrame(columns=segment_sizes, index=[row_label])
    
    # Create list of all possible features
    all_features = []
    
    # Add metadata features
    metadata_fields = config.get('metadata_fields', [])
    if metadata_fields:
        all_features.extend(metadata_fields)
    
    # Add global features with their statistics
    global_features = config.get('global_features', {})
    if global_features:
        for feature_name, feature_config in global_features.items():
            statistics = feature_config.get('statistics', [])
            for stat in statistics:
                all_features.append(f"{feature_name}_{stat}")
    
    # Get required features
    required_metadata = set(config.get('required_metadata_fields', []))
    required_global = set(config.get('required_global_features', []))
    
    # Get combination sizes
    combo_sizes = config.get('combination_sizes', [len(all_features)])
    
    # Generate row labels for each combination size
    row_labels = []
    for size in combo_sizes:
        for combo in combinations(all_features, size):
            # Check if combination includes all required features
            if required_metadata and not required_metadata.issubset(set(combo)):
                continue
            if required_global and not required_global.issubset(set(combo)):
                continue
            
            # Generate row label - features sorted and joined by underscore
            row_labels.append('_'.join(sorted(combo)))
    
    if not row_labels:
        raise ValueError("No features found for score table. Check your configuration.")
        
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
    """Generate all possible combinations for experiments"""
    if single_run:
        return {
            'metadata': {'all': config.get('metadata_fields', [])},
            'global_features': {'all_global': config.get('global_features', {})}
        }
    
    # Create list of all possible features
    all_features = []
    
    # Add metadata features
    metadata_fields = config.get('metadata_fields', [])
    if metadata_fields:
        for field in metadata_fields:
            all_features.append(('metadata', field))
    
    # Add global features with their statistics
    global_features = config.get('global_features', {})
    if global_features:
        for feature_name, feature_config in global_features.items():
            statistics = feature_config.get('statistics', [])
            for stat in statistics:
                all_features.append(('global', (feature_name, stat)))
    
    # Get required features
    required_metadata = set(config.get('required_metadata_fields', []))
    required_global_stats = set()
    
    # Parse required global features to include statistics
    for feat in config.get('required_global_features', []):
        if '_' in feat:
            feature_name, stat = feat.rsplit('_', 1)
            required_global_stats.add((feature_name, stat))
    
    # Get combination sizes
    combo_sizes = config.get('combination_sizes', [len(all_features)])
    
    # Initialize result structure
    result_combinations = {
        'metadata': {},
        'global_features': {}
    }
    
    # Generate combinations for each size
    for size in combo_sizes:
        for combo in combinations(all_features, size):
            # Check if this combination includes all required features
            metadata = set()
            global_stats = set()
            
            for feature_type, feature in combo:
                if feature_type == 'metadata':
                    metadata.add(feature)
                else:  # global feature
                    global_stats.add(feature)
            
            # Skip if combination doesn't include all required features
            if not (required_metadata.issubset(metadata) and 
                   all(stat in global_stats for stat in required_global_stats)):
                continue
            
            # Process valid combination
            metadata_list = []
            global_feats = {}
            feature_names = []
            
            for feature_type, feature in combo:
                if feature_type == 'metadata':
                    metadata_list.append(feature)
                    feature_names.append(feature)
                else:  # global feature
                    feature_name, stat = feature
                    if feature_name not in global_feats:
                        global_feats[feature_name] = {'features': [feature_name], 'statistics': []}
                    global_feats[feature_name]['statistics'].append(stat)
                    feature_names.append(f"{feature_name}_{stat}")
            
            # Generate simple key name - just the features joined by underscore
            key = '_'.join(sorted(feature_names))
            
            result_combinations['metadata'][key] = metadata_list
            result_combinations['global_features'][key] = global_feats
    
    # Add debug logging
    num_combinations = len(result_combinations['metadata'])
    print(f"Generated {num_combinations} feature combinations for sizes {combo_sizes}")
    print(f"Required metadata: {required_metadata}")
    print(f"Required global features: {required_global_stats}")
    
    if not result_combinations['metadata']:
        raise ValueError("No valid feature combinations were generated. Check your configuration.")
    
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
    
    # Initialize result tables
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
    run_experiments('regression', single_run=False)
    #run_experiments('classification')