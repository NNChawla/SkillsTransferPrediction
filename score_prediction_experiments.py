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
        row_label = "all_trackers_all_measurements_all_all_global"
        return pd.DataFrame(columns=segment_sizes, index=[row_label])
    
    # Generate global feature combinations
    global_feature_combinations = ['none']  # Always include 'none'
    if 'global_features' in config:
        global_features = list(config['global_features'].keys())
        for r in range(1, len(global_features) + 1):
            for combo in combinations(global_features, r):
                key_name = '_'.join(combo)
                if len(combo) == len(global_features):
                    key_name = 'all_global'
                global_feature_combinations.append(key_name)

    # Generate tracker combinations (e.g., H+L+R, H+L, etc.)
    tracker_combinations = []
    trackers = config['trackers']
    for r in range(1, len(trackers) + 1):
        for combo in combinations(trackers, r):
            # Create key name from first letter of each tracker
            key_name = '+'.join(t[0] for t in combo)
            tracker_combinations.append(key_name)
    
    # Generate measurement combinations (e.g., pos_euler_quat_sixD, pos_only, etc.)
    measurement_types = []
    measurements = list(config['measurements'].keys())
    for r in range(1, len(measurements) + 1):
        for combo in combinations(measurements, r):
            # Create measurement type name
            key_name = '_'.join(combo) + '_only' if len(combo) == 1 else '_'.join(combo)
            measurement_types.append(key_name)
    
    # Generate metadata combinations (e.g., none, age, gender, all, etc.)
    metadata_types = ['none']  # Always include 'none'
    metadata_fields = config.get('metadata_fields', [])  # Default to empty list if not present
    if metadata_fields:  # Only generate combinations if metadata fields exist
        for r in range(1, len(metadata_fields) + 1):
            for combo in combinations(metadata_fields, r):
                key_name = '_'.join(field.lower() for field in combo)
                if len(combo) == len(metadata_fields):
                    key_name = 'all'
                metadata_types.append(key_name)
    
    # Create row labels that combine all combinations
    row_labels = [f"{t}_{m}_{meta}_{g}" for t in tracker_combinations 
                 for m in measurement_types 
                 for meta in metadata_types
                 for g in global_feature_combinations]
    
    return pd.DataFrame(columns=segment_sizes, index=row_labels)

def generate_combinations(config, single_run=False):
    """Generate all possible combinations for experiments"""
    def generate_global_feature_combinations(global_features, single_run):
        if not global_features:
            return {'none': {}}
            
        if single_run:
            return {'all_global': global_features}
            
        global_feature_sets = {'none': {}}
        feature_types = list(global_features.keys())
        
        for r in range(1, len(feature_types) + 1):
            for combo in combinations(feature_types, r):
                key_name = '_'.join(combo)
                if len(combo) == len(feature_types):
                    key_name = 'all_global'
                feature_dict = {feat_type: global_features[feat_type] for feat_type in combo}
                global_feature_sets[key_name] = feature_dict
                
        return global_feature_sets

    def generate_measurement_combinations(measurements, single_run):
        if single_run:
            return {'all_measurements': measurements}
            
        measurement_sets = {}
        measure_types = list(measurements.keys())
        
        for r in range(1, len(measure_types) + 1):
            for combo in combinations(measure_types, r):
                key_name = '_'.join(combo) + '_only' if len(combo) == 1 else '_'.join(combo)
                measurement_dict = {measure_type: measurements[measure_type] for measure_type in combo}
                measurement_sets[key_name] = measurement_dict
        
        return measurement_sets

    def generate_tracker_combinations(trackers, single_run):
        if single_run:
            return {'all_trackers': trackers}
            
        tracker_combinations = {}
        for r in range(1, len(trackers) + 1):
            for combo in combinations(trackers, r):
                key_name = '+'.join(t[0] for t in combo)
                tracker_combinations[key_name] = list(combo)
        return tracker_combinations

    def generate_metadata_combinations(metadata_fields, single_run):
        if not metadata_fields:
            return {'none': []}
            
        if single_run:
            return {'all': metadata_fields}
            
        metadata_sets = {'none': []}
        if metadata_fields:
            for r in range(1, len(metadata_fields) + 1):
                for combo in combinations(metadata_fields, r):
                    key_name = '_'.join(field.lower() for field in combo)
                    if len(combo) == len(metadata_fields):
                        key_name = 'all'
                    metadata_sets[key_name] = list(combo)
        return metadata_sets

    return {
        'measurements': generate_measurement_combinations(config['measurements'], single_run),
        'trackers': generate_tracker_combinations(config['trackers'], single_run),
        'metadata': generate_metadata_combinations(config.get('metadata_fields', []), single_run),
        'global_features': generate_global_feature_combinations(config.get('global_features', {}), single_run)
    }

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
    return predictor.run_single_experiment(**params)

def generate_experiment_params(config, predictor_type, single_run):
    """Generator function for experiment parameters"""
    combinations = generate_combinations(config, single_run)
    
    for params in product(
        config['nan_policy'], 
        config['train_policy'], 
        config['sample_rates'], 
        config['segment_policy'], 
        combinations['trackers'].items(),
        combinations['measurements'].items(), 
        combinations['metadata'].items(),
        combinations['global_features'].items()
    ):
        handle_nan, train_set, sample_rate, segment_size, \
        (tracker_key, trackers), (measure_key, measurements), \
        (meta_key, metadata_features), (global_key, global_features) = params
        
        yield {
            'predictor_type': predictor_type,
            'handle_nan': handle_nan,
            'train_set': train_set,
            'sample_rate': sample_rate,
            'segment_size': segment_size,
            'tracker_key': tracker_key,
            'trackers': trackers,
            'measure_key': measure_key,
            'measurements': measurements,
            'meta_key': meta_key,
            'metadata_features': metadata_features,
            'global_key': global_key,
            'global_features': global_features
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
                    f"Rate: {result['sample_rate'] if result['sample_rate'] else 'original'}, "
                    f"Tracker: {result['tracker_key']}"
                )
                
                # Process result
                row_key = f"{result['tracker_key']}_{result['measure_key']}_" \
                         f"{result['meta_key']}_{result['global_key']}"
                current_tables = tables[result['train_set']][result['handle_nan']][result['sample_rate']]
                
                if predictor_type == 'regression':
                    metrics = ['rmse', 'mae', 'r2', 'mse']
                else:
                    metrics = ['accuracy', 'f1', 'precision', 'recall']
                    
                for metric in metrics:
                    current_tables[metric].loc[row_key, result['segment_size']] = result[metric]
    
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
    run_experiments('regression', single_run=True)
    #run_experiments('classification')