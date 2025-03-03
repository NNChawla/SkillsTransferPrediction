from scripts.combination_interface import run_experiments
from scripts.base import load_config, write_config
import pickle, tqdm

# datasets = ['v3', 'BodyRelative_Motion_PQ']
# model_types = ['lightgbm', 'knn', 'random_forest']
# train_sets = ['A', 'B']
# cross_tasks = [True, False]
# inner_cv_ks = [4, 5]
# outer_cv_ks = [5, 10]
# cls_score_thresholds = [0, 4, 6, 8]
# use_resamplings = [True, False]

datasets = ['v3']
model_types = ['lightgbm']
train_sets = ['A', 'B']
cross_tasks = [True]
inner_cv_ks = [5]
outer_cv_ks = [None]
cls_score_thresholds = [0]
use_resamplings = [False]

parameter_config = load_config('configs/parameter_config.yaml')
feature_config = load_config('configs/feature_config.yaml')
model_config = load_config('configs/model_config.yaml')

results = {}

def wrapper(params, progress_bar):
    parameter_config['dataset'] = params['dataset']
    model_config['model']['type'] = params['model_type']
    parameter_config['cross_task'] = params['cross_task']
    parameter_config['outer_cv_k'] = params['outer_cv_k']
    parameter_config['inner_cv_k'] = params['inner_cv_k']
    model_config['model']['use_resampling'] = params['use_resampling']
    parameter_config['cls_score_threshold'] = params['cls_score_threshold']
    parameter_config['train_policy'] = params['train_policy']

    write_config('./parameter_config.yaml', parameter_config)
    write_config('./feature_config.yaml', feature_config)
    write_config('./model_config.yaml', model_config)

    key = f"{params['dataset']}_{params['model_type']}_{params['cross_task']}_{params['outer_cv_k']}_{params['inner_cv_k']}_{params['use_resampling']}_{params['cls_score_threshold']}_{params['train_policy']}"
    progress_bar.set_description(f"Running {key}")
    results[key] = run_experiments()
    progress_bar.update(1)

if __name__ == "__main__":

    parameter_combinations = [
        {
            'dataset': dataset,
            'model_type': model,
            'cross_task': task, 
            'outer_cv_k': outer_cv_k,
            'inner_cv_k': inner_cv_k,
            'use_resampling': use_resampling,
            'cls_score_threshold': cls_score_threshold,
            'train_policy': train_set
        }
        for dataset in datasets
        for model in model_types 
        for task in cross_tasks
        for outer_cv_k in outer_cv_ks
        for inner_cv_k in inner_cv_ks
        for use_resampling in use_resamplings
        for cls_score_threshold in cls_score_thresholds
        for train_set in train_sets
    ]

    pbar = tqdm.tqdm(total=len(parameter_combinations), desc="Running parameterized experiments:")

    _ = [wrapper(params, pbar) for params in parameter_combinations]

    pbar.close()

    if parameter_config['run_mode'] == 'manual':
        with open('./results/manual_experiments_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    elif parameter_config['run_mode'] == 'automatic':
        with open('./results/automatic_experiments_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Invalid run mode: {parameter_config['run_mode']}")