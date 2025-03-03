import numpy as np
import pandas as pd
import seaborn as sns
import os, logging
import matplotlib.pyplot as plt
from .base import ScorePredictionBase
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from utils.supervised_pca import SupervisedPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           balanced_accuracy_score, confusion_matrix, matthews_corrcoef,
                           make_scorer)

class ClassificationPredictor(ScorePredictionBase):
    def __init__(self):
        super().__init__()
        self.cls_score_threshold = self.parameter_config['cls_score_threshold']
        self.predictor_type = 'classification'
        
    def score_to_class(self, score):
        """Convert numerical score to class category"""
        try:
            score = int(score)
            labels = [0, 1]
            
            if score <= self.cls_score_threshold:
                return labels[0]
            elif score > self.cls_score_threshold:
                return labels[1]
            else:
                raise ValueError(f"Invalid score value: {score}")
            
        except (ValueError, TypeError):
            logging.error(f"Invalid score value: {score}")
            return labels[0]
        
    # Create the appropriate scoring function once
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
        
    def initialize_classifier_metrics(self, empty=True, handle_nan=None, train_set=None, sample_rate=None, segment_size=None):
        if empty:
            cv_metrics = {
                'accuracy': [],
                'balanced_accuracy': [],
                'f1': [],
                'precision': [],
                'recall': [],
                'feature_selection_measure': [],
                'sensitivity': [],
                'specificity': [],
                'confusion_matrices': [],
                'selected_features': [],
                'best_params': []
            }
        else:
            cv_metrics = {
                'accuracy': 0.0,
                'accuracy_std': 0.0,
                'balanced_accuracy': 0.0,
                'balanced_accuracy_std': 0.0,
                'f1': 0.0,
                'f1_std': 0.0,
                'precision': 0.0,
                'precision_std': 0.0,
                'recall': 0.0,
                'recall_std': 0.0,
                'sensitivity': 0.0,
                'sensitivity_std': 0.0,
                'specificity': 0.0,
                'specificity_std': 0.0,
                'feature_selection_measure': 0.0,
                'feature_selection_measure_std': 0.0,
                'handle_nan': handle_nan,
                'train_set': train_set,
                'sample_rate': sample_rate,
                'segment_size': segment_size,
                'feature_selection': [],
                'hyperparameter_tuning': {}
            }

        return cv_metrics

    def run_feature_selection_experiment(self, handle_nan, train_set, sample_rate, segment_size, 
                                metadata_features, global_features, **kwargs):
        """
        Run experiment with nested stratified 5-fold cross validation by participant,
        with the outer loop parallelized. In each outer fold, the entire outer test set
        (i.e. multiple participants) is evaluated together.
        When cross_task=True, training is performed on VR data (Set A) and testing on IRL data (Set B).
        When cross_task=False, both training and testing are performed on the same dataset (here, Set A).
        """
        
        self.train_set = train_set
        cross_task = self.parameter_config['cross_task']
        model_type = self.model_config['model']['type']
        param_grid = self.model_config['model'].get('param_grids', {}).get(model_type, {})
        feature_names = self.get_feature_names(metadata_features, global_features)
        feature_selection_measure = self.parameter_config['feature_selection_measure']
        scoring_function = self.get_scoring_function(feature_selection_measure)
        num_jobs = -1 if self.parameter_config['run_mode'] == 'automatic' else 1

        if (len(feature_names) == 0):
            return self.initialize_classifier_metrics(empty=False, handle_nan=handle_nan, train_set=train_set, sample_rate=sample_rate, segment_size=segment_size)
        else:
            cv_metrics = self.initialize_classifier_metrics()

        train_data_dict, test_data_dict, train_score_col, test_score_col = self.get_train_test_data(train_set)

        # For the outer loop, the splitting is done based on the test set.
        test_pids = list(test_data_dict.keys())
        participant_labels = []
        for pid in test_pids:
            # Assume the label for a participant is obtained from the first row of that participant's data.
            df = test_data_dict[pid][0]['data']
            label = self.score_to_class(df[test_score_col].iloc[0])
            participant_labels.append(label)        
        
        def process_outer_fold(outer_test_pids):
            """
            Process one outer fold where outer_test_pids is the set of participant IDs held out as test.
            The test data from all these participants is aggregated together.
            """

            train_features = []
            train_scores = []
            train_pids = []
            for pid, data in train_data_dict.items():
                if not cross_task:
                    if pid in outer_test_pids:
                        continue  # Exclude any participant in the outer test set.
                df = data[0]['data']
                segment_features = self.segment_and_extract_features(
                    df, pid=pid,
                    segment_size=segment_size,
                    enabled_metadata=metadata_features,
                    global_features=global_features,
                    train_set=train_set,
                    sample_rate=sample_rate
                )
                score = self.score_to_class(df[train_score_col].iloc[0])
                for features in segment_features:
                    train_features.append(features)
                    train_scores.append(score)
                    train_pids.append(pid)
            # Convert training data to numpy arrays.
            X_train_full = np.array(train_features)
            y_train_full = np.array(train_scores)
            train_pids = np.array(train_pids)

            # Prepare test data: aggregate data from all participants in the outer test set.
            test_features = []
            test_scores = []
            for pid in outer_test_pids:
                test_df = test_data_dict[pid][0]['data']
                test_segments = self.segment_and_extract_features(
                    test_df, pid=pid,
                    segment_size=segment_size,
                    enabled_metadata=metadata_features,
                    global_features=global_features,
                    train_set=train_set,
                    sample_rate=sample_rate
                )
                label = self.score_to_class(test_df[test_score_col].iloc[0])
                test_features.extend(test_segments)
                test_scores.extend([label] * len(test_segments))
            X_test = np.array(test_features)
            y_test = np.array(test_scores)

            # Preprocess data.
            scaler = RobustScaler()
            imputer = SimpleImputer(strategy='mean')

            X_train_scaled = scaler.fit_transform(X_train_full)
            X_test_scaled = scaler.transform(X_test)

            if handle_nan != 'impute': raise ValueError(f"Invalid handle_nan value: {handle_nan}")
            X_train_scaled = imputer.fit_transform(X_train_scaled)
            X_test_scaled = imputer.transform(X_test_scaled)

            # --- Inner Loop: Feature Selection & Hyperparameter Tuning ---
            if self.parameter_config['run_mode'] == 'automatic':
                inner_cv = StratifiedKFold(n_splits=self.parameter_config['inner_cv_k'], shuffle=True, random_state=self.parameter_config['random_state'])
            elif self.parameter_config['run_mode'] == 'manual':
                inner_cv = RepeatedStratifiedKFold(n_splits=self.parameter_config['inner_cv_k'], n_repeats=self.parameter_config['inner_cv_repeats'], random_state=self.parameter_config['random_state'])
            
            if self.parameter_config['run_mode'] == 'automatic':
                k_features_to_select = (10, 15)
                sfs = SequentialFeatureSelector(
                    self.get_model(self.model_config['model']),
                    k_features=k_features_to_select,
                    forward=True,
                    scoring=make_scorer(scoring_function),
                    floating=True,
                    cv=inner_cv,
                    n_jobs=num_jobs,
                    verbose=0
                )
                sfs.fit(X_train_scaled, y_train_full)
                selected_feature_indices = sfs.k_feature_idx_
                selected_features = [feature_names[i] for i in selected_feature_indices]

                # Apply feature selection.
                X_train_selected = X_train_scaled[:, selected_feature_indices]
                X_test_selected = X_test_scaled[:, selected_feature_indices]

            else:
                X_train_selected = X_train_scaled
                X_test_selected = X_test_scaled
                selected_features = feature_names

            # Hyperparameter tuning.
            if param_grid:
                grid_search = GridSearchCV(
                    estimator=self.get_model(self.model_config['model']),
                    param_grid=param_grid,
                    scoring=make_scorer(scoring_function),
                    cv=inner_cv,
                    n_jobs=num_jobs
                )
                grid_search.fit(X_train_selected, y_train_full)
                best_params = grid_search.best_params_
                final_model = grid_search.best_estimator_
            else:
                final_model = self.get_model(self.model_config['model'])
                best_params = {}

            # Optionally wrap with a resampling pipeline if needed.
            if self.model_config['model'].get('use_resampling', False):
                resampling_strategy = self.model_config['model'].get('resampling_strategy', 'none')
                if resampling_strategy == 'smote_tomek':
                    self._resampler = SMOTETomek(random_state=self.parameter_config['random_state'])
                    pipeline = ImbPipeline([
                        ('resampler', self._resampler),
                        ('classifier', final_model)
                    ])
                else:
                    pipeline = Pipeline([('classifier', final_model)])
            else:
                pipeline = Pipeline([('classifier', final_model)])

            # Train the final model on the full VR training data and evaluate on the aggregated IRL test data.
            pipeline.fit(X_train_selected, y_train_full)
            test_predictions = pipeline.predict(X_test_selected)

            # Calculate metrics.
            fold_accuracy = accuracy_score(y_test, test_predictions)
            fold_balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)
            fold_f1 = f1_score(y_test, test_predictions, average='weighted', zero_division=0)
            fold_precision = precision_score(y_test, test_predictions, average='weighted', zero_division=0)
            fold_recall = recall_score(y_test, test_predictions, average='weighted', zero_division=0)
            fold_sensitivity = recall_score(y_test, test_predictions, average='binary', pos_label=1, zero_division=0)
            fold_specificity = recall_score(y_test, test_predictions, average='binary', pos_label=0, zero_division=0)
            fold_feature_selection_measure = scoring_function(y_test, test_predictions)
            fold_cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])

            # Package the results for this outer fold.
            fold_results = {
                'accuracy': fold_accuracy,
                'balanced_accuracy': fold_balanced_accuracy,
                'f1': fold_f1,
                'precision': fold_precision,
                'recall': fold_recall,
                'sensitivity': fold_sensitivity,
                'specificity': fold_specificity,
                'feature_selection_measure': fold_feature_selection_measure,
                'confusion_matrix': fold_cm,
                'selected_features': selected_features,
                'best_params': best_params,
                'y_true': y_test,
                'y_pred': test_predictions
            }
            return fold_results

        # --- End of process_outer_fold definition ---

        # Use StratifiedKFold on the participant level for the outer loop.
        outer_results = []

        if cross_task:
            fold_results = process_outer_fold(test_pids)
            outer_results.append(fold_results)
        else:
            outer_k = self.parameter_config['outer_cv_k']
            outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=self.parameter_config['random_state'])

            for train_idx, test_idx in outer_cv.split(test_pids, participant_labels):
                # The outer test set for this fold:
                outer_test_pids = [test_pids[i] for i in test_idx]
                fold_results = process_outer_fold(set(outer_test_pids))
                outer_results.append(fold_results)

        # Aggregate metrics from all outer folds.
        for result in [res for res in outer_results if res is not None]:
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

        for features in cv_metrics['selected_features']:
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        for params in cv_metrics['best_params']:
            for param, value in params.items():
                if param not in param_counts:
                    param_counts[param] = {}
                param_counts[param][value] = param_counts[param].get(value, 0) + 1

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
            'handle_nan': handle_nan,
            'train_set': train_set,
            'sample_rate': sample_rate,
            'segment_size': segment_size,
            'feature_selection': sorted(
                [(feature, count) for feature, count in 
                {f: cv_metrics['selected_features'].count(f) for f in feature_names}.items()],
                key=lambda x: x[1], reverse=True),
            'hyperparameter_tuning': cv_metrics['best_params']
        }

        # Plot both average and per-fold confusion matrices
        tag = f'{self.parameter_config["dataset"]}_{self.parameter_config["train_policy"]}_{self.model_config["model"]["type"]}'
        tag += f'_{self.model_config["model"]["use_resampling"]}_{self.feature_config["combination_sizes"][0]}'
        output_dir = f'figures/confusion_matrices/{tag}'
        os.makedirs(output_dir, exist_ok=True)

        # First create the average confusion matrix plot
        plt.figure(figsize=(10, 8))

        # Calculate average of normalized confusion matrices
        normalized_matrices = []
        for cm in cv_metrics['confusion_matrices']:
            # Normalize each confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            normalized_matrices.append(cm_norm)
        
        # Calculate the average normalized confusion matrix
        avg_cm = np.mean(normalized_matrices, axis=0)
        
        # Calculate standard deviation for error reporting
        std_cm = np.std(normalized_matrices, axis=0)

        labels = [0, 1]

        annotations = np.empty(avg_cm.shape, dtype='<U20')
        for i in range(avg_cm.shape[0]):
            for j in range(avg_cm.shape[1]):
                percentage = avg_cm[i, j]
                std = std_cm[i, j]
                annotations[i, j] = f'{percentage:.2%} (±{std:.2%})'

        sns.heatmap(avg_cm, 
                   annot=annotations,
                   fmt='s',
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   vmin=0, vmax=1)

        feature_tag = ['_'.join([i[0].lower() for i in j.split('_')[:-1]] + [j.split('_')[-1]]) for j in feature_names]
        feature_tag = ' + '.join(feature_tag)

        plt.title(f'Average Confusion Matrix\n{feature_selection_measure}: {avg_metrics["feature_selection_measure"]:.3f} (±{avg_metrics["feature_selection_measure_std"]:.3f})')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.savefig(f'{output_dir}/{feature_tag}.png')
        plt.close()

        return avg_metrics

        # # Now create a new figure for per-fold confusion matrices
        # n_folds = len(cv_metrics['confusion_matrices'])
        # n_cols = 5  # Number of columns in the subplot grid
        # n_rows = (n_folds + n_cols - 1) // n_cols  # Calculate required number of rows

        # plt.figure(figsize=(4*n_cols, 4*n_rows))
        # plt.suptitle(f'Per-Fold Confusion Matrices - {train_set} Set', fontsize=16, y=1.02)

        # for i, cm in enumerate(cv_metrics['confusion_matrices']):
        #     plt.subplot(n_rows, n_cols, i+1)
            
        #     # Normalize the confusion matrix
        #     cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        #     # Create annotations with counts and percentages
        #     annotations = np.empty_like(cm_norm, dtype='<U20')
        #     for row in range(cm_norm.shape[0]):
        #         for col in range(cm_norm.shape[1]):
        #             count = cm[row, col]
        #             percentage = cm_norm[row, col]
        #             annotations[row, col] = f'{count} ({percentage:.2%})'
            
        #     # Create heatmap
        #     sns.heatmap(cm_norm,
        #                 annot=annotations,
        #                 fmt='s',
        #                 cmap='Blues',
        #                 xticklabels=labels,
        #                 yticklabels=labels,
        #                 cbar=False)  # Remove colorbar for cleaner subplots
            
        #     plt.title(f'Fold {i+1}')
        #     plt.xlabel('Predicted')
        #     plt.ylabel('Actual')

        # plt.tight_layout()
        # plt.savefig(f'{output_dir}/per_fold_cnf_{train_set}_{model_type}_{self.cls_score_threshold}.png', 
        #             bbox_inches='tight', 
        #             dpi=300)
        # plt.close()