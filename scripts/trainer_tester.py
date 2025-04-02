import numpy as np
import pandas as pd
import seaborn as sns
import os, logging
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from .processor import Processor

class TrainerTester(Processor):
    def __init__(self):
        super().__init__()

    def get_feature_names(self, metadata_features, global_features):
        """Generate feature names in the same order as they're created"""
        feature_names = []
        
        # Add metadata feature names
        if metadata_features:
            feature_names.extend([f for f in metadata_features 
                                if not ((f == 'A_Build_Time' and self.train_set != 'A') or 
                                      (f == 'B_Build_Time' and self.train_set != 'B'))])
        
        # Add global feature names
        if global_features:
            for feature_type, feature_config in global_features.items():
                cols = feature_config['features']
                statistics = feature_config.get('statistics', ['min', 'max', 'median', 'mean', 'std'])
                
                for col in cols:
                    feature_names.extend([f"{col}_{stat}" for stat in statistics])
        
        return feature_names
    
    def train_model(self, model, train_set, metadata_features, global_features, cross_task=True, pids_to_exclude=[], step=-1):
        self.train_set = train_set
        
        train_data_dict, _, train_score_col, _ = self.get_train_test_data(train_set, cross_task)

        train_features = []
        train_scores = []
        train_pids = []
        for pid, data in train_data_dict.items():
            if not cross_task:
                if pid in pids_to_exclude:
                    continue

            assembly_df = data[0]['assembly_data']
            tracking_df = data[0]['tracking_data']
            step_features = self.extract_features(
                assembly_df, tracking_df,
                pid=pid, train_set=train_set,
                metadata_features=metadata_features,
                global_features=global_features,
                step = step
            )
            score = self.score_to_class(tracking_df[train_score_col].iloc[0], score_threshold=0)
            
            # Append the features directly since they're already a single feature vector
            train_features.append(step_features)
            train_scores.append(score)
            train_pids.append(pid)

        # Convert training data to numpy arrays.
        X_train_full = np.array(train_features)
        y_train_full = np.array(train_scores)
        train_pids = np.array(train_pids)

        X_train_scaled = self._scaler.fit_transform(X_train_full)
        X_train_scaled = self._imputer.fit_transform(X_train_scaled)

        model.fit(X_train_scaled, y_train_full)
        return model
        
    def test_model(self, model, train_set, metadata_features, global_features, cross_task=True, pids_to_test=[], step=-1, use_proba=False):
        feature_names = self.get_feature_names(metadata_features, global_features)
        if (len(feature_names) == 0):
            return {}

        _, test_data_dict, _, test_score_col = self.get_train_test_data(train_set, cross_task)

        test_pids = list(test_data_dict.keys()) if len(pids_to_test) == 0 else pids_to_test
        test_features = []
        test_scores = []
        for pid in test_pids:
            assembly_df = test_data_dict[pid][0]['assembly_data']
            tracking_df = test_data_dict[pid][0]['tracking_data']
            step_features = self.extract_features(
                assembly_df, tracking_df,
                pid=pid, train_set=train_set,
                metadata_features=metadata_features,
                global_features=global_features,
                step = step
            )
            label = self.score_to_class(tracking_df[test_score_col].iloc[0], score_threshold=0)
            
            # Append features directly since they're already a single feature vector
            test_features.append(step_features)
            test_scores.append(label)
        
        X_test = np.array(test_features)
        y_test = np.array(test_scores)

        X_test_scaled = self._scaler.transform(X_test)
        X_test_scaled = self._imputer.transform(X_test_scaled)

        y_pred = model.predict_proba(X_test_scaled) if use_proba else model.predict(X_test_scaled)

        return y_pred, y_test
    
    def cross_validate_model(self, model, train_set, metadata_features, global_features, cross_task=True):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

        _, test_data_dict, _, test_score_col = self.get_train_test_data(train_set, cross_task)

        # For the outer loop, the splitting is done based on the test set.
        test_pids = list(test_data_dict.keys())
        participant_labels = []
        for pid in test_pids:
            # Assume the label for a participant is obtained from the first row of that participant's data.
            df = test_data_dict[pid][0]['tracking_data']
            label = self.score_to_class(df[test_score_col].iloc[0], score_threshold=0)
            participant_labels.append(label)

        results = []
        for train_idx, test_idx in cv.split(test_pids, participant_labels):
            outer_test_pids = set([test_pids[i] for i in test_idx])
            model = self.train_model(model, train_set, metadata_features, global_features, cross_task, pids_to_exclude=outer_test_pids)
            y_pred, y_test = self.test_model(model, train_set, metadata_features, global_features, cross_task, pids_to_test=outer_test_pids, use_proba=False)
            result = self.evaluate_predictions(y_test, y_pred)
            print(f"\t\tmcc: {result['feature_selection_measure']}")
            results.append(result)

        cv_metrics = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'sensitivity': [],
            'specificity': [],
            'feature_selection_measure': [],
            'confusion_matrices': []
        }

        for result in [res for res in results if res is not None]:
            cv_metrics['accuracy'].append(result['accuracy'])
            cv_metrics['balanced_accuracy'].append(result['balanced_accuracy'])
            cv_metrics['f1'].append(result['f1'])
            cv_metrics['precision'].append(result['precision'])
            cv_metrics['recall'].append(result['recall'])
            cv_metrics['sensitivity'].append(result['sensitivity'])
            cv_metrics['specificity'].append(result['specificity'])
            cv_metrics['feature_selection_measure'].append(result['feature_selection_measure'])
            cv_metrics['confusion_matrices'].append(result['confusion_matrix'])

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
            'confusion_matrices': cv_metrics['confusion_matrices']
        }

        return avg_metrics
    
    def plot_model_results(self, results, train_set, model_type, feature_names):
        tag = f'{model_type}_{train_set}'
        output_dir = f'figures/confusion_matrices/{tag}'
        os.makedirs(output_dir, exist_ok=True)

        # First create the average confusion matrix plot
        plt.figure(figsize=(10, 8))

        # Calculate average of normalized confusion matrices
        normalized_matrices = []
        for cm in results['confusion_matrices']:
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

        feature_tag = sorted(['_'.join([i[0].lower() for i in j.split('_')[:-1]] + [j.split('_')[-1]]) for j in feature_names])[:10]
        feature_tag = ' + '.join(feature_tag)

        plt.title(f'Average Confusion Matrix\nmcc: {results["feature_selection_measure"]:.3f} (±{results["feature_selection_measure_std"]:.3f})')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.savefig(f'{output_dir}/{feature_tag}.png')
        plt.close()