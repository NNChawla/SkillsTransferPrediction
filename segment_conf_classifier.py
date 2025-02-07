import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os
import yaml
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

class SegmentClassifier:
    def __init__(self, config_path='segment_config.yaml'):
        self.config = self.load_config(config_path)
        self._metadata = self.load_metadata()
        self.model = self.get_model(self.config['model'])
        self._scaler = RobustScaler()
        self._imputer = SimpleImputer(strategy='mean')
        # Add resampler
        self._resampler = SMOTETomek(random_state=42)
    
    def load_config(self, config_path):
        """Load experiment configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_metadata(self):
        """Load metadata with scores"""
        metadata_parquet = './data/FAB/metadata.parquet'
        metadata_csv = './data/FAB/metadata.csv'
        
        if os.path.exists(metadata_parquet):
            return pd.read_table(metadata_parquet).to_pandas()
        else:
            return pd.read_csv(metadata_csv, header=0)
    
    def get_model(self, model_config):
        """Create model instance based on configuration"""
        model_type = model_config.get('type', 'random_forest')
        params = model_config.get('model_params', {}).get(model_type, {})
        
        if model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_type == 'svm':
            # If you want probability outputs from SVC, set probability=True in the config
            return SVC(**params)
        elif model_type == 'knn':
            return KNeighborsClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported types are: random_forest, svm, knn")
    
    def score_to_class(self, score):
        """Convert numerical score to class category"""
        try:
            score = int(score)
            labels = ['Non-0', '0']
            return labels[0] if score > 0 else labels[1]
        except (ValueError, TypeError):
            logging.error(f"Invalid score value: {score}")
            return labels[0]
    
    def load_and_preprocess_data(self, segments_dir, set_type='A'):
        """Load and preprocess segmented data"""
        all_segments = []
        score_col = f'Score_{set_type}'
        
        # Get list of segment files
        segment_files = [f for f in os.listdir(segments_dir) if f.endswith('_segments.csv')]
        
        for file in segment_files:
            # Extract FAB ID from filename
            fab_id = file.split('_')[0]
            
            # Get score from metadata
            id_metadata = self._metadata[self._metadata['ID'] == fab_id]
            if len(id_metadata) == 0:
                continue
            
            score = id_metadata[score_col].iloc[0]
            score_class = self.score_to_class(score)
            
            # Read segment file
            file_path = os.path.join(segments_dir, file)
            segments_df = pd.read_csv(file_path)
            
            # Add metadata to segments
            segments_df['FAB_ID'] = fab_id
            segments_df['Score'] = score_class
            
            all_segments.append(segments_df)
        
        # Combine all segments
        if not all_segments:
            raise ValueError(f"No valid segment files found in {segments_dir}")
        
        combined_df = pd.concat(all_segments, ignore_index=True)
        
        # Extract features based on config
        feature_cols = []
        for feature_name, feature_config in self.config['segment_features'].items():
            feature_cols.extend([
                f"{col}_{stat}" 
                for col in feature_config['features']
                for stat in feature_config.get('statistics', ['mean'])
            ])
        
        X = combined_df[feature_cols].values
        y = combined_df['Score'].values
        ids = combined_df['FAB_ID'].values
        
        return X, y, ids, feature_cols
    
    def train_and_evaluate(self, train_dir, test_dir, train_set='A'):
        """Train and evaluate the classifier using confidence-based aggregation"""
        # Load and preprocess data
        X_train, y_train, train_ids, feature_names = self.load_and_preprocess_data(
            train_dir, set_type=train_set)
        X_test, y_test, test_ids, _ = self.load_and_preprocess_data(
            test_dir, set_type='B' if train_set == 'A' else 'A')
        
        # Create a pipeline with scaling, imputation, resampling, and classification
        pipeline = ImbPipeline([
            ('scaler', self._scaler),
            ('imputer', self._imputer),
            ('resampler', self._resampler),
            ('classifier', self.model)
        ])
        
        # Train pipeline
        pipeline.fit(X_train, y_train)
        
        # Scale and impute test data (without resampling)
        X_test_processed = self._imputer.transform(self._scaler.transform(X_test))
        
        # Get the trained classifier from the pipeline
        self.model = pipeline.named_steps['classifier']
        
        # -- CONFIDENCE-BASED PREDICTIONS --
        # Attempt to retrieve class probabilities or decision_function
        # We assume 2 classes: "0" and "Non-0"
        
        class_labels = self.model.classes_  # e.g., array(['0', 'Non-0'], dtype=object) or vice versa
        # Ensure we know the index for each class in the probability output
        # We'll handle two scenarios: predict_proba or decision_function
        
        if hasattr(self.model, "predict_proba") and callable(self.model.predict_proba):
            # If the model supports predict_proba, use that
            probs = self.model.predict_proba(X_test_processed)
            # probs is shape (num_samples, 2), if only 2 classes
        elif hasattr(self.model, "decision_function") and callable(self.model.decision_function):
            # For SVC(probability=False), we can get distances from decision boundary as "confidence"
            distances = self.model.decision_function(X_test_processed)
            # distances is shape (num_samples,) for binary classification
            # We'll create a pseudo "prob-like" measure:  the further from boundary, the more confident
            # But we need a positive confidence for predicted class. 
            # This is a simplified approach: we'll convert distances into something that sums to 1
            # This is just a heuristic, not a strict probability.
            # For a 2-class problem, let's do a logistic transform:
            import scipy.special
            prob_pos = 1 / (1 + np.exp(-distances))
            probs = np.column_stack([1 - prob_pos, prob_pos])
        else:
            # Fallback: just do raw predictions (no confidence). This won't be truly confidence-based
            # but we can at least keep the rest of the pipeline.
            raw_preds = self.model.predict(X_test_processed)
            # We'll give "1.0" to predicted class, "0.0" to the other class
            # This means no real weighting, but we won't crash
            probs = np.zeros((len(raw_preds), 2))
            for i, p in enumerate(raw_preds):
                if p == '0':
                    probs[i, 0] = 1.0
                else:
                    probs[i, 1] = 1.0
        
        # Build a DataFrame for per-segment predictions & confidences
        # We'll store the predicted label (class with highest probability) + the two columns of probability
        pred_indices = np.argmax(probs, axis=1)
        predicted_labels = [class_labels[idx] for idx in pred_indices]
        
        results_df = pd.DataFrame({
            'FAB_ID': test_ids,
            'Actual': y_test,
            'Predicted': predicted_labels,
            'Prob_0': probs[:, 0],         # Probability/confidence for class "0"
            'Prob_Non0': probs[:, 1],      # Probability/confidence for class "Non-0"
        })
        
        # -- AGGREGATE BY CONFIDENCE RATIO --
        aggregated_rows = []
        
        for fab_id, group in results_df.groupby('FAB_ID'):
            actual_label = group['Actual'].iloc[0]
            
            # Sum probability for each class
            sum_prob_0 = group['Prob_0'].sum()
            sum_prob_non0 = group['Prob_Non0'].sum()
            
            print(f"Sum of confidence for {fab_id}: {sum_prob_0:.2f} + {sum_prob_non0:.2f}\tAvg: {(group['Prob_0'].sum()/group['Prob_0'].count()):.2f} + {(group['Prob_Non0'].sum()/group['Prob_Non0'].count()):.2f}")

            # Calculate ratio of Zero confidence to total confidence
            total_confidence = sum_prob_0 + sum_prob_non0
            zero_ratio = sum_prob_0 / total_confidence if total_confidence > 0 else 0.5
            
            # Final prediction based on ratio threshold
            final_label = '0' if zero_ratio > 0.5 else 'Non-0'
            
            aggregated_rows.append({
                'FAB_ID': fab_id,
                'Actual': actual_label,
                'Predicted': final_label,
                'Zero_Ratio': zero_ratio,
                'Sum_Prob_0': sum_prob_0,
                'Sum_Prob_Non0': sum_prob_non0
            })
        
        aggregated_df = pd.DataFrame(aggregated_rows)
        
        # Now compute metrics using aggregated user-level predictions
        metrics = self._calculate_metrics(aggregated_df['Actual'], aggregated_df['Predicted'])
        
        # Create confusion matrix using aggregated predictions
        self._plot_confusion_matrix(
            aggregated_df['Actual'],
            aggregated_df['Predicted'],
            train_set,
            aggregated_df['Zero_Ratio']  # Pass confidence ratios for potential visualization
        )
        
        return metrics
    
    def _calculate_metrics(self, y_true, predictions):
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, predictions),
            'f1': f1_score(y_true, predictions, average='weighted', zero_division=0),
            'precision': precision_score(y_true, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_true, predictions, average='weighted', zero_division=0)
        }
    
    def _plot_confusion_matrix(self, y_true, predictions, train_set, confidence_ratios=None):
        """Create and save confusion matrix plot with confidence information"""
        output_dir = 'figures/prediction_plots'
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        cm = confusion_matrix(y_true, predictions)
        
        # Create normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        
        # Add title with metrics
        metrics = self._calculate_metrics(y_true, predictions)
        plt.title(f'Confusion Matrix (Accuracy: {metrics["accuracy"]:.3f}, F1: {metrics["f1"]:.3f})')
        
        # Add experiment details and average confidence ratio
        if confidence_ratios is not None:
            avg_conf = confidence_ratios.mean()
            plt.figtext(0.02, 0.02, 
                       f'Train Set: {train_set}\nAvg Zero Ratio: {avg_conf:.3f}', 
                       fontsize=8)
        else:
            plt.figtext(0.02, 0.02, f'Train Set: {train_set}', fontsize=8)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_conf_{train_set}.png')
        plt.close()

if __name__ == "__main__":
    # Initialize classifier
    classifier = SegmentClassifier()
    
    # Define directories
    train_dir = "data/FAB/FAB_A_Motion_Segments"  # Directory with A set segments
    test_dir = "data/FAB/FAB_B_Motion_Segments"   # Directory with B set segments
    
    # Train and evaluate on set A
    print("\nTraining on Set A, Testing on Set B:")
    metrics_a = classifier.train_and_evaluate(train_dir, test_dir, train_set='A')
    print("Metrics:", metrics_a)
    
    # Train and evaluate on set B
    print("\nTraining on Set B, Testing on Set A:")
    metrics_b = classifier.train_and_evaluate(test_dir, train_dir, train_set='B')
    print("Metrics:", metrics_b)
