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

class SegmentClassifier:
    def __init__(self, config_path='segment_config.yaml'):
        self.config = self.load_config(config_path)
        self._metadata = self.load_metadata()
        self.model = self.get_model(self.config['model'])
        self._scaler = RobustScaler()
        self._imputer = SimpleImputer(strategy='mean')
    
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
        
        # Extract features and labels
        feature_cols = ['duration_frames', 'mean_velocity', 'max_velocity', 
                       'mean_acceleration', 'displacement', 'path_length']
        X = combined_df[feature_cols].values
        y = combined_df['Score'].values
        ids = combined_df['FAB_ID'].values
        
        return X, y, ids, feature_cols
    
    def train_and_evaluate(self, train_dir, test_dir, train_set='A'):
        """Train and evaluate the classifier"""
        # Load and preprocess data
        X_train, y_train, train_ids, feature_names = self.load_and_preprocess_data(
            train_dir, set_type=train_set)
        X_test, y_test, test_ids, _ = self.load_and_preprocess_data(
            test_dir, set_type='B' if train_set == 'A' else 'A')
        
        # Scale features
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)
        
        # Handle missing values
        X_train_scaled = self._imputer.fit_transform(X_train_scaled)
        X_test_scaled = self._imputer.transform(X_test_scaled)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'FAB_ID': test_ids,
            'Actual': y_test,
            'Predicted': predictions
        })
        
        # Get majority vote predictions for each participant
        majority_predictions = results_df.groupby('FAB_ID').agg({
            'Actual': 'first',
            'Predicted': lambda x: x.value_counts().index[0]
        }).reset_index()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            majority_predictions['Actual'],
            majority_predictions['Predicted']
        )
        
        # Create confusion matrix plot
        self._plot_confusion_matrix(
            majority_predictions['Actual'],
            majority_predictions['Predicted'],
            train_set
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
    
    def _plot_confusion_matrix(self, y_true, predictions, train_set):
        """Create and save confusion matrix plot"""
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
        
        # Add experiment details
        plt.figtext(0.02, 0.02, f'Train Set: {train_set}', fontsize=8)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_{train_set}.png')
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