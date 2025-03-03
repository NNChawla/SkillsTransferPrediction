import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier, TabPFNRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           balanced_accuracy_score, confusion_matrix, matthews_corrcoef)

from .base import ScorePredictionBase

class Processor(ScorePredictionBase):
    def __init__(self):
        super().__init__()

    _data_by_id_A = None  # Class-level cache
    _data_by_id_B = None
    _metadata = None
    _is_data_loaded = False  # Add flag to track if data is loaded
    _scaler = None  # Add class-level scaler
    _imputer = None  # Add class-level imputer
    _feature_cache = {}  # New class-level feature cache
    
    def __init__(self):
        self.ensure_data_loaded()
        self._scaler = StandardScaler()
        self._imputer = SimpleImputer(strategy='mean')
        self.predictor_type = None
    
    @classmethod
    def ensure_data_loaded(cls):
        """Load data only once per class"""
        if cls._is_data_loaded:  # Check flag before loading
            return
            
        if cls._metadata is None:
            # Try to load parquet first, fall back to CSV if needed
            metadata_parquet = './data/FAB/metadata.parquet'
            metadata_csv = './data/FAB/metadata.csv'
            
            if os.path.exists(metadata_parquet):
                cls._metadata = pq.read_table(metadata_parquet).to_pandas()
            else:
                cls._metadata = pd.read_csv(metadata_csv, header=0)
                
            cls._data_by_id_A = cls._process_files('A')
            cls._data_by_id_B = cls._process_files('B')
            cls._is_data_loaded = True  # Set flag after loading
    
    @classmethod
    def _process_files(cls, set_type):
        """Process parquet files for either set A or B"""
        data_dict = {}
        tracking_directory = f'./data/FAB/FAB_{set_type}_v3'
        assembly_directory = f'./data/FAB/Assembly_{set_type}'
        
        tracking_files = sorted([f for f in os.listdir(tracking_directory) if f.endswith('.parquet')])
        assembly_files = sorted([f for f in os.listdir(assembly_directory) if f.endswith('.csv')])
        
        for file in tracking_files:
            id_num = file.split('_')[0]
            file_path = os.path.join(tracking_directory, file)
            
            df = pq.read_table(file_path).to_pandas()
            id_metadata = cls._metadata[cls._metadata['ID'] == id_num]
            
            if len(id_metadata) == 0:
                continue
                
            id_metadata = id_metadata.iloc[0, 1:]
            for col_name, value in id_metadata.items():
                df[col_name] = value
            
            if id_num not in data_dict:
                data_dict[id_num] = []
            
            data_dict[id_num].append({'tracking_data': df})
        
        for file in assembly_files:
            id_num = file.split('_')[0]
            file_path = os.path.join(assembly_directory, file)
            
            df = pd.read_csv(file_path, header=0)
            data_dict[id_num][0]['assembly_data'] = df
            
        return data_dict

    @classmethod
    def _get_feature_cache_key(cls, feature_type, feature_name, pid, step):
        """Generate a unique key for feature caching"""
        return (feature_type, feature_name, pid, step)
    
    def get_model(self, model_config, task_type):
        """Create model instance based on configuration"""
        model_type = model_config.get('type')
        if not model_type:
            raise ValueError("Model type must be specified in config")
        
        params = model_config.get('model_params', {}).get(model_type, {})
        
        MODEL_REGISTRY = {
            'regression': {
                'knn': KNeighborsRegressor,
                'random_forest': RandomForestRegressor,
                'svm': SVR,
                'linear_svm': LinearSVR,
                'lightgbm': LGBMRegressor,
                'mlp': MLPRegressor,
                'tabpfn': TabPFNRegressor
            },
            'classification': {
                'knn': KNeighborsClassifier,
                'random_forest': RandomForestClassifier,
                'svm': SVC,
                'linear_svm': LinearSVC,
                'lightgbm': LGBMClassifier,
                'mlp': MLPClassifier,
                'tabpfn': TabPFNClassifier,
                'logistic': LogisticRegression
            }
        }
        
        model_class = MODEL_REGISTRY[task_type].get(model_type)
        
        if model_class is None:
            valid_types = list(MODEL_REGISTRY[task_type].keys())
            raise ValueError(f"Unknown model type: {model_type}. Valid types are: {valid_types}")
        
        try:
            base_model = model_class(**params)
            return base_model
        except Exception as e:
            raise ValueError(f"Error creating {model_type} model with params {params}: {str(e)}")
        
    def extract_features(self, assembly_df, tracking_df, pid, train_set=None, metadata_features=None, global_features=None, step = -1):
        """Extract features from a DataFrame with individual feature caching"""
        features = []
        used_features = []
        feature_count = 0

        df = tracking_df.copy()

        if (step >= 0):
            if (step == 0):
                start_time = 0
            else:
                start_time = assembly_df['Timestamp'][assembly_df[assembly_df['step'] == step - 1].index[-1]]
            
            step_filter = assembly_df[assembly_df['step'] == step]
            end_idx = step_filter.index[-1]
            end_time = assembly_df['Timestamp'][end_idx]
            start_idx = df['Timestamp'].sub(start_time).abs().idxmin()
            end_idx = df['Timestamp'].sub(end_time).abs().idxmin()
            df = df.iloc[start_idx:end_idx]

        # Add metadata features first
        if metadata_features:
            for meta_feature in metadata_features:
                if (meta_feature in ['A_Build_Time'] and train_set != 'A') or \
                   (meta_feature in ['B_Build_Time'] and train_set != 'B'):
                    continue

                cache_key = self._get_feature_cache_key('metadata', meta_feature, pid, step)
                
                if cache_key in self._feature_cache:
                    feature_value = self._feature_cache[cache_key]
                else:
                    # print(f"Calculating {meta_feature} for {pid} at step {step}")
                    if meta_feature in df.columns:
                        feature_value = df[meta_feature].iloc[0]
                        self._feature_cache[cache_key] = feature_value

                feature_count += 1
                used_features.append(meta_feature)
                features.append(feature_value)

        if global_features:
            for feature_type, feature_config in global_features.items():
                cols = feature_config['features']
                statistics = feature_config.get('statistics', ['min', 'max', 'median', 'mean', 'std'])
                
                if all(col in df.columns for col in cols):
                    feature_count += len(statistics) * len(cols)
                    used_features.extend(cols)
                    
                    # Define available statistics with safe handling of invalid values
                    stat_functions = {
                        'min': lambda x: x.min(),
                        'max': lambda x: x.max(),
                        'median': lambda x: x.median(),
                        'mean': lambda x: x.mean(),
                        'std': lambda x: x.std()
                    }
                    
                    # Calculate statistics for each column while maintaining order
                    for col in cols:
                        col_data = df[col]
                        
                        # Only calculate specified statistics
                        for stat_name in statistics:
                            if stat_name not in stat_functions:
                                continue
                            
                            cache_key = self._get_feature_cache_key('global', f"{col}_{stat_name}", pid, step)
                            
                            if cache_key in self._feature_cache:
                                feature_value = self._feature_cache[cache_key]
                            else:
                                feature_value = stat_functions[stat_name](col_data)
                                self._feature_cache[cache_key] = feature_value
                            
                            features.append(feature_value)
                        
                        col_data = None

        return features
    
    def get_train_test_data(self, train_set, cross_task, score_type="_Linear"):
        if cross_task:
            # Cross-task: train on VR data, test on IRL data.
            if train_set == 'A':  # If A is VR data
                train_data_dict = self._data_by_id_A
                test_data_dict = self._data_by_id_B  # B is IRL data
                train_score_col = f'Score_A{score_type}'
                test_score_col = f'Score_B{score_type}'
            else:  # If B is VR data
                train_data_dict = self._data_by_id_B
                test_data_dict = self._data_by_id_A  # A is IRL data
                train_score_col = f'Score_B{score_type}'
                test_score_col = f'Score_A{score_type}'
        else:
            # Within-task: use the same dataset for both training and testing (e.g., Set A).
            if train_set == 'A':
                train_data_dict = self._data_by_id_A
                test_data_dict = self._data_by_id_A
                train_score_col = f'Score_A{score_type}'
                test_score_col = f'Score_A{score_type}'
            else:
                train_data_dict = self._data_by_id_B
                test_data_dict = self._data_by_id_B
                train_score_col = f'Score_B{score_type}'
                test_score_col = f'Score_B{score_type}'
        
        return train_data_dict, test_data_dict, train_score_col, test_score_col
    
    def score_to_class(self, score, score_threshold):
        """Convert numerical score to class category"""        
        score = int(score)
        labels = [0, 1]
        
        if score <= score_threshold:
            return labels[0]
        elif score > score_threshold:
            return labels[1]
        else:
            raise ValueError(f"Invalid score value: {score}")
        
    # Create the appropriate scoring function once
    def get_scoring_function(self, measure):
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
        
    def evaluate_predictions(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        sensitivity = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        specificity = recall_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
        scoring_function = self.get_scoring_function('mcc')
        feature_selection_measure = scoring_function(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'feature_selection_measure': feature_selection_measure,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return results