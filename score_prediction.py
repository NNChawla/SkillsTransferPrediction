import pandas as pd
import os
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import product, combinations
import yaml
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
import pyarrow.parquet as pq
import gc

class ScorePredictionBase:
    _data_by_id_A = None  # Class-level cache
    _data_by_id_B = None
    _metadata = None
    _is_data_loaded = False  # Add flag to track if data is loaded
    _scaler = None  # Add class-level scaler
    _imputer = None  # Add class-level imputer
    _feature_cache = {}  # New class-level feature cache
    
    def __init__(self, config_path='experiment_config.yaml'):
        self.config = self.load_config(config_path)
        self.ensure_data_loaded()
        if self._scaler is None:
            self._scaler = StandardScaler()
        if self._imputer is None:
            self._imputer = SimpleImputer(strategy='mean')
    
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
            print("Data loaded into class-level cache")
    
    @classmethod
    def _process_files(cls, set_type):
        """Process parquet files for either set A or B"""
        data_dict = {}
        directory = f'./data/FAB/FAB_{set_type}_v2'
        
        # Get list of parquet files from directory
        files = sorted([f for f in os.listdir(directory) if f.endswith('.parquet')])
        print(f"Found {len(files)} parquet files in {directory}")
        
        for file in files:
            id_num = file.split('_')[0]
            file_path = os.path.join(directory, file)
            
            # Read parquet file instead of CSV
            df = pq.read_table(file_path).to_pandas()
            id_metadata = cls._metadata[cls._metadata['ID'] == id_num]
            
            if len(id_metadata) == 0:
                continue
                
            # Add metadata columns
            id_metadata = id_metadata.iloc[0, 1:]
            for col_name, value in id_metadata.items():
                df[col_name] = value
            
            if id_num not in data_dict:
                data_dict[id_num] = []
            
            data_dict[id_num].append({'data': df})
            #gc.collect()
            
        return data_dict
    
    def load_config(self, config_path):
        """Load experiment configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract_features(self, df, pid, enabled_trackers=None, enabled_measurements=None, 
                        enabled_metadata=None, global_features=None, train_set=None,
                        segment_size=None, sample_rate=None):
        """Extract features from a DataFrame with individual feature caching"""
        features = []
        used_features = []
        feature_count = 0
        df_id = df.index[0]  # Use first timestamp as ID

        # Add metadata features first
        if enabled_metadata:
            for meta_feature in enabled_metadata:
                if (meta_feature in ['A_Build_Time'] and train_set != 'A') or \
                   (meta_feature in ['B_Build_Time'] and train_set != 'B'):
                    continue

                cache_key = self._get_feature_cache_key(df_id, 'metadata', meta_feature, 
                                                      segment_size, sample_rate, pid)
                
                if cache_key in self._feature_cache:
                    feature_value = self._feature_cache[cache_key]
                else:
                    if meta_feature in df.columns:
                        feature_value = df[meta_feature].iloc[0]
                        self._feature_cache[cache_key] = feature_value
                    else:
                        feature_value = 0

                feature_count += 1
                used_features.append(meta_feature)
                features.append(feature_value)

        # Add global features if specified
        if global_features:
            for feature_type, feature_cols in global_features.items():
                cols = feature_cols
                
                if all(col in df.columns for col in cols):
                    feature_count += 5 * len(cols)
                    used_features.extend(cols)
                    
                    # Calculate statistics for each column while maintaining order
                    for col in cols:
                        col_data = df[col]
                        
                        # Calculate and cache each statistic
                        stats = [
                            ('min', lambda x: x.min() if not np.isnan(x.min()) else 0),
                            ('max', lambda x: x.max() if not np.isnan(x.max()) else 0),
                            ('median', lambda x: x.median() if not np.isnan(x.median()) else 0),
                            ('mean', lambda x: x.mean() if not np.isnan(x.mean()) else 0),
                            ('std', lambda x: x.std() if not np.isnan(x.std()) else 0)
                        ]
                        
                        for stat_name, stat_func in stats:
                            cache_key = self._get_feature_cache_key(df_id, 'global', 
                                                                  f"{col}_{stat_name}",
                                                                  segment_size, sample_rate, pid)
                            
                            if cache_key in self._feature_cache:
                                feature_value = self._feature_cache[cache_key]
                            else:
                                feature_value = stat_func(col_data)
                                self._feature_cache[cache_key] = feature_value
                            
                            features.append(feature_value)
                        
                        col_data = None
        
        # Add tracker-specific features
        trackers = enabled_trackers if enabled_trackers else ['Head', 'LeftHand', 'RightHand']
        measurements = enabled_measurements if enabled_measurements else {
            'position': ['_position_x', '_position_y', '_position_z'],
            'euler': ['_euler_x', '_euler_y', '_euler_z'],
            'quat': ['_quat_x', '_quat_y', '_quat_z', '_quat_w'],
            'sixD': ['_sixD_a', '_sixD_b', '_sixD_c', '_sixD_d', '_sixD_e', '_sixD_f']
        }
        
        # Vectorized feature extraction maintaining original order
        for tracker in trackers:
            for measure_type, measure_cols in measurements.items():
                cols = [tracker + suffix for suffix in measure_cols]
                
                if all(col in df.columns for col in cols):
                    feature_count += 5 * len(cols)
                    used_features.extend(cols)
                    
                    # Calculate statistics for each column while maintaining order
                    for col in cols:
                        col_data = df[col]
                        
                        # Calculate and cache each statistic
                        stats = [
                            ('min', lambda x: x.min() if not np.isnan(x.min()) else 0),
                            ('max', lambda x: x.max() if not np.isnan(x.max()) else 0),
                            ('median', lambda x: x.median() if not np.isnan(x.median()) else 0),
                            ('mean', lambda x: x.mean() if not np.isnan(x.mean()) else 0),
                            ('std', lambda x: x.std() if not np.isnan(x.std()) else 0)
                        ]
                        
                        for stat_name, stat_func in stats:
                            cache_key = self._get_feature_cache_key(df_id, 'tracker', 
                                                                  f"{col}_{stat_name}",
                                                                  segment_size, sample_rate, pid)
                            
                            if cache_key in self._feature_cache:
                                feature_value = self._feature_cache[cache_key]
                            else:
                                feature_value = stat_func(col_data)
                                self._feature_cache[cache_key] = feature_value
                            
                            features.append(feature_value)
                        
                        col_data = None
                else:
                    # If columns don't exist, add zeros for all missing statistics
                    features.extend([0] * (5 * len(measure_cols)))
        
        logging.info(f"Number of features extracted per segment: {feature_count}")
        used_features = None
        return features

    @classmethod
    def _get_feature_cache_key(cls, df_id, feature_type, feature_name, segment_size, sample_rate, pid):
        """Generate a unique key for feature caching"""
        return (df_id, feature_type, feature_name, segment_size, str(sample_rate), pid)

    def segment_and_extract_features(self, df, pid, segment_size='20s', enabled_trackers=None,
                                   enabled_measurements=None, enabled_metadata=None, 
                                   global_features=None, train_set=None, sample_rate=None):
        """Segments the dataframe by time and extracts features from each segment"""
        df = df.copy()
        
        logging.info(f"\nSegmenting data with segment_size: {segment_size}")
        logging.info(f"Original data length: {len(df)}")
        
        # Convert timestamps for both full and segmented cases
        if 'Timestamp' in df.columns:
            logging.info("\nTimestamp is in columns")
            df['Timestamp'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(df['Timestamp'], unit='s')
            logging.info(f"Timestamp type after conversion: {df['Timestamp'].dtype}")
            df.set_index('Timestamp', inplace=True)
        else:
            logging.info("\nTimestamp not in columns")
            df.index = pd.to_datetime('2024-01-01') + pd.to_timedelta(df.index, unit='s')
        
        # Resample if sample_rate is specified
        if sample_rate is not None:
            logging.info(f"Resampling data to {sample_rate}")
            original_len = len(df)
            df = df.resample(sample_rate).mean()
            logging.info(f"Data length after resampling: {len(df)} (was {original_len})")
        
        # If segment_size is 'full', process entire sequence at once
        if segment_size == 'full':
            features = self.extract_features(df, pid, enabled_trackers, enabled_measurements, 
                                          enabled_metadata, global_features, train_set,
                                          segment_size, sample_rate)
            return [features]
        
        # Resample data into segments
        segments = [group for _, group in df.resample(segment_size)]
        
        # Extract features from each segment
        segment_features = []
        for i, segment in enumerate(segments):
            if not segment.empty:
                features = self.extract_features(segment, pid, enabled_trackers, enabled_measurements, 
                                              enabled_metadata, global_features, train_set,
                                              segment_size, sample_rate)
                segment_features.append(features)
        
        segments = None
        return segment_features

    def get_model(self, model_config):
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
                'lightgbm': lgb.LGBMRegressor,
                'mlp': MLPRegressor
            },
            'classification': {
                'knn': KNeighborsClassifier,
                'random_forest': RandomForestClassifier,
                'svm': SVC,
                'lightgbm': lgb.LGBMClassifier,
                'mlp': MLPClassifier
            }
        }
        
        predictor_type = 'regression' if isinstance(self, RegressionPredictor) else 'classification'
        model_class = MODEL_REGISTRY[predictor_type].get(model_type)
        
        if model_class is None:
            valid_types = list(MODEL_REGISTRY[predictor_type].keys())
            raise ValueError(f"Unknown model type: {model_type}. Valid types are: {valid_types}")
        
        try:
            model = model_class(**params)
            return model
        except Exception as e:
            raise ValueError(f"Error creating {model_type} model with params {params}: {str(e)}")

    @classmethod
    def clear_feature_cache(cls):
        """Clear the feature cache to free memory"""
        cls._feature_cache.clear()

class RegressionPredictor(ScorePredictionBase):
    def __init__(self, config_path='experiment_config.yaml'):
        super().__init__(config_path)
        self.model = self.get_model(self.config['model'])
        
    def evaluate_predictions(self, y_true, predictions, segment_size, test_data=None):
        """Calculate regression metrics"""
        if segment_size == 'full':
            # For full sequences, calculate metrics directly
            mse = mean_squared_error(y_true, predictions)
            mae = mean_absolute_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)
            rmse = np.sqrt(mse)
        else:
            # Create results DataFrame with all segment predictions
            results_df = pd.DataFrame({
                'PID': [d['PID'] for d in test_data],
                'Score': y_true,
                'Prediction': predictions,
            })
            
            # Get mean predictions for each participant
            majority_predictions = results_df.groupby('PID').agg({
                'Score': 'first',
                'Prediction': 'mean'
            }).reset_index()
            
            # Calculate metrics using mean predictions
            mse = mean_squared_error(majority_predictions['Score'], majority_predictions['Prediction'])
            mae = mean_absolute_error(majority_predictions['Score'], majority_predictions['Prediction'])
            r2 = r2_score(majority_predictions['Score'], majority_predictions['Prediction'])
            rmse = np.sqrt(mse)
        
        # Log detailed results
        logging.info(f"\nResults for segment_size={segment_size}:")
        logging.info(f"MSE: {mse:.3f}")
        logging.info(f"RMSE: {rmse:.3f}")
        logging.info(f"MAE: {mae:.3f}")
        logging.info(f"R2: {r2:.3f}")
        
        return rmse, mae, r2, mse

    def run_single_experiment(self, handle_nan, train_set, sample_rate, segment_size, 
                            tracker_key, trackers, measure_key, measurements, 
                            meta_key, metadata_features, global_key, global_features, **kwargs):
        """Run a single regression experiment with given parameters"""
        
        # Use cached data instead of reloading
        if train_set == 'A':
            train_data_dict = self._data_by_id_A
            test_data_dict = self._data_by_id_B
            train_score_col = 'Score_A'
            test_score_col = 'Score_B'
        else:  # train_set == 'B'
            train_data_dict = self._data_by_id_B
            test_data_dict = self._data_by_id_A
            train_score_col = 'Score_B'
            test_score_col = 'Score_A'
        
        # Prepare training data
        train_data = []
        for pid_num, id_num in enumerate(train_data_dict.keys()):
            df = train_data_dict[id_num][0]['data']
            segment_features = self.segment_and_extract_features(
                df, pid=id_num,
                segment_size=segment_size,
                enabled_trackers=trackers,
                enabled_measurements=measurements,
                enabled_metadata=metadata_features,
                global_features=global_features,
                train_set=train_set,
                sample_rate=sample_rate
            )
            
            score = df[train_score_col].iloc[0]
            df = None  # Clean up DataFrame
            
            for features in segment_features:
                train_data.append({
                    'features': features,
                    'PID': id_num,
                    'PIDnum': pid_num,
                    'Score': score
                })
            segment_features = None  # Clean up segment features
            #if pid_num % 10 == 0:
                #gc.collect()
        
        # Prepare test data with similar cleanup
        test_data = []
        for pid_num, id_num in enumerate(test_data_dict.keys()):
            df = test_data_dict[id_num][0]['data']
            segment_features = self.segment_and_extract_features(
                df, pid=id_num,
                segment_size=segment_size,
                enabled_trackers=trackers,
                enabled_measurements=measurements,
                enabled_metadata=metadata_features,
                global_features=global_features,
                train_set=train_set,
                sample_rate=sample_rate
            )
            
            score = df[test_score_col].iloc[0]
            df = None  # Clean up DataFrame
            
            for features in segment_features:
                test_data.append({
                    'features': features,
                    'PID': id_num,
                    'PIDnum': pid_num,
                    'Score': score
                })
            segment_features = None  # Clean up segment features
            #if pid_num % 10 == 0:
                #gc.collect()
        
        # Prepare numpy arrays
        X_train = np.array([d['features'] for d in train_data])
        y_train = np.array([d['Score'] for d in train_data])
        X_test = np.array([d['features'] for d in test_data])
        y_test = np.array([d['Score'] for d in test_data])
        
        # After features are processed but before scaling/training
        if segment_size == 'full':  # Only export for full sequence analysis
            # Create feature names
            feature_names = []
            
            # Add metadata feature names
            if metadata_features:
                for meta_feature in metadata_features:
                    if (meta_feature in ['A_Build_Time'] and train_set != 'A') or \
                       (meta_feature in ['B_Build_Time'] and train_set != 'B'):
                        continue
                    else:
                        feature_names.append(meta_feature)
                print(f"After metadata: {len(feature_names)} features")
                
            # Add global feature names
            if global_features:
                for feature_type, cols in global_features.items():
                    for col in cols:
                        feature_names.extend([
                            f"{col}_min", f"{col}_max", f"{col}_median",
                            f"{col}_mean", f"{col}_std"
                        ])
                print(f"After global: {len(feature_names)} features")
            
            # Add tracker-specific feature names
            for tracker in trackers:
                for measure_type, measure_cols in measurements.items():
                    for col_suffix in measure_cols:
                        col = tracker + col_suffix
                        feature_names.extend([
                            f"{col}_min", f"{col}_max", f"{col}_median",
                            f"{col}_mean", f"{col}_std"
                        ])
                print(f"After tracker {tracker}: {len(feature_names)} features")
            
            print(f"X_train shape: {X_train.shape}")
            print(f"Number of feature names: {len(feature_names)}")
            
            # Debug prints
            print(f"First row of X_train: {X_train[0][:5]}")  # Print first 5 values
            print(f"Second row of X_train: {X_train[1][:5]}")  # Print first 5 values
            
            # Create list of PIDs and scores in the same order as X_train
            pids = [d['PID'] for d in train_data]
            scores = y_train.tolist()
            
            # Create DataFrame with PID first, then features, then score
            feature_df = pd.DataFrame(X_train, columns=feature_names)
            feature_df.insert(0, 'PID', pids)  # Insert PID as first column
            feature_df['Score'] = scores
            
            # Debug prints
            print("\nFirst few rows of feature_df:")
            print(feature_df.head(2))
            
            # Save to CSV
            output_dir = 'results/feature_exports'
            os.makedirs(output_dir, exist_ok=True)
            filename = f'{output_dir}/features_{train_set}_{tracker_key}_{measure_key}_{meta_key}_{global_key}.csv'
            feature_df.to_csv(filename, index=False)
            print(f"Exported features to: {filename}")

        # Clear data for memory management
        test_data_copy = test_data.copy()  # Keep a copy for evaluation
        train_data = None  # Clear train_data AFTER using it for feature export
        test_data = None
        #gc.collect()
        
        # Handle NaN values
        if handle_nan == 'drop':
            train_valid_mask = ~np.isnan(X_train).any(axis=1)
            test_valid_mask = ~np.isnan(X_test).any(axis=1)
            
            X_train = X_train[train_valid_mask]
            y_train = y_train[train_valid_mask]
            X_test = X_test[test_valid_mask]
            y_test = y_test[test_valid_mask]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
        else:  # handle_nan == 'impute'
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            imputer = SimpleImputer(strategy='mean')
            X_train_scaled = imputer.fit_transform(X_train_scaled)
            X_test_scaled = imputer.transform(X_test_scaled)
        
        # Update these debug prints
        #print(f"Data shape before preprocessing: {X_train.shape if X_train is not None else 'None'}")
        #print(f"Number of samples: {len(X_train)}")
        #print(f"Number of features: {X_train.shape[1]}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_train = None  # Clear original data
        #gc.collect()
        
        # Train and predict
        self.model.fit(X_train_scaled, y_train)
        
        predictions = self.model.predict(X_test_scaled)
        X_test = None  # Clear test data
        X_test_scaled = None
        #gc.collect()
        
        # Calculate metrics
        rmse, mae, r2, mse = self.evaluate_predictions(y_test, predictions, segment_size, test_data_copy)
        
        # After evaluation, clean up remaining data
        test_data_copy = None
        #gc.collect()

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse,
            'handle_nan': handle_nan,
            'train_set': train_set,
            'sample_rate': sample_rate,
            'segment_size': segment_size,
            'tracker_key': tracker_key,
            'measure_key': measure_key,
            'meta_key': meta_key,
            'global_key': global_key
        }

class ClassificationPredictor(ScorePredictionBase):
    def __init__(self, config_path='experiment_config.yaml'):
        super().__init__(config_path)
        self.model = self.get_model(self.config['model'])
        
    def score_to_class(self, score):
        """Convert numerical score to class category"""
        try:
            score = float(score)
            bins = [0, 5, 10, 15, 20, 25]
            labels = ['0-5', '5-10', '10-15', '15-20', '20-25']
            
            if score < 0:
                return labels[0]
            
            for i in range(len(bins)-1):
                if bins[i] <= score <= bins[i+1]:
                    return labels[i]
            return labels[-1]
        except (ValueError, TypeError):
            logging.error(f"Invalid score value: {score}")
            return labels[0]
            
    def evaluate_predictions(self, y_true, predictions, segment_size, test_data=None):
        """Calculate classification metrics"""
        try:
            if segment_size == 'full':
                metrics = self._calculate_classification_metrics(y_true, predictions)
            else:
                # Create results DataFrame with all segment predictions
                results_df = pd.DataFrame({
                    'PID': [d['PID'] for d in test_data],
                    'Score': y_true,
                    'Prediction': predictions,
                })
                
                # Get majority vote predictions for each participant
                majority_predictions = results_df.groupby('PID').agg({
                    'Score': 'first',
                    'Prediction': lambda x: x.value_counts().index[0]
                }).reset_index()
                
                metrics = self._calculate_classification_metrics(
                    majority_predictions['Score'],
                    majority_predictions['Prediction']
                )
                
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0
            
    def _calculate_classification_metrics(self, y_true, predictions):
        """Helper method to calculate classification metrics"""
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
        precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
        return accuracy, f1, precision, recall 

    def run_single_experiment(self, handle_nan, train_set, sample_rate, segment_size, 
                            tracker_key, trackers, measure_key, measurements, 
                            meta_key, metadata_features, **kwargs):
        """Run a single classification experiment with given parameters"""
        # Determine which dataset to use for training and testing
        if train_set == 'A':
            train_data_dict = self._data_by_id_A
            test_data_dict = self._data_by_id_B
            train_score_col = 'Score_A'
            test_score_col = 'Score_B'
        else:  # train_set == 'B'
            train_data_dict = self._data_by_id_B
            test_data_dict = self._data_by_id_A
            train_score_col = 'Score_B'
            test_score_col = 'Score_A'
        
        # Prepare training data
        train_data = []
        for pid_num, id_num in enumerate(train_data_dict.keys()):
            df = train_data_dict[id_num][0]['data']
            segment_features = self.segment_and_extract_features(
                df, segment_size=segment_size,
                enabled_trackers=trackers,
                enabled_measurements=measurements,
                enabled_metadata=metadata_features,
                train_set=train_set,
                sample_rate=sample_rate
            )
            
            score = self.score_to_class(df[train_score_col].iloc[0])
            
            for features in segment_features:
                train_data.append({
                    'features': features,
                    'PID': id_num,
                    'PIDnum': pid_num,
                    'Score': score
                })
        
        # Prepare test data
        test_data = []
        for pid_num, id_num in enumerate(test_data_dict.keys()):
            df = test_data_dict[id_num][0]['data']
            segment_features = self.segment_and_extract_features(
                df, segment_size=segment_size,
                enabled_trackers=trackers,
                enabled_measurements=measurements,
                enabled_metadata=metadata_features,
                train_set=train_set,
                sample_rate=sample_rate
            )
            
            score = self.score_to_class(df[test_score_col].iloc[0])
            
            for features in segment_features:
                test_data.append({
                    'features': features,
                    'PID': id_num,
                    'PIDnum': pid_num,
                    'Score': score
                })
        
        # Prepare numpy arrays
        X_train = np.array([d['features'] for d in train_data])
        y_train = np.array([d['Score'] for d in train_data])
        X_test = np.array([d['features'] for d in test_data])
        y_test = np.array([d['Score'] for d in test_data])
        
        # Handle NaN values
        if handle_nan == 'drop':
            train_valid_mask = ~np.isnan(X_train).any(axis=1)
            test_valid_mask = ~np.isnan(X_test).any(axis=1)
            
            X_train = X_train[train_valid_mask]
            y_train = y_train[train_valid_mask]
            X_test = X_test[test_valid_mask]
            y_test = y_test[test_valid_mask]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
        else:  # handle_nan == 'impute'
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            imputer = SimpleImputer(strategy='mean')
            X_train_scaled = imputer.fit_transform(X_train_scaled)
            X_test_scaled = imputer.transform(X_test_scaled)
        
        # Update these debug prints
        #print(f"Data shape before preprocessing: {X_train.shape if X_train is not None else 'None'}")
        #print(f"Number of samples: {len(X_train)}")
        #print(f"Number of features: {X_train.shape[1]}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train and predict
        self.model.fit(X_train_scaled, y_train)
        predictions = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy, f1, precision, recall = self.evaluate_predictions(y_test, predictions, segment_size, test_data)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'handle_nan': handle_nan,
            'train_set': train_set,
            'sample_rate': sample_rate,
            'segment_size': segment_size,
            'tracker_key': tracker_key,
            'measure_key': measure_key,
            'meta_key': meta_key
        }