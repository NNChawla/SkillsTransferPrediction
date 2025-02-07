import pandas as pd
import os
import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score,
                           balanced_accuracy_score)
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import yaml
import logging
from sklearn.model_selection import LeaveOneOut, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
import pyarrow.parquet as pq
import gc
import shap
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor, TabPFNClassifier
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import colors
from utils.supervised_pca import SupervisedPCA
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
# from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix, make_scorer)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import LabelEncoder
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
            self._scaler = RobustScaler()
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
        directory = f'./data/FAB/FAB_{set_type}_v3' #HandRelative_Motion_PQ'
        
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
    
    def extract_features(self, df, pid, enabled_metadata=None, global_features=None, 
                        train_set=None, segment_size=None, sample_rate=None):
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
            for feature_type, feature_config in global_features.items():
                cols = feature_config['features']
                statistics = feature_config.get('statistics', ['min', 'max', 'median', 'mean', 'std'])
                
                if all(col in df.columns for col in cols):
                    feature_count += len(statistics) * len(cols)
                    used_features.extend(cols)
                    
                    # Define available statistics
                    stat_functions = {
                        'min': lambda x: x.min() if not np.isnan(x.min()) else 0,
                        'max': lambda x: x.max() if not np.isnan(x.max()) else 0,
                        'median': lambda x: x.median() if not np.isnan(x.median()) else 0,
                        'mean': lambda x: x.mean() if not np.isnan(x.mean()) else 0,
                        'std': lambda x: x.std() if not np.isnan(x.std()) else 0
                    }
                    
                    # Calculate statistics for each column while maintaining order
                    for col in cols:
                        col_data = df[col]
                        
                        # Only calculate specified statistics
                        for stat_name in statistics:
                            if stat_name not in stat_functions:
                                logging.warning(f"Unknown statistic '{stat_name}' for feature {col}")
                                continue
                            
                            cache_key = self._get_feature_cache_key(df_id, 'global', 
                                                                  f"{col}_{stat_name}",
                                                                  segment_size, sample_rate, pid)
                            
                            if cache_key in self._feature_cache:
                                feature_value = self._feature_cache[cache_key]
                            else:
                                feature_value = stat_functions[stat_name](col_data)
                                self._feature_cache[cache_key] = feature_value
                            
                            features.append(feature_value)
                        
                        col_data = None
        
        logging.info(f"Number of features extracted per segment: {feature_count}")
        used_features = None
        return features
    
    def _apply_transformation(self, data, transform_type):
        """Apply the specified transformation to the data"""
        # print(f"\nApplying {transform_type} transformation")
        # print(f"Input data stats:")
        # print(f"Mean: {data.mean():.3f}")
        # print(f"Std: {data.std():.3f}")
        # print(f"Skew: {stats.skew(data):.3f}")
        # print(f"Min: {data.min():.3f}")
        
        # Remove inf values and convert to nan
        data = pd.Series(np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan))
        
        # Remove NaN values for transformation
        clean_data = data.dropna()
        
        # If no valid data points remain, return None
        if len(clean_data) == 0:
            print("No valid data points after cleaning")
            return None, None

        try:
            if transform_type == 'log':
                offset = abs(clean_data.min()) + 1 if clean_data.min() <= 0 else 0
                transformed = np.log1p(clean_data + offset)
            elif transform_type == 'sqrt':
                offset = abs(clean_data.min()) if clean_data.min() < 0 else 0
                transformed = np.sqrt(clean_data + offset)
            elif transform_type == 'boxcox':
                offset = abs(clean_data.min()) + 1 if clean_data.min() <= 0 else 0
                transformed, _ = stats.boxcox(clean_data + offset)
            elif transform_type == 'yeojohnson':
                transformed, _ = stats.yeojohnson(clean_data)
            else:
                raise ValueError(f"Unknown transformation type: {transform_type}")
            
            # print(f"Output data stats:")
            # print(f"Mean: {transformed.mean():.3f}")
            # print(f"Std: {transformed.std():.3f}")
            # print(f"Skew: {stats.skew(transformed):.3f}")
            
            return transformed, f'{transform_type}(x + offset)'
            
        except Exception as e:
            print(f"Error in transformation: {str(e)}")
            return None, None

    @classmethod
    def _get_feature_cache_key(cls, df_id, feature_type, feature_name, segment_size, sample_rate, pid):
        """Generate a unique key for feature caching"""
        return (df_id, feature_type, feature_name, segment_size, str(sample_rate), pid)

    def segment_and_extract_features(self, df, pid, segment_size='20s',
                                   enabled_metadata=None, global_features=None,
                                   train_set=None, sample_rate=None):
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
            features = self.extract_features(df, pid, enabled_metadata, global_features, 
                                          train_set, segment_size, sample_rate)
            return [features]
        
        # Resample data into segments
        segments = [group for _, group in df.resample(segment_size)]
        
        # Extract features from each segment
        segment_features = []
        for i, segment in enumerate(segments):
            if not segment.empty:
                features = self.extract_features(segment, pid, enabled_metadata, global_features, 
                                              train_set, segment_size, sample_rate)
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
                'linear_svm': LinearSVR,
                'lightgbm': lgb.LGBMRegressor,
                'mlp': MLPRegressor,
                'tabpfn': TabPFNRegressor
            },
            'classification': {
                'knn': KNeighborsClassifier,
                'random_forest': RandomForestClassifier,
                'svm': SVC,
                'linear_svm': LinearSVC,
                'lightgbm': lgb.LGBMClassifier,
                'mlp': MLPClassifier,
                'tabpfn': TabPFNClassifier,
                'logistic': LogisticRegression
            }
        }
        
        predictor_type = 'regression' if isinstance(self, RegressionPredictor) else 'classification'
        model_class = MODEL_REGISTRY[predictor_type].get(model_type)
        
        if model_class is None:
            valid_types = list(MODEL_REGISTRY[predictor_type].keys())
            raise ValueError(f"Unknown model type: {model_type}. Valid types are: {valid_types}")
        
        try:
            base_model = model_class(**params)
            
            # Check if PCA should be used
            use_pca = model_config.get('use_pca', False)
            if use_pca:
                n_components = model_config.get('pca_components', None)
                pca = PCA(n_components=n_components)
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', pca),
                    ('classifier', base_model)
                ])
            else:
                model = base_model
            
            return model
        except Exception as e:
            raise ValueError(f"Error creating {model_type} model with params {params}: {str(e)}")

    @classmethod
    def clear_feature_cache(cls):
        """Clear the feature cache to free memory"""
        cls._feature_cache.clear()

    def calculate_shap_values(self, X_train_scaled, X_test_scaled, feature_names=None):
        """Calculate SHAP values for supported models"""
        if not hasattr(self, 'model'):
            raise ValueError("Model must be trained before calculating SHAP values")

        # For pipeline, get the actual classifier
        if isinstance(self.model, Pipeline):
            classifier = self.model.named_steps['classifier']
        else:
            classifier = self.model

        # Check if model type is supported
        if not isinstance(classifier, (RandomForestRegressor, RandomForestClassifier, 
                                     lgb.LGBMRegressor, lgb.LGBMClassifier)):
            logging.warning("SHAP analysis skipped - only supported for Random Forest and LightGBM models")
            return None

        try:
            if isinstance(classifier, (RandomForestRegressor, RandomForestClassifier)):
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_test_scaled)
            else:  # LightGBM models
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_test_scaled)
                if isinstance(shap_values, list):  # For multi-class classification
                    shap_values = np.array(shap_values)

            return {
                'shap_values': shap_values,
                'explainer': explainer,
                'feature_names': feature_names
            }
        except Exception as e:
            logging.error(f"Error calculating SHAP values: {str(e)}")
            return None

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
                            metadata_features, global_features, **kwargs):
        """Run a single experiment with given parameters"""
        # Store train_set as instance variable at the start of the method
        self.train_set = train_set

        #print(f"Global features: {global_features}")
        #print(f"Metadata features: {metadata_features}")
        
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
        
        # Get feature names and apply transformations
        feature_names = self.get_feature_names(metadata_features, global_features)
        
        # Debug: Print feature values before transformation
        # for i, name in enumerate(feature_names):
        #     if 'RightHand_velocity_z' in name:
        #         print(f"\nBefore transform - {name}:")
        #         print(f"Train mean: {np.mean(X_train[:, i]):.3f}")
        #         print(f"Train std: {np.std(X_train[:, i]):.3f}")
        #         print(f"Train skew: {stats.skew(X_train[:, i]):.3f}")
        
        # Process each feature type
        if global_features:
            for feature_name, feature_config in global_features.items():
                # Debug print the actual config
                # print(f"\nFeature config for {feature_name}:")
                # print(f"Config: {feature_config}")
                
                transform_type = feature_config.get('transform', 'none')
                base_feature = feature_config['features'][0]  # Get the base feature name
                statistics = feature_config.get('statistics', [])
                
                # print(f"\nChecking feature {feature_name}:")  # Debug print
                # print(f"Transform type from config: {transform_type}")    # Debug print
                # print(f"Base feature: {base_feature}")        # Debug print
                
                # Find all columns that belong to this feature
                feature_indices = []
                for stat in statistics:
                    feature_name = f"{base_feature}_{stat}"
                    try:
                        idx = feature_names.index(feature_name)
                        feature_indices.append(idx)
                        # print(f"Found {feature_name} at index {idx}")  # Debug print
                    except ValueError:
                        print(f"Could not find {feature_name}")  # Debug print
                        continue
                
                if transform_type != 'none' and feature_indices:
                    # print(f"\nProcessing {base_feature} with {transform_type} transform")
                    for col_idx in feature_indices:
                        feature_name = feature_names[col_idx]
                        # print(f"Transforming column {col_idx}: {feature_name}")  # Debug print
                        
                        # Create Series and verify data
                        train_series = pd.Series(X_train[:, col_idx], name=feature_name)
                        # print(f"Series stats before transform:")
                        # print(f"Mean: {train_series.mean():.3f}")
                        # print(f"Std: {train_series.std():.3f}")
                        # print(f"Skew: {stats.skew(train_series):.3f}")
                        
                        # Transform training data
                        train_transformed, transform_info = self._apply_transformation(train_series, transform_type)
                        
                        if train_transformed is not None:
                            # print("Transform successful, updating data")  # Debug print
                            # Directly assign the numpy array
                            X_train[:, col_idx] = train_transformed
                            # print(f"Transformed {feature_name} using {transform_info}")
                            
                            # Verify the update
                            # print(f"Updated column stats:")
                            # print(f"Mean: {np.mean(X_train[:, col_idx]):.3f}")
                            # print(f"Std: {np.std(X_train[:, col_idx]):.3f}")
                            # print(f"Skew: {stats.skew(X_train[:, col_idx]):.3f}")
                        else:
                            pass # print(f"Transform returned None for {feature_name}")  # Debug print
                            
                        # Transform test data
                        test_series = pd.Series(X_test[:, col_idx], name=feature_name)
                        test_transformed, _ = self._apply_transformation(test_series, transform_type)
                        if test_transformed is not None:
                            X_test[:, col_idx] = test_transformed  # Directly assign the numpy array

        # Debug: Print feature values after transformation
        # for i, name in enumerate(feature_names):
        #     if 'RightHand_velocity_z' in name:
        #         print(f"\nAfter transform - {name}:")
        #         print(f"Train mean: {np.mean(X_train[:, i]):.3f}")
        #         print(f"Train std: {np.std(X_train[:, i]):.3f}")
        #         print(f"Train skew: {stats.skew(X_train[:, i]):.3f}")
        
        # After features are processed but before scaling/training
        if False:  # Only export for full sequence analysis
            # Create feature names
            feature_names = []
            
            # Add metadata feature names
            if metadata_features:
                feature_names.extend(metadata_features)
                print(f"After metadata: {len(feature_names)} features")
            
            # Add global feature names
            if global_features:
                for feature_type, feature_config in global_features.items():
                    cols = feature_config['features']
                    statistics = feature_config.get('statistics', ['min', 'max', 'median', 'mean', 'std'])
                    
                    for col in cols:
                        feature_names.extend([f"{col}_{stat}" for stat in statistics])
                print(f"After global: {len(feature_names)} features")
            
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
            filename = f'{output_dir}/features_{train_set}.csv'
            feature_df.to_csv(filename, index=False)
            print(f"Exported features to: {filename}")

        # Clear data for memory management
        test_data_copy = test_data.copy()  # Keep a copy for evaluation
        train_data = None  # Clear train_data AFTER using it for feature export
        test_data = None
        #gc.collect()

        # Add before scaling
        if X_train.shape[1] == 0:
            raise ValueError(f"No features were extracted for combination: metadata={metadata_features}, global={global_features}")
        
        # Initialize scaler and imputer
        scaler = RobustScaler()
        imputer = SimpleImputer(strategy='mean')
        
        # Handle NaN values and scale data consistently
        if handle_nan == 'drop':
            # Drop rows with NaN values first
            train_valid_mask = ~np.isnan(X_train).any(axis=1)
            test_valid_mask = ~np.isnan(X_test).any(axis=1)
            
            X_train = X_train[train_valid_mask]
            y_train = y_train[train_valid_mask]
            X_test = X_test[test_valid_mask]
            y_test = y_test[test_valid_mask]
            
            # Then scale the data
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
        else:  # handle_nan == 'impute'
            # Scale first (RobustScaler can handle NaN values)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Then impute the scaled data
            X_train_scaled = imputer.fit_transform(X_train_scaled)
            X_test_scaled = imputer.transform(X_test_scaled)
        
        # Verify no NaN values remain
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            raise ValueError("NaN values remain after preprocessing")
        
        # Train and predict
        self.model.fit(X_train_scaled, y_train)
        predictions = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse, mae, r2, mse = self.evaluate_predictions(y_test, predictions, segment_size, test_data_copy)
        
        # Calculate SHAP values BEFORE clearing X_test_scaled
        if isinstance(self.model, (RandomForestRegressor, lgb.LGBMRegressor)):
            feature_names = self.get_feature_names(
                metadata_features,
                global_features
            )
            shap_results = self.calculate_shap_values(X_train_scaled, X_test_scaled, feature_names)
            
            if shap_results:
                output_dir = 'results/shap_analysis'
                os.makedirs(output_dir, exist_ok=True)
                
                # Create a unique identifier from the features
                feature_id = hash(str(metadata_features) + str(global_features))
                
                # Save feature importance plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_results['shap_values'],
                    X_test_scaled,
                    feature_names=feature_names,
                    max_display=30,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(f'{output_dir}/shap_summary_{train_set}_{feature_id}.png')
                plt.close()
                
                # Save SHAP values to CSV
                if isinstance(shap_results['shap_values'], list):
                    for i, class_shap in enumerate(shap_results['shap_values']):
                        # Reshape if necessary
                        if len(class_shap.shape) > 2:
                            class_shap = class_shap.reshape(class_shap.shape[0], -1)
                        
                        # Create column names based on actual number of features
                        num_features = class_shap.shape[1]
                        if len(feature_names) != num_features:
                            # If feature names don't match, create generic column names
                            column_names = [f'feature_{i}' for i in range(num_features)]
                        else:
                            column_names = feature_names
                        
                        shap_df = pd.DataFrame(
                            class_shap,
                            columns=column_names
                        )
                        shap_df.to_csv(f'{output_dir}/shap_values_{train_set}_{feature_id}_class_{i}.csv')
                else:
                    # Reshape if necessary
                    shap_values = shap_results['shap_values']
                    if len(shap_values.shape) > 2:
                        shap_values = shap_values.reshape(shap_values.shape[0], -1)
                    
                    # Create column names based on actual number of features
                    num_features = shap_values.shape[1]
                    if len(feature_names) != num_features:
                        # If feature names don't match, create generic column names
                        column_names = [f'feature_{i}' for i in range(num_features)]
                    else:
                        column_names = feature_names
                        
                    shap_df = pd.DataFrame(
                        shap_values,
                        columns=column_names
                    )
                    shap_df.to_csv(f'{output_dir}/shap_values_{train_set}_{feature_id}.csv')
        
        # Clean up data AFTER SHAP analysis
        X_train = None
        X_test = None
        X_train_scaled = None
        X_test_scaled = None
        test_data_copy = None
        #gc.collect()

        # Create prediction vs actual plots
        output_dir = 'figures/prediction_plots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique identifier for this experiment
        feature_id = hash(str(metadata_features) + str(global_features))
        plot_filename = f'{output_dir}/pred_vs_actual_{train_set}_{feature_id}.png'
        
        plt.figure(figsize=(10, 6))
        
        if segment_size == 'full':
            if isinstance(self, RegressionPredictor):
                # Create 2D histogram
                hist = plt.hist2d(y_test, predictions, 
                                bins=50,
                                cmap='viridis',
                                norm=colors.LogNorm(),  # logarithmic color scale
                                density=True)
                
                # Add colorbar to show density
                plt.colorbar(hist[3], label='Density of predictions')
                
                # Add diagonal reference line
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                        'r--', label='Perfect prediction')
                
                plt.xlabel('Actual Scores')
                plt.ylabel('Predicted Scores')
                plt.legend()
            else:
                # Classification confusion matrix (unchanged)
                cm = confusion_matrix(y_test, predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
        else:
            # Create results DataFrame with all segment predictions
            results_df = pd.DataFrame({
                'PID': [d['PID'] for d in test_data_copy],
                'Actual': y_test,
                'Predicted': predictions,
            })
            
            if isinstance(self, RegressionPredictor):
                # Aggregate predictions for each participant
                agg_predictions = results_df.groupby('PID').agg({
                    'Actual': 'first',
                    'Predicted': 'mean',
                    'PID': 'size'  # Count number of segments per participant
                }).reset_index(drop=True)
                
                # Create scatter plot with size based on number of segments
                plt.scatter(agg_predictions['Actual'], 
                           agg_predictions['Predicted'],
                           s=agg_predictions['PID'] * 20,  # Scale point size by number of segments
                           alpha=0.6,
                           c=agg_predictions['PID'],  # Color by number of segments
                           cmap='viridis')
                
                plt.colorbar(label='Number of segments')
                
                # Add diagonal reference line
                plt.plot([min(agg_predictions['Actual']), max(agg_predictions['Actual'])],
                        [min(agg_predictions['Actual']), max(agg_predictions['Actual'])],
                        'r--', label='Perfect prediction')
                
                plt.xlabel('Actual Scores')
                plt.ylabel('Predicted Scores')
                plt.legend()
            else:
                # For classification, we'll add segment counts to confusion matrix
                agg_predictions = results_df.groupby('PID').agg({
                    'Actual': 'first',
                    'Predicted': lambda x: x.value_counts().index[0],
                    'PID': 'size'  # Count segments per participant
                }).reset_index(drop=True)
                
                cm = confusion_matrix(agg_predictions['Actual'], agg_predictions['Predicted'],
                                     labels=sorted(set(agg_predictions['Actual']) | set(agg_predictions['Predicted'])))
                
                # Create annotation that includes both count and average segments
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                segment_counts = pd.crosstab(agg_predictions['Actual'], 
                                           agg_predictions['Predicted'], 
                                           values=agg_predictions['PID'], 
                                           aggfunc='mean')
                
                annotations = np.empty_like(cm, dtype=str)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        count = cm[i, j]
                        avg_segments = segment_counts.iloc[i, j] if count > 0 else 0
                        annotations[i, j] = f'{count}\n(avg {avg_segments:.1f} seg)'
                
                sns.heatmap(cm_norm, annot=annotations, fmt='', cmap='Blues')
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
        
        # Add title with metrics
        plt.title(f'Prediction vs Actual (RMSE: {rmse:.3f}, R²: {r2:.3f})')
        
        # Add experiment details as text
        plt.figtext(0.02, 0.02, 
                    f'Train Set: {train_set}\nSample Rate: {sample_rate}\n'
                    f'Segment Size: {segment_size}\nNaN Handling: {handle_nan}',
                    fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse,
            'handle_nan': handle_nan,
            'train_set': train_set,
            'sample_rate': sample_rate,
            'segment_size': segment_size
        }

class ClassificationPredictor(ScorePredictionBase):
    def __init__(self, config_path='experiment_config.yaml'):
        super().__init__(config_path)
        self.label_encoder = LabelEncoder()  # Add label encoder as class attribute
        
        model_config = self.config['model']
        use_spca = model_config.get('use_spca', False)
        spca_components = model_config.get('spca_components', None)
        spca_threshold = model_config.get('spca_threshold', None)
        
        base_model = self.get_model(model_config)
        
        if use_spca:
            self.model = Pipeline([
                ('spca', SupervisedPCA(
                    n_components=spca_components,
                    threshold=spca_threshold
                )),
                ('classifier', base_model)
            ])
        else:
            self.model = base_model
        
    def score_to_class(self, score):
        """Convert numerical score to class category"""
        try:
            score = int(score)
            labels = ['Non-0', '0']
            
            if score > 0:
                return labels[0]
            elif score == 0:
                return labels[1]
            else:
                raise ValueError(f"Invalid score value: {score}")
            
        except (ValueError, TypeError):
            logging.error(f"Invalid score value: {score}")
            return labels[0]
            
    def evaluate_predictions(self, y_true, predictions, segment_size, test_data=None, feature_names=None):
        """Calculate classification metrics"""
        try:
            if segment_size == 'full':
                metrics = self._calculate_classification_metrics(y_true, predictions, feature_names)
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
            
    def _calculate_classification_metrics(self, y_true, predictions, feature_names=None):
        #print("Y_true: ", y_true, "Predictions: ", predictions)
        """Helper method to calculate classification metrics"""
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
        precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_true, predictions, feature_names)
        
        return accuracy, f1, precision, recall
    
    def _plot_confusion_matrix(self, y_true, predictions, feature_names=None):
        """Create and save confusion matrix plot"""
        output_dir = 'figures/confusion_matrices'
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, predictions)
        
        # Create normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        
        # Add title with metrics
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
        plt.title(f'Confusion Matrix (Accuracy: {accuracy:.3f}, F1: {f1:.3f})')
        
        # Add experiment details
        plt.figtext(0.02, 0.02, 
                   f'Train Set: {self.train_set}', 
                   fontsize=8)
        
        # Save plot
        plt.tight_layout()
        # feature_id = hash(str(self.config.get('metadata_features')) + 
        #                  str(self.config.get('global_features')))
        feature_names = ['_'.join([i[0].lower() for i in j.split('_')[:-1]] + [j.split('_')[-1]]) for j in feature_names]
        feature_names = '='.join(feature_names)
        plt.savefig(f'{output_dir}/confusion_matrix_{self.train_set}_{feature_names}.png')
        plt.close()

    def run_single_experiment(self, handle_nan, train_set, sample_rate, segment_size, 
                            metadata_features, global_features, **kwargs):
        """Run a single experiment with given parameters"""
        # Store train_set as instance variable at the start of the method
        self.train_set = train_set
        
        # Get feature names early in the method
        feature_names = self.get_feature_names(metadata_features, global_features)
        
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
                df, pid=id_num,
                segment_size=segment_size,
                enabled_metadata=metadata_features,
                global_features=global_features,
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
                df, pid=id_num,
                segment_size=segment_size,
                enabled_metadata=metadata_features,
                global_features=global_features,
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
        
        # Keep a copy for evaluation and clear data for memory management
        test_data_copy = test_data.copy()
        train_data = None
        test_data = None
        
        # Initialize scaler and imputer
        scaler = RobustScaler()
        imputer = SimpleImputer(strategy='mean')
        
        # Handle NaN values and scale data consistently
        if handle_nan == 'drop':
            # Drop rows with NaN values first
            train_valid_mask = ~np.isnan(X_train).any(axis=1)
            test_valid_mask = ~np.isnan(X_test).any(axis=1)
            
            X_train = X_train[train_valid_mask]
            y_train = y_train[train_valid_mask]
            X_test = X_test[test_valid_mask]
            y_test = y_test[test_valid_mask]
            
            # Then scale the data
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
        else:  # handle_nan == 'impute'
            # Scale first (RobustScaler can handle NaN values)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Then impute the scaled data
            X_train_scaled = imputer.fit_transform(X_train_scaled)
            X_test_scaled = imputer.transform(X_test_scaled)
        
        # Verify no NaN values remain
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            raise ValueError("NaN values remain after preprocessing")
        
        try:
            if self.config['model'].get('use_resampling', False):
                resampling_strategy = self.config['model'].get('resampling_strategy', 'none')
                if resampling_strategy == 'smote_tomek':
                    logging.info(f"Using SMOTE-Tomek resampling for training")
                    self._resampler = SMOTETomek(random_state=42)
                    
                    # Create pipeline without scaler since data is already scaled
                    pipeline = ImbPipeline([
                        ('resampler', self._resampler),
                        ('classifier', self.model)
                    ])
                else:
                    logging.info(f"No resampling strategy specified, using standard pipeline")
                    pipeline = Pipeline([
                        ('classifier', self.model)
                    ])
            else:
                pipeline = Pipeline([
                    ('classifier', self.model)
                ])

            # Print class distribution before training
            unique, counts = np.unique(y_train, return_counts=True)
            logging.info(f"Class distribution before training: {dict(zip(unique, counts))}")
            
            # Train pipeline with already scaled data
            pipeline.fit(X_train_scaled, y_train)
            
            # If resampling was used, print new class distribution
            if self._resampler:
                _, y_resampled = pipeline.named_steps['resampler'].fit_resample(X_train_scaled, y_train)
                unique, counts = np.unique(y_resampled, return_counts=True)
                logging.info(f"Class distribution after resampling: {dict(zip(unique, counts))}")
            
            # Get the trained classifier from the pipeline
            self.model = pipeline.named_steps['classifier']
            
            # Use already scaled test data
            predictions = self.model.predict(X_test_scaled)
            
        except Exception as e:
            logging.error(f"Error in pipeline creation/training: {str(e)}")
            raise
            
        # Calculate metrics
        accuracy, f1, precision, recall = self.evaluate_predictions(
            y_test, 
            predictions, 
            segment_size, 
            test_data_copy, 
            feature_names  # Now feature_names is defined
        )
        
        # Final cleanup
        test_data_copy = None
        #gc.collect()
        
        # After model training and prediction, calculate SHAP values if applicable
        if isinstance(self.model, Pipeline):
            base_model = self.model.named_steps['classifier']
            if isinstance(base_model, (RandomForestClassifier, lgb.LGBMClassifier)):
                # First fit the SPCA transformer
                feature_selector = self.model.named_steps['spca']
                X_train_transformed = feature_selector.fit_transform(X_train_scaled, y_train)
                X_test_transformed = feature_selector.transform(X_test_scaled)
                
                # Get transformed feature names
                feature_names = np.array(self.get_feature_names(
                    metadata_features,
                    global_features
                ))[feature_selector.selected_features_]
                
                shap_results = self.calculate_shap_values(
                    X_train_transformed, 
                    X_test_transformed, 
                    feature_names
                )
                
                if shap_results:
                    output_dir = 'results/shap_analysis'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create a unique identifier from the features
                    feature_id = hash(str(metadata_features) + str(global_features))
                    
                    # Convert feature_names to a numpy array
                    feature_names = np.array(feature_names, dtype=str)
                    
                    # Save feature importance plot
                    if isinstance(shap_results['shap_values'], list):  # Multi-class
                        # Plot for each class
                        for i, class_shap in enumerate(shap_results['shap_values']):
                            plt.figure(figsize=(12, 8))
                            
                            # Reshape if necessary
                            if len(class_shap.shape) > 2:
                                class_shap = class_shap.reshape(class_shap.shape[0], -1)
                                X_test_reshaped = X_test_transformed.reshape(X_test_transformed.shape[0], -1)
                            else:
                                X_test_reshaped = X_test_transformed
                            
                            # Calculate mean absolute SHAP values for sorting
                            mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
                            feature_order = np.argsort(-mean_abs_shap)
                            
                            # Use only top 30 features
                            if len(feature_order) > 30:
                                feature_order = feature_order[:30]
                            
                            try:
                                shap.summary_plot(
                                    class_shap[:, feature_order],
                                    X_test_reshaped[:, feature_order],
                                    feature_names=feature_names[feature_order],
                                    max_display=30,
                                    show=False
                                )
                                plt.tight_layout()
                                plt.savefig(f'{output_dir}/shap_summary_{train_set}_{feature_id}_class_{i}.png')
                            except Exception as e:
                                logging.error(f"Error creating SHAP plot for class {i}: {str(e)}")
                            finally:
                                plt.close()
                    else:
                        plt.figure(figsize=(12, 8))
                        
                        # Reshape if necessary
                        if len(shap_results['shap_values'].shape) > 2:
                            shap_values = shap_results['shap_values'].reshape(
                                shap_results['shap_values'].shape[0], -1)
                            X_test_reshaped = X_test_transformed.reshape(X_test_transformed.shape[0], -1)
                        else:
                            shap_values = shap_results['shap_values']
                            X_test_reshaped = X_test_transformed
                        
                        # Calculate mean absolute SHAP values for sorting
                        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                        feature_order = np.argsort(-mean_abs_shap)
                        
                        # Use only top 30 features
                        if len(feature_order) > 30:
                            feature_order = feature_order[:30]
                        
                        try:
                            shap.summary_plot(
                                shap_values[:, feature_order],
                                X_test_reshaped[:, feature_order],
                                feature_names=feature_names[feature_order],
                                max_display=30,
                                show=False
                            )
                            plt.tight_layout()
                            plt.savefig(f'{output_dir}/shap_summary_{train_set}_{feature_id}.png')
                        except Exception as e:
                            logging.error(f"Error creating SHAP plot: {str(e)}")
                        finally:
                            plt.close()
                    
                    # Save SHAP values to CSV
                    if isinstance(shap_results['shap_values'], list):
                        for i, class_shap in enumerate(shap_results['shap_values']):
                            # Reshape if necessary
                            if len(class_shap.shape) > 2:
                                class_shap = class_shap.reshape(class_shap.shape[0], -1)
                            
                            # Create column names based on actual number of features
                            num_features = class_shap.shape[1]
                            if len(feature_names) != num_features:
                                # If feature names don't match, create generic column names
                                column_names = [f'feature_{i}' for i in range(num_features)]
                            else:
                                column_names = feature_names
                            
                            shap_df = pd.DataFrame(
                                class_shap,
                                columns=column_names
                            )
                            shap_df.to_csv(f'{output_dir}/shap_values_{train_set}_{feature_id}_class_{i}.csv')
                    else:
                        # Reshape if necessary
                        shap_values = shap_results['shap_values']
                        if len(shap_values.shape) > 2:
                            shap_values = shap_values.reshape(shap_values.shape[0], -1)
                        
                        # Create column names based on actual number of features
                        num_features = shap_values.shape[1]
                        if len(feature_names) != num_features:
                            # If feature names don't match, create generic column names
                            column_names = [f'feature_{i}' for i in range(num_features)]
                        else:
                            column_names = feature_names
                            
                        shap_df = pd.DataFrame(
                            shap_values,
                            columns=column_names
                        )
                        shap_df.to_csv(f'{output_dir}/shap_values_{train_set}_{feature_id}.csv')
        elif isinstance(self.model, (RandomForestClassifier, lgb.LGBMClassifier)):
            feature_names = self.get_feature_names(
                metadata_features,
                global_features
            )
            
            # Add check for number of features
            if len(feature_names) == 1:
                logging.warning("Skipping SHAP analysis for single feature case")
            else:
                shap_results = self.calculate_shap_values(X_train_scaled, X_test_scaled, feature_names)
                
                if shap_results:
                    output_dir = 'results/shap_analysis'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create a unique identifier from the features
                    feature_id = hash(str(metadata_features) + str(global_features))
                    
                    # Convert feature_names to a numpy array
                    feature_names = np.array(feature_names, dtype=str)
                    
                    # Save feature importance plot
                    if isinstance(shap_results['shap_values'], list):  # Multi-class
                        # Plot for each class
                        for i, class_shap in enumerate(shap_results['shap_values']):
                            plt.figure(figsize=(12, 8))
                            
                            # Reshape if necessary
                            if len(class_shap.shape) > 2:
                                class_shap = class_shap.reshape(class_shap.shape[0], -1)
                                X_test_reshaped = X_test_transformed.reshape(X_test_transformed.shape[0], -1)
                            else:
                                X_test_reshaped = X_test_transformed
                            
                            # Calculate mean absolute SHAP values for sorting
                            mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
                            feature_order = np.argsort(-mean_abs_shap)
                            
                            # Use only top 30 features
                            if len(feature_order) > 30:
                                feature_order = feature_order[:30]
                            
                            try:
                                shap.summary_plot(
                                    class_shap[:, feature_order],
                                    X_test_reshaped[:, feature_order],
                                    feature_names=feature_names[feature_order],
                                    max_display=30,
                                    show=False
                                )
                                plt.tight_layout()
                                plt.savefig(f'{output_dir}/shap_summary_{train_set}_{feature_id}_class_{i}.png')
                            except Exception as e:
                                logging.error(f"Error creating SHAP plot for class {i}: {str(e)}")
                            finally:
                                plt.close()
                    else:
                        plt.figure(figsize=(12, 8))
                        
                        # Reshape if necessary
                        if len(shap_results['shap_values'].shape) > 2:
                            shap_values = shap_results['shap_values'].reshape(
                                shap_results['shap_values'].shape[0], -1)
                            X_test_reshaped = X_test_transformed.reshape(X_test_transformed.shape[0], -1)
                        else:
                            shap_values = shap_results['shap_values']
                            X_test_reshaped = X_test_transformed
                        
                        # Calculate mean absolute SHAP values for sorting
                        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                        feature_order = np.argsort(-mean_abs_shap)
                        
                        # Use only top 30 features
                        if len(feature_order) > 30:
                            feature_order = feature_order[:30]
                        
                        try:
                            shap.summary_plot(
                                shap_values[:, feature_order],
                                X_test_reshaped[:, feature_order],
                                feature_names=feature_names[feature_order],
                                max_display=30,
                                show=False
                            )
                            plt.tight_layout()
                            plt.savefig(f'{output_dir}/shap_summary_{train_set}_{feature_id}.png')
                        except Exception as e:
                            logging.error(f"Error creating SHAP plot: {str(e)}")
                        finally:
                            plt.close()
                    
                    # Save SHAP values to CSV
                    if isinstance(shap_results['shap_values'], list):
                        for i, class_shap in enumerate(shap_results['shap_values']):
                            # Reshape if necessary
                            if len(class_shap.shape) > 2:
                                class_shap = class_shap.reshape(class_shap.shape[0], -1)
                            
                            # Create column names based on actual number of features
                            num_features = class_shap.shape[1]
                            if len(feature_names) != num_features:
                                # If feature names don't match, create generic column names
                                column_names = [f'feature_{i}' for i in range(num_features)]
                            else:
                                column_names = feature_names
                            
                            shap_df = pd.DataFrame(
                                class_shap,
                                columns=column_names
                            )
                            shap_df.to_csv(f'{output_dir}/shap_values_{train_set}_{feature_id}_class_{i}.csv')
                    else:
                        # Reshape if necessary
                        shap_values = shap_results['shap_values']
                        if len(shap_values.shape) > 2:
                            shap_values = shap_values.reshape(shap_values.shape[0], -1)
                        
                        # Create column names based on actual number of features
                        num_features = shap_values.shape[1]
                        if len(feature_names) != num_features:
                            # If feature names don't match, create generic column names
                            column_names = [f'feature_{i}' for i in range(num_features)]
                        else:
                            column_names = feature_names
                            
                        shap_df = pd.DataFrame(
                            shap_values,
                            columns=column_names
                        )
                        shap_df.to_csv(f'{output_dir}/shap_values_{train_set}_{feature_id}.csv')
        
        # Create prediction vs actual plots
        output_dir = 'figures/prediction_plots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique identifier for this experiment
        feature_id = hash(str(metadata_features) + str(global_features))
        
        plot_filename = f'{output_dir}/pred_vs_actual_{train_set}_{feature_id}.png'
        
        plt.figure(figsize=(10, 6))
        
        if segment_size == 'full':
            if isinstance(self, RegressionPredictor):
                # Create 2D histogram
                hist = plt.hist2d(y_test, predictions, 
                                bins=50,
                                cmap='viridis',
                                norm=colors.LogNorm(),  # logarithmic color scale
                                density=True)
                
                # Add colorbar to show density
                plt.colorbar(hist[3], label='Density of predictions')
                
                # Add diagonal reference line
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                        'r--', label='Perfect prediction')
                
                plt.xlabel('Actual Scores')
                plt.ylabel('Predicted Scores')
                plt.legend()
            else:
                # Classification confusion matrix (unchanged)
                cm = confusion_matrix(y_test, predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
        else:
            # Create results DataFrame with all segment predictions
            results_df = pd.DataFrame({
                'PID': [d['PID'] for d in test_data_copy],
                'Actual': y_test,
                'Predicted': predictions,
            })
            
            if isinstance(self, RegressionPredictor):
                # Aggregate predictions for each participant
                agg_predictions = results_df.groupby('PID').agg({
                    'Actual': 'first',
                    'Predicted': 'mean',
                    'PID': 'size'  # Count number of segments per participant
                }).reset_index(drop=True)
                
                # Create scatter plot with size based on number of segments
                plt.scatter(agg_predictions['Actual'], 
                           agg_predictions['Predicted'],
                           s=agg_predictions['PID'] * 20,  # Scale point size by number of segments
                           alpha=0.6,
                           c=agg_predictions['PID'],  # Color by number of segments
                           cmap='viridis')
                
                plt.colorbar(label='Number of segments')
                
                # Add diagonal reference line
                plt.plot([min(agg_predictions['Actual']), max(agg_predictions['Actual'])],
                        [min(agg_predictions['Actual']), max(agg_predictions['Actual'])],
                        'r--', label='Perfect prediction')
                
                plt.xlabel('Actual Scores')
                plt.ylabel('Predicted Scores')
                plt.legend()
            else:
                # For classification, we'll add segment counts to confusion matrix
                agg_predictions = results_df.groupby('PID').agg({
                    'Actual': 'first',
                    'Predicted': lambda x: x.value_counts().index[0],
                    'PID': 'size'  # Count segments per participant
                }).reset_index(drop=True)
                
                cm = confusion_matrix(agg_predictions['Actual'], agg_predictions['Predicted'],
                                     labels=sorted(set(agg_predictions['Actual']) | set(agg_predictions['Predicted'])))
                
                # Create annotation that includes both count and average segments
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                segment_counts = pd.crosstab(agg_predictions['Actual'], 
                                           agg_predictions['Predicted'], 
                                           values=agg_predictions['PID'], 
                                           aggfunc='mean')
                
                annotations = np.empty_like(cm, dtype=str)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        count = cm[i, j]
                        avg_segments = segment_counts.iloc[i, j] if count > 0 else 0
                        annotations[i, j] = f'{count}\n(avg {avg_segments:.1f} seg)'
                
                sns.heatmap(cm_norm, annot=annotations, fmt='', cmap='Blues')
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
        
        # Add title with metrics
        plt.title(f'Prediction vs Actual (Accuracy: {accuracy:.3f}, F1: {f1:.3f})')
        
        # Add experiment details as text
        plt.figtext(0.02, 0.02, 
                    f'Train Set: {train_set}\nSample Rate: {sample_rate}\n'
                    f'Segment Size: {segment_size}\nNaN Handling: {handle_nan}',
                    fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'handle_nan': handle_nan,
            'train_set': train_set,
            'sample_rate': sample_rate,
            'segment_size': segment_size
        }
    
    def run_nested_cv_experiment(self, handle_nan, train_set, sample_rate, segment_size, 
                                metadata_features, global_features, **kwargs):
        """Run experiment with nested stratified 5-fold cross validation by participant,
        with the outer loop parallelized. In each outer fold, the entire outer test set
        (i.e. multiple participants) is evaluated together.
        """
        # Store train_set as instance variable
        self.train_set = train_set

        # Get feature names
        feature_names = self.get_feature_names(metadata_features, global_features)

        # Get parameter grid from config
        model_type = self.config['model']['type']
        param_grid = self.config['model'].get('param_grids', {}).get(model_type, {})

        if not param_grid:
            logging.warning(f"No parameter grid found for model type {model_type}. Using default parameters.")

        # Determine VR (training) and IRL (test) datasets
        if train_set == 'A':  # If A is VR data
            train_data_dict = self._data_by_id_A
            test_data_dict = self._data_by_id_B  # B is IRL data
            train_score_col = 'Score_A'
            test_score_col = 'Score_B'
        else:  # If B is VR data
            train_data_dict = self._data_by_id_B
            test_data_dict = self._data_by_id_A  # A is IRL data
            train_score_col = 'Score_B'
            test_score_col = 'Score_A'

        # Initialize a dictionary to collect overall metrics
        cv_metrics = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'confusion_matrices': [],
            'selected_features': [],
            'best_params': []
        }

        # --- Outer Loop Setup: Stratified 5-Fold CV on Participants ---
        # Get the unique participant IDs from the IRL (test) dataset and build labels for stratification.
        test_pids = list(test_data_dict.keys())
        participant_labels = []
        for pid in test_pids:
            # Assume that the label for a participant is obtained from the first row of the participant's data.
            df = test_data_dict[pid][0]['data']
            label = self.score_to_class(df[test_score_col].iloc[0])
            participant_labels.append(label)

        outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        
        # --- Define a Helper Function for an Entire Outer Fold ---
        def process_outer_fold(outer_test_pids):
            """
            Process one outer fold where outer_test_pids is the set of participant IDs held out as test.
            In this version, the test data from all these participants is aggregated together.
            """
            logging.info(f"\nProcessing outer fold with test participants: {outer_test_pids}")

            # Prepare VR training data: exclude any participant in the outer test set.
            train_features = []
            train_scores = []
            train_pids = []

            for pid, data in train_data_dict.items():
                # if pid in outer_test_pids:
                #     continue  # Exclude VR data for all outer test participants.
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

            print(f"Training on {len(X_train_full)} segments from {len(train_pids)} participants.")

            # Prepare IRL (test) data for the entire outer test set by aggregating data from all test participants.
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
                # Duplicate the label for each segment.
                label = self.score_to_class(test_df[test_score_col].iloc[0])
                test_features.extend(test_segments)
                test_scores.extend([label] * len(test_segments))

            X_test = np.array(test_features)
            y_test = np.array(test_scores)

            print(f"Testing on {len(X_test)} segments from {len(test_pids)} participants.")

            # Preprocess data.
            scaler = RobustScaler()
            imputer = SimpleImputer(strategy='mean')

            if handle_nan == 'drop':
                train_valid_mask = ~np.isnan(X_train_full).any(axis=1)
                test_valid_mask = ~np.isnan(X_test).any(axis=1)
                X_train_full = X_train_full[train_valid_mask]
                y_train_full = y_train_full[train_valid_mask]
                train_pids = train_pids[train_valid_mask]
                X_test = X_test[test_valid_mask]
                y_test = y_test[test_valid_mask]

            X_train_scaled = scaler.fit_transform(X_train_full)
            X_test_scaled = scaler.transform(X_test)
            if handle_nan != 'drop':
                X_train_scaled = imputer.fit_transform(X_train_scaled)
                X_test_scaled = imputer.transform(X_test_scaled)

            # print(X_train_scaled, y_train_full)

            # --- Inner Loop: Feature Selection & Hyperparameter Tuning ---
            # Use StratifiedKFold for the inner loop.
            inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            base_classifier = self.get_model(self.config['model'])
            k_features_to_select = (10, 15)

            # def balanced_accuracy(y_true, y_pred):
            #     C = confusion_matrix(y_true, y_pred, labels=['Non-0', '0'])
            #     with np.errstate(divide="ignore", invalid="ignore"):
            #         per_class = np.diag(C) / C.sum(axis=1)
            #     if np.any(np.isnan(per_class)):
            #         per_class = per_class[~np.isnan(per_class)]
            #     score = np.mean(per_class)
            #     return score 

            # # Define a custom scorer that handles the single-class case.
            # def balanced_accuracy_with_single_class(y_true, y_pred):
            #     unique_true = np.unique(y_true)
            #     if len(unique_true) == 1:
            #         return 1.0 if np.array_equal(y_true, y_pred) else 0.0
            #     y_pred_clean = np.where(y_pred == 'Non-0', 'Non-0', '0')
            #     return balanced_accuracy(y_true, y_pred_clean)

            def specificity_score(y_true, y_pred):
                #return recall_score(y_true, y_pred, average='binary', pos_label=0)
                return balanced_accuracy_score(y_true, y_pred)

            # After preparing y_train_full and before model training
            y_train_full_encoded = self.label_encoder.fit_transform(y_train_full)
            y_test_encoded = self.label_encoder.transform(y_test)

            # Modify the SFS to use encoded labels
            sfs = SequentialFeatureSelector(
                base_classifier,
                k_features=k_features_to_select,
                forward=True,
                scoring=make_scorer(specificity_score),
                floating=True,
                cv=inner_cv,
                n_jobs=-1,
                verbose=0
            )
            sfs.fit(X_train_scaled, y_train_full_encoded)  # Use encoded labels

            selected_feature_indices = sfs.k_feature_idx_  # The tuple of selected feature indices.
            selected_features = [feature_names[i] for i in selected_feature_indices]
            logging.info(f"Selected features for outer fold with test participants {outer_test_pids}: {selected_features}")

            # Apply feature selection.
            X_train_selected = X_train_scaled[:, selected_feature_indices]
            X_test_selected = X_test_scaled[:, selected_feature_indices]

            # Hyperparameter tuning.
            if param_grid:
                logging.info("Starting hyperparameter tuning...")
                grid_search = GridSearchCV(
                    estimator=self.get_model(self.config['model']),
                    param_grid=param_grid,
                    scoring=make_scorer(specificity_score),
                    cv=inner_cv,
                    n_jobs=-1
                )
                grid_search.fit(X_train_selected, y_train_full_encoded)
                best_params = grid_search.best_params_
                final_model = grid_search.best_estimator_
            else:
                final_model = self.get_model(self.config['model'])
                best_params = {}

            # Optionally wrap with a resampling pipeline if needed.
            if self.config['model'].get('use_resampling', False):
                resampling_strategy = self.config['model'].get('resampling_strategy', 'none')
                if resampling_strategy == 'smote_tomek':
                    self._resampler = SMOTETomek(random_state=42)
                    pipeline = ImbPipeline([
                        ('resampler', self._resampler),
                        ('classifier', final_model)
                    ])
                else:
                    pipeline = Pipeline([('classifier', final_model)])
            else:
                pipeline = Pipeline([('classifier', final_model)])

            # Train final model on the full VR training data and evaluate on the aggregated IRL test data.
            pipeline.fit(X_train_selected, y_train_full_encoded)
            test_predictions = pipeline.predict(X_test_selected)

            # After getting predictions, decode them back to original labels
            test_predictions = self.label_encoder.inverse_transform(test_predictions)

            # Calculate metrics using original string labels
            fold_accuracy = accuracy_score(y_test, test_predictions)
            fold_balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)
            fold_f1 = f1_score(y_test, test_predictions, average='weighted', zero_division=0)
            fold_precision = precision_score(y_test, test_predictions, average='weighted', zero_division=0)
            fold_recall = recall_score(y_test, test_predictions, average='weighted', zero_division=0)
            fold_cm = confusion_matrix(y_test, test_predictions, labels=['Non-0', '0'])

            # Package the results for this outer fold.
            fold_results = {
                'accuracy': fold_accuracy,
                'balanced_accuracy': fold_balanced_accuracy,
                'f1': fold_f1,
                'precision': fold_precision,
                'recall': fold_recall,
                'confusion_matrix': fold_cm,
                'selected_features': selected_features,
                'best_params': best_params,
                # Optionally, store y_true and y_pred for further analysis.
                'y_true': y_test,
                'y_pred': test_predictions
            }
            return fold_results

    # --- End of process_outer_fold definition ---

        # Use StratifiedKFold on the participant level for the outer loop.
        outer_fold_start_time = time.time()
        outer_results = []
        for train_idx, test_idx in outer_cv.split(test_pids, participant_labels):
            # The outer test set for this fold:
            outer_test_pids = [test_pids[i] for i in test_idx]
            logging.info(f"\nProcessing outer fold with test participants: {outer_test_pids}")
            inner_fold_start_time = time.time()
            fold_results = process_outer_fold(set(outer_test_pids))
            outer_results.append(fold_results)
            inner_fold_end_time = time.time()
            print(f"Inner fold time: {inner_fold_end_time - inner_fold_start_time} seconds\n")

        outer_fold_end_time = time.time()
        print(f"Total time for outer folds: {outer_fold_end_time - outer_fold_start_time} seconds")

        # Aggregate metrics from all outer folds.
        for result in [res for res in outer_results if res is not None]:
            cv_metrics['accuracy'].append(result['accuracy'])
            cv_metrics['balanced_accuracy'].append(result['balanced_accuracy'])
            cv_metrics['f1'].append(result['f1'])
            cv_metrics['precision'].append(result['precision'])
            cv_metrics['recall'].append(result['recall'])
            cv_metrics['confusion_matrices'].append(result['confusion_matrix'])
            cv_metrics['selected_features'].append(result['selected_features'])
            cv_metrics['best_params'].append(result['best_params'])

        # Analyze feature selection and hyperparameter results (as in your original code)
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

        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

        logging.info("\nFeature selection summary:")
        for feature, count in sorted_features:
            logging.info(f"{feature}: selected in {count}/{len(test_pids)} folds")

        logging.info("\nHyperparameter tuning summary:")
        for param, values in param_counts.items():
            logging.info(f"\n{param}:")
            for value, count in sorted(values.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {value}: selected in {count}/{len(test_pids)} folds")

        # Example: Calculate average metrics.
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
        output_dir = 'figures/confusion_matrices'
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

        print("avg_cm: ", avg_cm)
        print("std_cm: ", std_cm)

        # Define the labels
        labels = ['Non-0', '0']

        # Create annotations that include average percentage and std
        annotations = np.empty(avg_cm.shape, dtype='<U20')
        for i in range(avg_cm.shape[0]):
            for j in range(avg_cm.shape[1]):
                percentage = avg_cm[i, j]
                std = std_cm[i, j]
                annotations[i, j] = f'{percentage:.2%} (±{std:.2%})'

        # Create heatmap with combined annotations
        sns.heatmap(avg_cm, 
                   annot=annotations,
                   fmt='s',
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   vmin=0, vmax=1)  # Set fixed scale from 0 to 1

        plt.title(f'Average Confusion Matrix\nBalanced Accuracy: {avg_metrics["balanced_accuracy"]:.3f} (±{avg_metrics["balanced_accuracy_std"]:.3f})')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.savefig(f'{output_dir}/average_confusion_matrix_{train_set}_{model_type}_cv.png')
        plt.close()

        # Now create a new figure for per-fold confusion matrices
        n_folds = len(cv_metrics['confusion_matrices'])
        n_cols = 5  # Number of columns in the subplot grid
        n_rows = (n_folds + n_cols - 1) // n_cols  # Calculate required number of rows

        plt.figure(figsize=(4*n_cols, 4*n_rows))
        plt.suptitle(f'Per-Fold Confusion Matrices - {train_set} Set', fontsize=16, y=1.02)

        for i, cm in enumerate(cv_metrics['confusion_matrices']):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Normalize the confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create annotations with counts and percentages
            annotations = np.empty_like(cm_norm, dtype='<U20')
            for row in range(cm_norm.shape[0]):
                for col in range(cm_norm.shape[1]):
                    count = cm[row, col]
                    percentage = cm_norm[row, col]
                    annotations[row, col] = f'{count} ({percentage:.2%})'
            
            # Create heatmap
            sns.heatmap(cm_norm,
                        annot=annotations,
                        fmt='s',
                        cmap='Blues',
                        xticklabels=labels,
                        yticklabels=labels,
                        cbar=False)  # Remove colorbar for cleaner subplots
            
            plt.title(f'Fold {i+1}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/per_fold_confusion_matrices_{train_set}_{model_type}_cv.png', 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()

        logging.info("\nCross-validation results:")
        logging.info(f"Average Accuracy: {avg_metrics['accuracy']:.3f} (±{avg_metrics['accuracy_std']:.3f})")
        logging.info(f"Average Balanced Accuracy: {avg_metrics['balanced_accuracy']:.3f} (±{avg_metrics['balanced_accuracy_std']:.3f})")
        logging.info(f"Average F1: {avg_metrics['f1']:.3f} (±{avg_metrics['f1_std']:.3f})")
        logging.info(f"Average Precision: {avg_metrics['precision']:.3f} (±{avg_metrics['precision_std']:.3f})")
        logging.info(f"Average Recall: {avg_metrics['recall']:.3f} (±{avg_metrics['recall_std']:.3f})")

        return avg_metrics
