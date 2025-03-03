import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.stats as stats
import pyarrow.parquet as pq
import os, shap, time, yaml, logging
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNRegressor, TabPFNClassifier
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def time_method(phrase):
    """Decorator that times a method and prints the time with a custom phrase.
    
    Args:
        phrase (str): The phrase to print before the timing result
        
    Returns:
        function: The decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{phrase}: {end_time - start_time:.2f} seconds")
            return result
        return wrapper
    return decorator

def load_config(config_path):
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def write_config(config_path, config):
    """Write experiment configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

class ScorePredictionBase:
    _data_by_id_A = None  # Class-level cache
    _data_by_id_B = None
    _metadata = None
    _is_data_loaded = False  # Add flag to track if data is loaded
    _scaler = None  # Add class-level scaler
    _imputer = None  # Add class-level imputer
    _feature_cache = {}  # New class-level feature cache
    parameter_config_path = '/srv/STP/parameter_config.yaml'
    feature_config_path = '/srv/STP/feature_config.yaml'
    model_config_path = '/srv/STP/model_config.yaml'
    
    def __init__(self):
        self.parameter_config = load_config(self.parameter_config_path)
        self.feature_config = load_config(self.feature_config_path)
        self.model_config = load_config(self.model_config_path)
        self.ensure_data_loaded()
        if self._scaler is None:
            self._scaler = RobustScaler()
        if self._imputer is None:
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
        directory = f'./data/FAB/FAB_{set_type}_{load_config(cls.parameter_config_path)["dataset"]}'
        
        # Get list of parquet files from directory
        files = sorted([f for f in os.listdir(directory) if f.endswith('.parquet')])
        # print(f"Found {len(files)} parquet files in {directory}")
        
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
            
        return data_dict
    
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
                        # Safely get the value, replacing inf and -inf with nan
                        feature_value = np.nan_to_num(df[meta_feature].iloc[0], nan=np.nan)
                        self._feature_cache[cache_key] = feature_value
                    else:
                        feature_value = np.nan

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
                    
                    # Define available statistics with safe handling of invalid values
                    stat_functions = {
                        'min': lambda x: np.nan_to_num(x.min(), nan=np.nan),
                        'max': lambda x: np.nan_to_num(x.max(), nan=np.nan),
                        'median': lambda x: np.nan_to_num(x.median(), nan=np.nan),
                        'mean': lambda x: np.nan_to_num(x.mean(), nan=np.nan),
                        'std': lambda x: np.nan_to_num(x.std(), nan=np.nan)
                    }
                    
                    # Calculate statistics for each column while maintaining order
                    for col in cols:
                        col_data = df[col]
                        
                        # Replace inf values with nan before calculations
                        col_data = col_data.replace([np.inf, -np.inf], np.nan)
                        
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
        
        # Convert features to numpy array with explicit dtype and handle invalid values
        features = np.array(features, dtype=np.float64)
        features = np.nan_to_num(features, nan=np.nan, posinf=np.nan, neginf=np.nan)
        
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
        
        model_class = MODEL_REGISTRY[self.predictor_type].get(model_type)
        
        if model_class is None:
            valid_types = list(MODEL_REGISTRY[self.predictor_type].keys())
            raise ValueError(f"Unknown model type: {model_type}. Valid types are: {valid_types}")
        
        try:
            base_model = model_class(**params)
            return base_model
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
    
    def get_train_test_data(self, train_set):
        if self.parameter_config['cross_task']:
            # Cross-task: train on VR data, test on IRL data.
            if train_set == 'A':  # If A is VR data
                train_data_dict = self._data_by_id_A
                test_data_dict = self._data_by_id_B  # B is IRL data
                train_score_col = f'Score_A{self.parameter_config["score_type"]}'
                test_score_col = f'Score_B{self.parameter_config["score_type"]}'
            else:  # If B is VR data
                train_data_dict = self._data_by_id_B
                test_data_dict = self._data_by_id_A  # A is IRL data
                train_score_col = f'Score_B{self.parameter_config["score_type"]}'
                test_score_col = f'Score_A{self.parameter_config["score_type"]}'
        else:
            # Within-task: use the same dataset for both training and testing (e.g., Set A).
            if train_set == 'A':
                train_data_dict = self._data_by_id_A
                test_data_dict = self._data_by_id_A
                train_score_col = f'Score_A{self.parameter_config["score_type"]}'
                test_score_col = f'Score_A{self.parameter_config["score_type"]}'
            else:
                train_data_dict = self._data_by_id_B
                test_data_dict = self._data_by_id_B
                train_score_col = f'Score_B{self.parameter_config["score_type"]}'
                test_score_col = f'Score_B{self.parameter_config["score_type"]}'
        
        return train_data_dict, test_data_dict, train_score_col, test_score_col