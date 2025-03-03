import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import os, logging, shap
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from .base import ScorePredictionBase
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, confusion_matrix)

class RegressionPredictor(ScorePredictionBase):
    def __init__(self):
        self.model = self.get_model(self.model_config['model'])
        self.predictor_type = 'regression'
        
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
        
        # After features are processed but before scaling/training
        if self.parameter_config['export_feature_table']:  # Only export for full sequence analysis
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