import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from motion_features import calculate_motion_derivatives
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import yaml

def find_segments(accelerations, threshold):
    """
    Find segments where acceleration magnitude is below threshold.
    
    Args:
        accelerations: numpy array of shape (n, 3) containing xyz accelerations
        threshold: acceleration magnitude threshold for segmentation
    
    Returns:
        List of tuples containing (start_idx, end_idx) for each segment
    """
    acc_magnitude = np.linalg.norm(accelerations, axis=1)
    below_threshold = acc_magnitude >= threshold
    
    # Find where the signal changes from above to below threshold or vice versa
    changes = np.diff(below_threshold.astype(int))
    
    # Get indices where segments start and end
    segment_starts = np.where(changes == 1)[0] + 1
    segment_ends = np.where(changes == -1)[0] + 1
    
    # Handle edge cases
    if below_threshold[0]:
        segment_starts = np.insert(segment_starts, 0, 0)
    if below_threshold[-1]:
        segment_ends = np.append(segment_ends, len(below_threshold))
    
    return list(zip(segment_starts, segment_ends))

def find_segments_combined(linear_acc, angular_acc, linear_threshold, angular_threshold, mode='either'):
    """
    Find segments based on both linear and angular acceleration thresholds.
    
    Args:
        linear_acc: numpy array of shape (n, 3) containing xyz linear accelerations
        angular_acc: numpy array of shape (n, 3) containing xyz angular accelerations
        linear_threshold: linear acceleration magnitude threshold
        angular_threshold: angular acceleration magnitude threshold
        mode: string indicating how to combine thresholds:
            'both' - both accelerations must be above threshold
            'either' - either acceleration above threshold
            'both_below' - both accelerations must be below threshold
    
    Returns:
        List of tuples containing (start_idx, end_idx) for each segment
    """
    linear_magnitude = np.linalg.norm(linear_acc, axis=1)
    angular_magnitude = np.linalg.norm(angular_acc, axis=1)
    
    linear_above = linear_magnitude >= linear_threshold
    angular_above = angular_magnitude >= angular_threshold
    
    if mode == 'both':
        combined = linear_above & angular_above
    elif mode == 'linear':
        combined = linear_above
    elif mode == 'angular':
        combined = angular_above
    elif mode == 'either':
        combined = linear_above | angular_above
    elif mode == 'both_below':
        combined = ~(linear_above | angular_above)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Find where the signal changes
    changes = np.diff(combined.astype(int))
    
    # Get indices where segments start and end
    segment_starts = np.where(changes == 1)[0] + 1
    segment_ends = np.where(changes == -1)[0] + 1
    
    # Handle edge cases
    if combined[0]:
        segment_starts = np.insert(segment_starts, 0, 0)
    if combined[-1]:
        segment_ends = np.append(segment_ends, len(combined))
    
    return list(zip(segment_starts, segment_ends))

def calculate_segment_features(positions, velocities, accelerations, segments, df, config, min_duration=1):
    """
    Calculate features for each motion segment using configuration-specified features.
    
    Args:
        positions: numpy array of shape (n, 3) containing xyz positions
        velocities: numpy array of shape (n, 3) containing xyz velocities
        accelerations: numpy array of shape (n, 3) containing xyz accelerations
        segments: list of (start_idx, end_idx) tuples
        df: pandas DataFrame containing all motion data
        config: configuration dictionary containing feature specifications
        min_duration: minimum number of frames for a valid segment
    
    Returns:
        List of dictionaries containing segment features
    """
    segment_features = []
    
    # Define available statistics functions
    stat_functions = {
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'mean': np.mean,
        'std': np.std
    }
    
    for start_idx, end_idx in segments:
        # Skip segments that are too short
        if (end_idx - start_idx) < min_duration:
            continue
        
        features = {
            'start_frame': start_idx,
            'end_frame': end_idx,
            'duration_frames': end_idx - start_idx,
        }
        
        # Calculate configured features
        for feature_name, feature_config in config.get('segment_features', {}).items():
            feature_cols = feature_config['features']
            statistics = feature_config.get('statistics', ['mean'])
            
            for col in feature_cols:
                if col not in df.columns:
                    continue
                    
                segment_data = df[col].iloc[start_idx:end_idx]
                
                for stat in statistics:
                    if stat not in stat_functions:
                        continue
                        
                    stat_value = stat_functions[stat](segment_data)
                    features[f'{col}_{stat}'] = stat_value
        
        segment_features.append(features)
    
    return segment_features

def segment_motion_file(input_path, output_path, config, linear_threshold=0.5, 
                       angular_threshold=1.0, min_duration=10, mode='either'):
    """
    Segment motion data based on both linear and angular acceleration thresholds.
    
    Args:
        input_path: path to input CSV file
        output_path: path to output CSV file
        config: configuration dictionary containing feature specifications
        linear_threshold: linear acceleration magnitude threshold (m/s^2)
        angular_threshold: angular acceleration magnitude threshold (rad/s^2)
        min_duration: minimum number of frames for a valid segment
        mode: how to combine thresholds ('both', 'either', or 'both_below')
    """
    # Read input file
    df = pd.read_csv(input_path)
    
    # Calculate dt from timestamp
    dt = df['Timestamp'].diff().mean()
    
    # Process each tracked object
    objects = ['Head', 'LeftHand', 'RightHand']
    all_segments = {}
    
    for obj in objects:
        # Get position and quaternion data
        pos_cols = [f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']
        quat_cols = [f'{obj}_quat_w', f'{obj}_quat_x', f'{obj}_quat_y', f'{obj}_quat_z']
        
        pos = df[pos_cols].to_numpy()
        quat = df[quat_cols].to_numpy()
        
        # Calculate motion derivatives (now using angular acceleration)
        vel, acc, _, _, ang_acc, _ = calculate_motion_derivatives(pos, quat, dt)
        
        # Find segments using both linear and angular acceleration
        segments = find_segments_combined(acc, ang_acc, linear_threshold, 
                                        angular_threshold, mode)
        
        # Calculate features for each segment
        segment_features = calculate_segment_features(
            pos, vel, acc, segments, 
            df=df,  # Pass the full DataFrame
            config=config,  # Pass the configuration
            min_duration=min_duration
        )
        
        # Store results
        all_segments[obj] = segment_features
    
    # Convert to DataFrame format
    segment_rows = []
    for obj, segments in all_segments.items():
        for i, segment in enumerate(segments):
            row = {
                'object': obj,
                'segment_id': i,
                **segment
            }
            segment_rows.append(row)
    
    # Save results
    df_segments = pd.DataFrame(segment_rows)
    df_segments.to_csv(output_path, index=False)
    return df_segments

def process_file(filename, input_dir, output_dir, config, linear_threshold, angular_threshold, min_duration, mode):
    """Process a single file with motion segmentation."""
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename.replace('.csv', '_segments.csv'))
    try:
        segment_motion_file(input_path, output_path, config, linear_threshold, angular_threshold, min_duration, mode)
        return True
    except Exception as e:
        print(f"\nError processing {filename}: {str(e)}")
        return False

if __name__ == "__main__":
    # Load configuration
    config_path = 'segment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update parameters to include both thresholds
    input_dir = "data/FAB/FAB_B_Modified_Motion"
    output_dir = "data/FAB/FAB_B_Motion_Segments"
    linear_threshold = 0.5  # m/s^2
    angular_threshold = 1.0  # rad/s^2
    min_duration = 30    # frames
    mode = 'linear'     # 'both', 'linear', 'angular', 'either', or 'both_below'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    # Set up multiprocessing
    num_cores = mp.cpu_count() - 1
    print(f"Processing {total_files} files using {num_cores} cores...")
    
    # Create partial function with fixed arguments
    process_file_partial = partial(
        process_file,
        input_dir=input_dir,
        output_dir=output_dir,
        config=config,
        linear_threshold=linear_threshold,
        angular_threshold=angular_threshold,
        min_duration=min_duration,
        mode=mode
    )
    
    # Process files in parallel with progress bar
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_file_partial, csv_files),
            total=total_files,
            desc="Processing files",
            unit="file"
        ))
    
    # Print summary
    successful = sum(results)
    print(f"\nProcessing complete: {successful}/{total_files} files processed successfully") 