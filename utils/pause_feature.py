import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from scipy.signal import savgol_filter
from motion_features import calculate_motion_derivatives
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def calculate_velocities(positions, quaternions, dt=1/30):
    """
    Calculate linear and angular velocities from positions and quaternions.
    
    Args:
        positions: numpy array of shape (n, 3) containing xyz positions
        quaternions: numpy array of shape (n, 4) containing quaternions (w,x,y,z)
        dt: time step between frames (default: 1/30 second)
    """
    # Add small epsilon to avoid division by zero
    dt = max(dt, 1e-6)
    
    # Handle NaN values in input data
    if np.any(np.isnan(positions)) or np.any(np.isnan(quaternions)):
        raise ValueError("Input data contains NaN values")
    
    # Calculate linear velocity using central differences
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    
    # Calculate angular velocity from quaternions
    angular_vel = np.zeros((len(quaternions), 3))
    for i in range(1, len(quaternions)):
        q1 = quaternions[i-1]
        q2 = quaternions[i]
        
        # Convert to rotation objects
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        
        # Calculate relative rotation
        r_diff = r2 * r1.inv()
        
        # Convert to axis-angle representation
        rotvec = r_diff.as_rotvec()
        angular_vel[i] = rotvec / dt
    
    # Optional: Apply smoothing to reduce noise
    velocities = savgol_filter(velocities, window_length=5, polyorder=2, axis=0)
    angular_vel = savgol_filter(angular_vel, window_length=5, polyorder=2, axis=0)
    
    return velocities, angular_vel

def detect_pauses(velocities, accelerations, jerks, angular_velocities, angular_accelerations, angular_jerks, percentile=25):
    """
    Detect periods where motion derivatives are below thresholds determined by the data distribution.
    
    Args:
        velocities: numpy array of shape (n, 3) containing xyz velocities
        accelerations: numpy array of shape (n, 3) containing xyz accelerations
        jerks: numpy array of shape (n, 3) containing xyz jerks
        angular_velocities: numpy array of shape (n, 3) containing angular velocities
        angular_accelerations: numpy array of shape (n, 3) containing angular accelerations
        angular_jerks: numpy array of shape (n, 3) containing angular jerks
        percentile: percentile below which motion is considered 'paused' (default: 25)
    
    Returns:
        Dictionary containing pause signals for each motion derivative
    """
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100")
    
    # Calculate magnitudes
    linear_vel_mag = np.linalg.norm(velocities, axis=1)
    linear_acc_mag = np.linalg.norm(accelerations, axis=1)
    linear_jerk_mag = np.linalg.norm(jerks, axis=1)
    angular_vel_mag = np.linalg.norm(angular_velocities, axis=1)
    angular_acc_mag = np.linalg.norm(angular_accelerations, axis=1)
    angular_jerk_mag = np.linalg.norm(angular_jerks, axis=1)
    
    # Calculate thresholds
    linear_vel_threshold = np.percentile(linear_vel_mag, percentile)
    linear_acc_threshold = np.percentile(linear_acc_mag, percentile)
    linear_jerk_threshold = np.percentile(linear_jerk_mag, percentile)
    angular_vel_threshold = np.percentile(angular_vel_mag, percentile)
    angular_acc_threshold = np.percentile(angular_acc_mag, percentile)
    angular_jerk_threshold = np.percentile(angular_jerk_mag, percentile)
    
    # Detect pauses for each derivative
    return {
        'velocity': (linear_vel_mag < linear_vel_threshold).astype(int),
        'acceleration': (linear_acc_mag < linear_acc_threshold).astype(int),
        'jerk': (linear_jerk_mag < linear_jerk_threshold).astype(int),
        'angular_velocity': (angular_vel_mag < angular_vel_threshold).astype(int),
        'angular_acceleration': (angular_acc_mag < angular_acc_threshold).astype(int),
        'angular_jerk': (angular_jerk_mag < angular_jerk_threshold).astype(int)
    }

def calculate_pause_features(binary_pauses, dt=1/30):
    """
    Convert binary pause signals into more meaningful features.
    
    Args:
        binary_pauses: numpy array of shape (n,) with 1s indicating pauses
        dt: time step between frames (in seconds)
    
    Returns:
        Dictionary containing:
        - pause_duration: How long the current pause has lasted (0 if not paused)
        - time_since_pause: Time since last pause ended (0 if currently paused)
    
    Raises:
        ValueError: If binary_pauses contains values other than 0 and 1
        ValueError: If dt is <= 0
    """
    if not np.all(np.isin(binary_pauses, [0, 1])):
        raise ValueError("binary_pauses must contain only 0s and 1s")
    
    if dt <= 0:
        raise ValueError("dt must be positive")
    
    # Initialize output arrays
    n_samples = len(binary_pauses)
    pause_duration = np.zeros(n_samples)
    time_since_pause = np.zeros(n_samples)
    
    # Calculate pause durations
    current_duration = 0
    for i in range(n_samples):
        if binary_pauses[i]:
            current_duration += dt
            pause_duration[i] = current_duration
        else:
            current_duration = 0
    
    # Calculate time since last pause
    last_pause_end = 0
    current_time = 0
    for i in range(n_samples):
        current_time = i * dt
        if binary_pauses[i]:
            time_since_pause[i] = 0
            last_pause_end = current_time + dt  # Add dt to mark the end of this pause
        else:
            time_since_pause[i] = current_time - last_pause_end
    
    return {
        'pause_duration': pause_duration,
        'time_since_pause': time_since_pause
    }

def calculate_motion_features(input_path, output_path):
    """
    Calculate pause features for all motion derivatives.
    """
    # Verify file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Verify file is readable
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise IOError(f"Error reading CSV file: {str(e)}")
    
    # Verify output directory is writable
    output_dir = os.path.dirname(output_path)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to output directory: {output_dir}")
    
    df_out = df.copy()
    
    # Verify Timestamp column exists and calculate dt
    if 'Timestamp' not in df.columns:
        raise ValueError("Input CSV must contain a 'Timestamp' column")
    dt = df['Timestamp'].diff().mean()
    
    # Process scene-relative features
    objects = ['Head', 'LeftHand', 'RightHand']
    for obj in objects:
        # Verify required columns exist
        pos_cols = [f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']
        quat_cols = [
            f'{obj}_quat_w',  # Ensure w component comes first
            f'{obj}_quat_x',
            f'{obj}_quat_y',
            f'{obj}_quat_z'
        ]
        
        if not all(col in df.columns for col in pos_cols + quat_cols):
            raise ValueError(f"Missing required columns for {obj}")
        
        # Scene-relative velocities
        pos = df[pos_cols].to_numpy()
        quat = df[quat_cols].to_numpy()
        
        vel, acc, jerk, ang_vel, ang_acc, ang_jerk = calculate_motion_derivatives(pos, quat, dt)
        
        # Get binary pauses for all derivatives
        binary_pauses = detect_pauses(
            vel, acc, jerk, ang_vel, ang_acc, ang_jerk,
            percentile=10
        )
        
        # Calculate meaningful pause features for each derivative type
        for derivative_type, pause_signal in binary_pauses.items():
            pause_features = calculate_pause_features(pause_signal, dt)
            df_out[f'{obj}_{derivative_type}_pause_duration'] = pause_features['pause_duration']
            df_out[f'{obj}_{derivative_type}_time_since_pause'] = pause_features['time_since_pause']
    
    # Process body-relative features for hands
    hands = ['Left', 'Right']
    for hand in hands:
        # Verify required columns exist
        pos_cols_relative = [
            f'{hand}Hand_position_x_relative',
            f'{hand}Hand_position_y_relative',
            f'{hand}Hand_position_z_relative'
        ]
        quat_cols_relative = [
            f'{hand}Hand_quat_w_relative',  # Ensure w component comes first
            f'{hand}Hand_quat_x_relative',
            f'{hand}Hand_quat_y_relative',
            f'{hand}Hand_quat_z_relative'
        ]
        
        if not all(col in df.columns for col in pos_cols_relative + quat_cols_relative):
            raise ValueError(f"Missing required relative columns for {hand}Hand")
        
        # Body-relative velocities
        pos_relative = df[pos_cols_relative].to_numpy()
        quat_relative = df[quat_cols_relative].to_numpy()
        
        vel_relative, acc_relative, jerk_relative, ang_vel_relative, ang_acc_relative, ang_jerk_relative = calculate_motion_derivatives(pos_relative, quat_relative, dt)
        
        # Get binary pauses for all derivatives
        binary_pauses = detect_pauses(
            vel_relative, acc_relative, jerk_relative,
            ang_vel_relative, ang_acc_relative, ang_jerk_relative,
            percentile=10
        )
        
        # Calculate meaningful pause features for each derivative type
        for derivative_type, pause_signal in binary_pauses.items():
            pause_features = calculate_pause_features(pause_signal, dt)
            df_out[f'{hand}Hand_{derivative_type}_pause_duration_relative'] = pause_features['pause_duration']
            df_out[f'{hand}Hand_{derivative_type}_time_since_pause_relative'] = pause_features['time_since_pause']
    
    # Process head no-yaw quaternion
    head_no_yaw_quat_cols = [
        'Head_no_yaw_quat_w',  # Ensure w component comes first
        'Head_no_yaw_quat_x',
        'Head_no_yaw_quat_y',
        'Head_no_yaw_quat_z'
    ]
    
    if all(col in df.columns for col in head_no_yaw_quat_cols):
        head_no_yaw_quat = df[head_no_yaw_quat_cols].to_numpy()
        _, _, _, ang_vel_no_yaw, ang_acc_no_yaw, ang_jerk_no_yaw = calculate_motion_derivatives(
            np.zeros((len(df), 3)),
            head_no_yaw_quat, 
            dt
        )
        
        # For no-yaw, we only consider angular derivatives
        binary_pauses = detect_pauses(
            np.zeros_like(ang_vel_no_yaw), # Dummy values for linear derivatives
            np.zeros_like(ang_vel_no_yaw),
            np.zeros_like(ang_vel_no_yaw),
            ang_vel_no_yaw,
            ang_acc_no_yaw,
            ang_jerk_no_yaw,
            percentile=10
        )
        
        # Calculate pause features for angular derivatives only
        for derivative_type in ['angular_velocity', 'angular_acceleration', 'angular_jerk']:
            pause_features = calculate_pause_features(binary_pauses[derivative_type], dt)
            df_out[f'Head_no_yaw_{derivative_type}_pause_duration'] = pause_features['pause_duration']
            df_out[f'Head_no_yaw_{derivative_type}_time_since_pause'] = pause_features['time_since_pause']
    
    # Save to new CSV file
    df_out.to_csv(output_path, index=False)
    return df_out

def process_file(filename, input_dir, output_dir):
    """Process a single file with motion feature calculation."""
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    try:
        df_transformed = calculate_motion_features(input_path, output_path)
        return True
    except Exception as e:
        print(f"\nError processing {filename}: {str(e)}")
        return False

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "data/FAB/FAB_B_Modified_Motion"
    output_dir = "data/FAB/FAB_B_Modified_Motion_Pause"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    # Set up multiprocessing
    num_cores = mp.cpu_count() - 1  # Leave one core free
    print(f"Processing {total_files} files using {num_cores} cores...")
    
    # Create partial function with fixed arguments
    process_file_partial = partial(process_file, input_dir=input_dir, output_dir=output_dir)
    
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
