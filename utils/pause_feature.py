import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from scipy.signal import savgol_filter

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

def detect_pauses(velocities, angular_velocities, percentile=25):
    """
    Detect periods where both linear and angular motion are below thresholds
    determined by the data distribution.
    
    Args:
        velocities: numpy array of shape (n, 3) containing xyz velocities
        angular_velocities: numpy array of shape (n, 3) containing angular velocities
        percentile: percentile below which motion is considered 'paused' (default: 25)
    
    Returns:
        numpy array of shape (n,) with 1s indicating pauses and 0s indicating motion
    """
    # Add input validation
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100")
    
    if velocities.shape != angular_velocities.shape[:2]:
        raise ValueError("Velocity arrays must have compatible shapes")
    
    # Calculate magnitude of velocities
    linear_magnitude = np.linalg.norm(velocities, axis=1)
    angular_magnitude = np.linalg.norm(angular_velocities, axis=1)
    
    # Calculate thresholds based on data distribution
    linear_threshold = np.percentile(linear_magnitude, percentile)
    angular_threshold = np.percentile(angular_magnitude, percentile)
    
    # Detect where both linear and angular motion are below thresholds
    is_paused = (linear_magnitude < linear_threshold) & (angular_magnitude < angular_threshold)
    return is_paused.astype(int)

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
    Calculate pause features for both scene-relative and body-relative coordinates.
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
        
        vel, ang_vel = calculate_velocities(pos, quat, dt)
        
        # Get binary pauses first
        binary_pauses = detect_pauses(vel, ang_vel, percentile=10)
        
        # Calculate meaningful pause features
        pause_features = calculate_pause_features(binary_pauses, dt)
        df_out[f'{obj}_pause_duration'] = pause_features['pause_duration']
        df_out[f'{obj}_time_since_pause'] = pause_features['time_since_pause']
    
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
        
        vel_relative, ang_vel_relative = calculate_velocities(pos_relative, quat_relative, dt)
        
        # Calculate relative pause features
        binary_pauses = detect_pauses(vel_relative, ang_vel_relative, percentile=10)
        pause_features = calculate_pause_features(binary_pauses, dt)
        df_out[f'{hand}Hand_pause_duration_relative'] = pause_features['pause_duration']
        df_out[f'{hand}Hand_time_since_pause_relative'] = pause_features['time_since_pause']
    
    # Process head no-yaw quaternion
    head_no_yaw_quat_cols = [
        'Head_no_yaw_quat_w',  # Ensure w component comes first
        'Head_no_yaw_quat_x',
        'Head_no_yaw_quat_y',
        'Head_no_yaw_quat_z'
    ]
    
    if all(col in df.columns for col in head_no_yaw_quat_cols):
        head_no_yaw_quat = df[head_no_yaw_quat_cols].to_numpy()
        _, ang_vel_no_yaw = calculate_velocities(
            np.zeros((len(df), 3)),
            head_no_yaw_quat, 
            dt
        )
        
        # For no-yaw, we only consider angular velocity
        angular_magnitude = np.linalg.norm(ang_vel_no_yaw, axis=1)
        angular_threshold = np.percentile(angular_magnitude, 10)
        binary_pauses = (angular_magnitude < angular_threshold).astype(int)
        
        pause_features = calculate_pause_features(binary_pauses, dt)
        df_out['Head_no_yaw_pause_duration'] = pause_features['pause_duration']
        df_out['Head_no_yaw_time_since_pause'] = pause_features['time_since_pause']
    
    # Save to new CSV file
    df_out.to_csv(output_path, index=False)
    return df_out

# Example usage
if __name__ == "__main__":
    # Define input and output directories
    input_dir = "data/FAB/FAB_A_Modified_Motion"
    output_dir = "data/FAB/FAB_A_Modified_Motion_Pause"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing {filename}...")
            try:
                df_transformed = calculate_motion_features(input_path, output_path)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
