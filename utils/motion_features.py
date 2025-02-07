import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from scipy.signal import savgol_filter

def calculate_motion_derivatives(positions, quaternions, dt=1/30):
    """
    Calculate linear and angular velocities, accelerations, and jerk from positions and quaternions.
    
    Args:
        positions: numpy array of shape (n, 3) containing xyz positions
        quaternions: numpy array of shape (n, 4) containing quaternions (w,x,y,z)
        dt: time step between frames (default: 1/30 second)
    """
    # Calculate linear velocity using central differences
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    
    # Calculate linear acceleration
    accelerations = np.zeros_like(positions)
    accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt)
    accelerations[0] = (velocities[1] - velocities[0]) / dt
    accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
    
    # Calculate linear jerk
    jerks = np.zeros_like(positions)
    jerks[1:-1] = (accelerations[2:] - accelerations[:-2]) / (2 * dt)
    jerks[0] = (accelerations[1] - accelerations[0]) / dt
    jerks[-1] = (accelerations[-1] - accelerations[-2]) / dt

    # Calculate angular derivatives
    angular_vel = np.zeros((len(quaternions), 3))
    for i in range(1, len(quaternions)):
        q1 = quaternions[i-1]
        q2 = quaternions[i]
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        r_diff = r2 * r1.inv()
        rotvec = r_diff.as_rotvec()
        angular_vel[i] = rotvec / dt
    
    # Calculate angular acceleration and jerk
    angular_acc = np.zeros_like(angular_vel)
    angular_acc[1:-1] = (angular_vel[2:] - angular_vel[:-2]) / (2 * dt)
    angular_acc[0] = (angular_vel[1] - angular_vel[0]) / dt
    angular_acc[-1] = (angular_vel[-1] - angular_vel[-2]) / dt
    
    angular_jerk = np.zeros_like(angular_vel)
    angular_jerk[1:-1] = (angular_acc[2:] - angular_acc[:-2]) / (2 * dt)
    angular_jerk[0] = (angular_acc[1] - angular_acc[0]) / dt
    angular_jerk[-1] = (angular_acc[-1] - angular_acc[-2]) / dt
    
    # Apply smoothing to reduce noise
    velocities = savgol_filter(velocities, window_length=5, polyorder=2, axis=0)
    accelerations = savgol_filter(accelerations, window_length=5, polyorder=2, axis=0)
    jerks = savgol_filter(jerks, window_length=5, polyorder=2, axis=0)
    angular_vel = savgol_filter(angular_vel, window_length=5, polyorder=2, axis=0)
    angular_acc = savgol_filter(angular_acc, window_length=5, polyorder=2, axis=0)
    angular_jerk = savgol_filter(angular_jerk, window_length=5, polyorder=2, axis=0)
    
    return velocities, accelerations, jerks, angular_vel, angular_acc, angular_jerk

def calculate_motion_features(input_path, output_path):
    """
    Calculate motion features (linear and angular velocities) for scene-relative coordinates.
    """
    # Read the CSV file
    df = pd.read_csv(input_path)
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
        
        # Add all motion derivatives
        df_out[f'{obj}_velocity_x'] = vel[:, 0]
        df_out[f'{obj}_velocity_y'] = vel[:, 1]
        df_out[f'{obj}_velocity_z'] = vel[:, 2]
        df_out[f'{obj}_acceleration_x'] = acc[:, 0]
        df_out[f'{obj}_acceleration_y'] = acc[:, 1]
        df_out[f'{obj}_acceleration_z'] = acc[:, 2]
        df_out[f'{obj}_jerk_x'] = jerk[:, 0]
        df_out[f'{obj}_jerk_y'] = jerk[:, 1]
        df_out[f'{obj}_jerk_z'] = jerk[:, 2]
        df_out[f'{obj}_angular_velocity_x'] = ang_vel[:, 0]
        df_out[f'{obj}_angular_velocity_y'] = ang_vel[:, 1]
        df_out[f'{obj}_angular_velocity_z'] = ang_vel[:, 2]
        df_out[f'{obj}_angular_acceleration_x'] = ang_acc[:, 0]
        df_out[f'{obj}_angular_acceleration_y'] = ang_acc[:, 1]
        df_out[f'{obj}_angular_acceleration_z'] = ang_acc[:, 2]
        df_out[f'{obj}_angular_jerk_x'] = ang_jerk[:, 0]
        df_out[f'{obj}_angular_jerk_y'] = ang_jerk[:, 1]
        df_out[f'{obj}_angular_jerk_z'] = ang_jerk[:, 2]
    
    # Save to new CSV file
    df_out.to_csv(output_path, index=False)
    return df_out

# Example usage
if __name__ == "__main__":
    # Define input and output directories
    input_dir = "data/FAB/FAB_B_HandRelative"
    output_dir = "data/FAB/FAB_B_HandRelative_Motion"
    
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
