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

def calculate_motion_features(input_path, output_path):
    """
    Calculate motion features (linear and angular velocities) for both 
    scene-relative and body-relative coordinates.
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
        quat = df[quat_cols].to_numpy()  # Now correctly ordered as (w,x,y,z)
        
        vel, ang_vel = calculate_velocities(pos, quat, dt)
        
        # Add scene-relative velocities
        df_out[f'{obj}_velocity_x'] = vel[:, 0]
        df_out[f'{obj}_velocity_y'] = vel[:, 1]
        df_out[f'{obj}_velocity_z'] = vel[:, 2]
        df_out[f'{obj}_angular_velocity_x'] = ang_vel[:, 0]
        df_out[f'{obj}_angular_velocity_y'] = ang_vel[:, 1]
        df_out[f'{obj}_angular_velocity_z'] = ang_vel[:, 2]
    
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
        quat_relative = df[quat_cols_relative].to_numpy()  # Now correctly ordered as (w,x,y,z)
        
        vel_relative, ang_vel_relative = calculate_velocities(pos_relative, quat_relative, dt)
        
        # Add body-relative velocities
        df_out[f'{hand}Hand_velocity_x_relative'] = vel_relative[:, 0]
        df_out[f'{hand}Hand_velocity_y_relative'] = vel_relative[:, 1]
        df_out[f'{hand}Hand_velocity_z_relative'] = vel_relative[:, 2]
        df_out[f'{hand}Hand_angular_velocity_x_relative'] = ang_vel_relative[:, 0]
        df_out[f'{hand}Hand_angular_velocity_y_relative'] = ang_vel_relative[:, 1]
        df_out[f'{hand}Hand_angular_velocity_z_relative'] = ang_vel_relative[:, 2]
    
    # Process head no-yaw quaternion
    head_no_yaw_quat_cols = [
        'Head_no_yaw_quat_w',  # Ensure w component comes first
        'Head_no_yaw_quat_x',
        'Head_no_yaw_quat_y',
        'Head_no_yaw_quat_z'
    ]
    
    if all(col in df.columns for col in head_no_yaw_quat_cols):
        # Calculate angular velocity for head no-yaw rotation
        head_no_yaw_quat = df[head_no_yaw_quat_cols].to_numpy()  # Now correctly ordered as (w,x,y,z)
        _, ang_vel_no_yaw = calculate_velocities(
            np.zeros((len(df), 3)),  # Dummy positions since we only need angular velocity
            head_no_yaw_quat, 
            dt
        )
        
        # Add head no-yaw angular velocities
        df_out['Head_no_yaw_angular_velocity_x'] = ang_vel_no_yaw[:, 0]
        df_out['Head_no_yaw_angular_velocity_y'] = ang_vel_no_yaw[:, 1]
        df_out['Head_no_yaw_angular_velocity_z'] = ang_vel_no_yaw[:, 2]
    
    # Save to new CSV file
    df_out.to_csv(output_path, index=False)
    return df_out

# Example usage
if __name__ == "__main__":
    # Define input and output directories
    input_dir = "data/FAB/FAB_B_Modified"
    output_dir = "data/FAB/FAB_B_Modified_Motion"
    
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
