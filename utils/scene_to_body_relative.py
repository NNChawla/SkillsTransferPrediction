import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# Function to convert scene-relative data to body-relative
def convert_to_body_relative(df):
    """
    Converts scene-relative data to body-relative data.
    Input DataFrame must have columns for positions and quaternions:
    - Positions: head_x, head_y, head_z, left_x, left_y, left_z, right_x, right_y, right_z
    - Quaternions: head_w, head_x_q, head_y_q, head_z_q, left_w, left_x_q, left_y_q, left_z_q, right_w, right_x_q, right_y_q, right_z_q
    """
    # Create columns for the output
    df_body_relative = pd.DataFrame(index=df.index)

    # Convert positions to body-relative
    for hand in ['left', 'right']:
        df_body_relative[f'{hand}_x_br'] = df[f'{hand}_x'] - df['head_x']
        df_body_relative[f'{hand}_y_br'] = df[f'{hand}_y'] - df['head_y']
        df_body_relative[f'{hand}_z_br'] = df[f'{hand}_z'] - df['head_z']

    # Process rotations
    for hand in ['left', 'right']:
        # Extract head and hand quaternions
        head_quat = df[['head_w', 'head_x_q', 'head_y_q', 'head_z_q']].to_numpy()
        hand_quat = df[[f'{hand}_w', f'{hand}_x_q', f'{hand}_y_q', f'{hand}_z_q']].to_numpy()

        # Compute the inverse of the head quaternion
        head_rotation = R.from_quat(head_quat)
        head_rotation_inv = head_rotation.inv()

        # Compute body-relative rotation for the hand
        hand_rotation = R.from_quat(hand_quat)
        body_relative_rotation = head_rotation_inv * hand_rotation
        body_relative_quat = body_relative_rotation.as_quat()

        # Save the body-relative rotation in the DataFrame
        df_body_relative[f'{hand}_w_br'] = body_relative_quat[:, 0]
        df_body_relative[f'{hand}_x_q_br'] = body_relative_quat[:, 1]
        df_body_relative[f'{hand}_y_q_br'] = body_relative_quat[:, 2]
        df_body_relative[f'{hand}_z_q_br'] = body_relative_quat[:, 3]

    # Use existing euler angles and zero out yaw for head
    head_euler = np.array([
        df['Head_euler_x'],
        df['Head_euler_y'],
        np.zeros_like(df['Head_euler_x'])  # Zero out yaw with matching length
    ]).T
    
    # Convert euler angles (with no yaw) back to quaternion
    head_no_yaw = R.from_euler('xyz', head_euler)
    head_no_yaw_quat = head_no_yaw.as_quat()
    
    df_body_relative['Head_no_yaw_quat_x'] = head_no_yaw_quat[:, 0]
    df_body_relative['Head_no_yaw_quat_y'] = head_no_yaw_quat[:, 1]
    df_body_relative['Head_no_yaw_quat_z'] = head_no_yaw_quat[:, 2]
    df_body_relative['Head_no_yaw_quat_w'] = head_no_yaw_quat[:, 3]

    return df_body_relative

def convert_fab_to_body_relative(input_path, output_path):
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Create a mapping dictionary to match the original column names
    column_mapping = {
        'head_x': 'Head_position_x',
        'head_y': 'Head_position_y',
        'head_z': 'Head_position_z',
        'head_w': 'Head_quat_w',
        'head_x_q': 'Head_quat_x',
        'head_y_q': 'Head_quat_y',
        'head_z_q': 'Head_quat_z',
        'Head_euler_x': 'Head_euler_x',
        'Head_euler_y': 'Head_euler_y',
        'Head_euler_z': 'Head_euler_z',
        'left_x': 'LeftHand_position_x',
        'left_y': 'LeftHand_position_y',
        'left_z': 'LeftHand_position_z',
        'left_w': 'LeftHand_quat_w',
        'left_x_q': 'LeftHand_quat_x',
        'left_y_q': 'LeftHand_quat_y',
        'left_z_q': 'LeftHand_quat_z',
        'LeftHand_euler_x': 'LeftHand_euler_x',
        'LeftHand_euler_y': 'LeftHand_euler_y',
        'LeftHand_euler_z': 'LeftHand_euler_z',
        'right_x': 'RightHand_position_x',
        'right_y': 'RightHand_position_y',
        'right_z': 'RightHand_position_z',
        'right_w': 'RightHand_quat_w',
        'right_x_q': 'RightHand_quat_x',
        'right_y_q': 'RightHand_quat_y',
        'right_z_q': 'RightHand_quat_z',
        'RightHand_euler_x': 'RightHand_euler_x',
        'RightHand_euler_y': 'RightHand_euler_y',
        'RightHand_euler_z': 'RightHand_euler_z'
    }
    
    # Create a new DataFrame with renamed columns for processing
    process_df = pd.DataFrame()
    for new_col, old_col in column_mapping.items():
        process_df[new_col] = df[old_col]
    
    # Get body-relative transformations
    body_relative = convert_to_body_relative(process_df)
    
    # Create output DataFrame starting with all original columns
    df_body_relative = df.copy()
    
    # Add body-relative columns with the correct naming scheme
    output_mapping = {
        'left_x_br': 'LeftHand_position_x_relative',
        'left_y_br': 'LeftHand_position_y_relative',
        'left_z_br': 'LeftHand_position_z_relative',
        'left_w_br': 'LeftHand_quat_w_relative',
        'left_x_q_br': 'LeftHand_quat_x_relative',
        'left_y_q_br': 'LeftHand_quat_y_relative',
        'left_z_q_br': 'LeftHand_quat_z_relative',
        'right_x_br': 'RightHand_position_x_relative',
        'right_y_br': 'RightHand_position_y_relative',
        'right_z_br': 'RightHand_position_z_relative',
        'right_w_br': 'RightHand_quat_w_relative',
        'right_x_q_br': 'RightHand_quat_x_relative',
        'right_y_q_br': 'RightHand_quat_y_relative',
        'right_z_q_br': 'RightHand_quat_z_relative',
        'Head_no_yaw_quat_x': 'Head_no_yaw_quat_x',
        'Head_no_yaw_quat_y': 'Head_no_yaw_quat_y',
        'Head_no_yaw_quat_z': 'Head_no_yaw_quat_z',
        'Head_no_yaw_quat_w': 'Head_no_yaw_quat_w'
    }
    
    # Add the transformed columns at the end
    for old_col, new_col in output_mapping.items():
        df_body_relative[new_col] = body_relative[old_col]
    
    # Save to new CSV file
    df_body_relative.to_csv(output_path, index=False)
    return df_body_relative

# Example usage
if __name__ == "__main__":
    # Define input and output directories
    input_dir = "data/FAB/FAB_B_Complete"
    output_dir = "data/FAB/FAB_B_Modified"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing {filename}...")
            try:
                df_transformed = convert_fab_to_body_relative(input_path, output_path)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
