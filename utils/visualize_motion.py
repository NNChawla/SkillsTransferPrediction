import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os

def plot_frame(ax, position, rotation, scale=0.1, label=None):
    """Plot coordinate frame at given position with given rotation"""
    # Colors for each axis
    colors = ['r', 'g', 'b']
    
    # Create rotation matrix
    rot_matrix = R.from_quat(rotation).as_matrix()
    
    # Plot each axis
    for i, color in enumerate(colors):
        direction = rot_matrix[:, i]
        ax.quiver(position[0], position[1], position[2],
                 direction[0] * scale, direction[1] * scale, direction[2] * scale,
                 color=color, alpha=0.6)
    
    # Plot position point
    ax.scatter(position[0], position[1], position[2], color='black', s=50, label=label)

def visualize_frame(df, frame_idx=0, save_path=None):
    """Visualize a single frame of motion data and save to file"""
    fig = plt.figure(figsize=(20, 10))
    
    # World space plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Get positions and rotations for the frame (world space)
    head_pos = np.array([
        df.iloc[frame_idx]['Head_position_x'],
        df.iloc[frame_idx]['Head_position_y'],
        df.iloc[frame_idx]['Head_position_z']
    ])
    
    head_rot = np.array([
        df.iloc[frame_idx]['Head_quat_x'],
        df.iloc[frame_idx]['Head_quat_y'],
        df.iloc[frame_idx]['Head_quat_z'],
        df.iloc[frame_idx]['Head_quat_w']
    ])
    
    left_pos = np.array([
        df.iloc[frame_idx]['LeftHand_position_x'],
        df.iloc[frame_idx]['LeftHand_position_y'],
        df.iloc[frame_idx]['LeftHand_position_z']
    ])
    
    left_rot = np.array([
        df.iloc[frame_idx]['LeftHand_quat_x'],
        df.iloc[frame_idx]['LeftHand_quat_y'],
        df.iloc[frame_idx]['LeftHand_quat_z'],
        df.iloc[frame_idx]['LeftHand_quat_w']
    ])
    
    right_pos = np.array([
        df.iloc[frame_idx]['RightHand_position_x'],
        df.iloc[frame_idx]['RightHand_position_y'],
        df.iloc[frame_idx]['RightHand_position_z']
    ])
    
    right_rot = np.array([
        df.iloc[frame_idx]['RightHand_quat_x'],
        df.iloc[frame_idx]['RightHand_quat_y'],
        df.iloc[frame_idx]['RightHand_quat_z'],
        df.iloc[frame_idx]['RightHand_quat_w']
    ])
    
    # Plot world space
    plot_frame(ax1, head_pos, head_rot, scale=0.2, label='Head')
    plot_frame(ax1, left_pos, left_rot, scale=0.2, label='Left Hand')
    plot_frame(ax1, right_pos, right_rot, scale=0.2, label='Right Hand')
    
    # Connect points with lines
    points = np.vstack([head_pos, left_pos, right_pos])
    ax1.plot([head_pos[0], left_pos[0]], [head_pos[1], left_pos[1]], 
            [head_pos[2], left_pos[2]], 'k--', alpha=0.3)
    ax1.plot([head_pos[0], right_pos[0]], [head_pos[1], right_pos[1]], 
            [head_pos[2], right_pos[2]], 'k--', alpha=0.3)
    
    # Body-relative plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # For body-relative, head is at origin with identity rotation
    head_pos_rel = np.array([0, 0, 0])
    head_rot_rel = np.array([0, 0, 0, 1])  # Identity quaternion
    
    # Get relative rotations
    left_rot_rel = np.array([
        df.iloc[frame_idx]['LeftHand_quat_x_relative'],
        df.iloc[frame_idx]['LeftHand_quat_y_relative'],
        df.iloc[frame_idx]['LeftHand_quat_z_relative'],
        df.iloc[frame_idx]['LeftHand_quat_w_relative']
    ])
    
    right_rot_rel = np.array([
        df.iloc[frame_idx]['RightHand_quat_x_relative'],
        df.iloc[frame_idx]['RightHand_quat_y_relative'],
        df.iloc[frame_idx]['RightHand_quat_z_relative'],
        df.iloc[frame_idx]['RightHand_quat_w_relative']
    ])
    
    # Calculate relative positions (subtract head position)
    left_pos_rel = left_pos - head_pos
    right_pos_rel = right_pos - head_pos
    
    # Plot body-relative
    plot_frame(ax2, head_pos_rel, head_rot_rel, scale=0.2, label='Head')
    plot_frame(ax2, left_pos_rel, left_rot_rel, scale=0.2, label='Left Hand')
    plot_frame(ax2, right_pos_rel, right_rot_rel, scale=0.2, label='Right Hand')
    
    # Connect points with lines
    points_rel = np.vstack([head_pos_rel, left_pos_rel, right_pos_rel])
    ax2.plot([head_pos_rel[0], left_pos_rel[0]], [head_pos_rel[1], left_pos_rel[1]], 
            [head_pos_rel[2], left_pos_rel[2]], 'k--', alpha=0.3)
    ax2.plot([head_pos_rel[0], right_pos_rel[0]], [head_pos_rel[1], right_pos_rel[1]], 
            [head_pos_rel[2], right_pos_rel[2]], 'k--', alpha=0.3)
    
    # Set labels and titles
    for ax, title in [(ax1, 'World Space'), (ax2, 'Body-Relative')]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title} - Frame {frame_idx}')
        
        # Set equal aspect ratio
        if title == 'World Space':
            points_for_scale = points
        else:
            points_for_scale = points_rel
            
        max_range = np.array([
            points_for_scale[:,0].max()-points_for_scale[:,0].min(),
            points_for_scale[:,1].max()-points_for_scale[:,1].min(),
            points_for_scale[:,2].max()-points_for_scale[:,2].min()
        ]).max() / 2.0
        
        mid_x = (points_for_scale[:,0].max()+points_for_scale[:,0].min()) * 0.5
        mid_y = (points_for_scale[:,1].max()+points_for_scale[:,1].min()) * 0.5
        mid_z = (points_for_scale[:,2].max()+points_for_scale[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.legend()
    
    # Save plot instead of showing it
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv("data/FAB/example_body_relative.csv")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Visualize first 10 frames
    for i in range(10):
        visualize_frame(df, frame_idx=i, save_path=f"logs/frame_{i:03d}.png")