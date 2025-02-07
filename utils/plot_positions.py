import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
from tqdm import tqdm

def quaternion_to_euler(quat):
    """Convert quaternions to Euler angles (in degrees)."""
    # Reorder from [w,x,y,z] to [x,y,z,w] for scipy
    quat_reordered = np.column_stack((quat[:, 1:], quat[:, 0]))
    r = Rotation.from_quat(quat_reordered)
    return r.as_euler('xyz', degrees=True)

def plot_participant_positions(input_dir, output_path=None, max_files=None, max_frames=2000, participant_id=None):
    """
    Create an overlay plot of positions and rotations for all participants or a single participant.
    
    Args:
        input_dir: Directory containing motion CSV files
        output_path: Optional path to save the plot
        max_files: Optional limit on number of files to process
        max_frames: Maximum number of frames to plot
        participant_id: Optional participant ID to plot single participant data
    """
    # Get list of CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if participant_id:
        csv_files = [f for f in csv_files if participant_id in f]
        print(f"Looking for files containing '{participant_id}'")
        print(f"Available files: {csv_files}")
        if not csv_files:
            print(f"All files in directory: {[f for f in os.listdir(input_dir) if f.endswith('.csv')]}")
            raise ValueError(f"No files found for participant {participant_id}")
    if max_files:
        csv_files = csv_files[:max_files]
    
    # Create separate subplots for each object, with two columns (position and rotation)
    fig, axes = plt.subplots(3, 2, figsize=(20, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Process each file
    for filename in tqdm(csv_files, desc="Processing files"):
        # Read data
        df = pd.read_csv(os.path.join(input_dir, filename))
        
        # Process each tracked object
        objects = ['Head', 'LeftHand', 'RightHand']
        
        for obj in objects:
            # Get position and quaternion data
            pos_cols = [f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']
            quat_cols = [f'{obj}_quat_w', f'{obj}_quat_x', f'{obj}_quat_y', f'{obj}_quat_z']
            
            pos = df[pos_cols].to_numpy()[:max_frames]
            quat = df[quat_cols].to_numpy()[:max_frames]
            
            # Convert quaternions to Euler angles
            euler = quaternion_to_euler(quat)
            
            # Adjust line visibility based on whether plotting single or multiple participants
            line_alpha = 0.8 if participant_id else 0.1
            line_width = 1.5 if participant_id else 0.5
            
            # Plot time series
            row_idx = objects.index(obj)
            
            # Position plot (x, y, z)
            ax_pos = axes[row_idx, 0]
            ax_pos.plot(range(len(pos)), pos[:, 0], alpha=line_alpha, linewidth=line_width, 
                       color='red', label='X')
            ax_pos.plot(range(len(pos)), pos[:, 1], alpha=line_alpha, linewidth=line_width, 
                       color='green', label='Y')
            ax_pos.plot(range(len(pos)), pos[:, 2], alpha=line_alpha, linewidth=line_width, 
                       color='blue', label='Z')
            
            # Rotation plot (roll, pitch, yaw)
            ax_rot = axes[row_idx, 1]
            ax_rot.plot(range(len(euler)), euler[:, 0], alpha=line_alpha, linewidth=line_width, 
                       color='red', label='Roll')
            ax_rot.plot(range(len(euler)), euler[:, 1], alpha=line_alpha, linewidth=line_width, 
                       color='green', label='Pitch')
            ax_rot.plot(range(len(euler)), euler[:, 2], alpha=line_alpha, linewidth=line_width, 
                       color='blue', label='Yaw')
    
    # Customize each subplot
    for idx, obj in enumerate(objects):
        ax_pos = axes[idx, 0]
        ax_rot = axes[idx, 1]
        
        # Position subplot
        ax_pos.set_title(f'{obj} Position Over Time')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.grid(True, which="both", ls="-", alpha=0.2)
        ax_pos.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        # Rotation subplot
        ax_rot.set_title(f'{obj} Rotation Over Time')
        ax_rot.set_ylabel('Angle (degrees)')
        ax_rot.grid(True, which="both", ls="-", alpha=0.2)
        ax_rot.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        # Set x-axis properties for both subplots
        for ax in [ax_pos, ax_rot]:
            ax.set_xlim(0, max_frames)
            num_ticks = 15
            ax.xaxis.set_major_locator(plt.LinearLocator(num_ticks))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: int(x)))
        
        # Set y-axis limits
        ax_pos.set_ylim(-2, 2)  # Adjust based on your data
        ax_rot.set_ylim(-180, 180)  # Euler angles range
    
    # Add common x-label
    for ax in axes[-1]:
        ax.set_xlabel('Frame')
    
    # Update plot title based on whether showing single or multiple participants
    title_prefix = f'Participant {participant_id}' if participant_id else 'All Participants'
    plt.suptitle(f'{title_prefix} Position and Rotation Time Series by Body Part', fontsize=14, y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    input_dir = "data/FAB/FAB_B_Modified_Motion"
    output_path = "figures/position_rotation_timeseries.png"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create the plot
    plot_participant_positions(
        input_dir=input_dir,
        output_path=output_path,
        max_files=None,
        max_frames=2300,
        participant_id="20552M"
    ) 