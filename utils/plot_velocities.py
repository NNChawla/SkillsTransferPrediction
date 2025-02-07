import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from motion_features import calculate_motion_derivatives
import os
from tqdm import tqdm

def plot_participant_velocities(input_dir, output_path=None, max_files=None, max_frames=2000, participant_id=None, num_xticks=10):
    """
    Create an overlay plot of linear and angular velocities for all participants or a single participant.
    
    Args:
        input_dir: Directory containing motion CSV files
        output_path: Optional path to save the plot
        max_files: Optional limit on number of files to process
        max_frames: Maximum number of frames to plot
        participant_id: Optional participant ID to plot single participant data
        num_xticks: Number of ticks to show on x-axis (default: 10)
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
    
    # Create separate subplots for each object, with two columns (linear and angular)
    fig, axes = plt.subplots(3, 3, figsize=(24, 12), sharex=True)  # Changed to 3x3 grid
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Define thresholds for linear and angular velocity
    linear_thresholds = [0.1, 0.25, 0.5, 1.0]  # m/s
    angular_thresholds = [0.5, 1.0, 2.0, 4.0]  # rad/s
    threshold_colors = ['#FF9999', '#FF6666', '#FF3333', '#FF0000']
    
    # Dictionary to store velocity data for percentile calculation
    vel_data = {
        'Head': {'linear_x': [], 'linear_y': [], 'linear_z': [], 'angular_x': [], 'angular_y': [], 'angular_z': []},
        'LeftHand': {'linear_x': [], 'linear_y': [], 'linear_z': [], 'angular_x': [], 'angular_y': [], 'angular_z': []},
        'RightHand': {'linear_x': [], 'linear_y': [], 'linear_z': [], 'angular_x': [], 'angular_y': [], 'angular_z': []}
    }
    
    # Process each file
    for filename in tqdm(csv_files, desc="Processing files"):
        # Read data
        df = pd.read_csv(os.path.join(input_dir, filename))
        dt = df['Timestamp'].diff().mean()
        
        # Process each tracked object
        objects = ['Head', 'LeftHand', 'RightHand']
        
        for obj in objects:
            # Get position and quaternion data
            pos_cols = [f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']
            quat_cols = [f'{obj}_quat_w', f'{obj}_quat_x', f'{obj}_quat_y', f'{obj}_quat_z']
            
            pos = df[pos_cols].to_numpy()
            quat = df[quat_cols].to_numpy()
            
            # Calculate velocities
            lin_vel, _, _, ang_vel, _, _ = calculate_motion_derivatives(pos, quat, dt)
            
            # Limit to max_frames
            lin_vel = lin_vel[:max_frames]
            ang_vel = ang_vel[:max_frames]
            
            # Adjust line visibility based on whether plotting single or multiple participants
            line_alpha = 0.8 if participant_id else 0.1
            line_width = 1.5 if participant_id else 0.5
            
            # Plot time series
            row_idx = objects.index(obj)
            colors = ['r', 'g', 'b']
            labels = ['x', 'y', 'z']
            
            # Plot each component
            for i, (color, label) in enumerate(zip(colors, labels)):
                axes[row_idx, i].plot(range(len(lin_vel)), lin_vel[:, i],
                                    alpha=line_alpha, linewidth=line_width, color=color,
                                    label=f'Linear {label}')
                axes[row_idx, i].plot(range(len(ang_vel)), ang_vel[:, i],
                                    alpha=line_alpha, linewidth=line_width, color=color,
                                    linestyle='--', label=f'Angular {label}')
            
            # Store data for percentile calculation
            vel_data[obj]['linear_x'].extend(lin_vel[:, 0])
            vel_data[obj]['linear_y'].extend(lin_vel[:, 1])
            vel_data[obj]['linear_z'].extend(lin_vel[:, 2])
            vel_data[obj]['angular_x'].extend(ang_vel[:, 0])
            vel_data[obj]['angular_y'].extend(ang_vel[:, 1])
            vel_data[obj]['angular_z'].extend(ang_vel[:, 2])
    
    # Customize each subplot
    for idx, obj in enumerate(objects):
        for i, component in enumerate(['X', 'Y', 'Z']):
            ax = axes[idx, i]
            ax.set_title(f'{obj} {component}-axis Velocities')
            ax.set_ylabel('Velocity (m/s, rad/s)')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.set_xlim(0, max_frames)
            ax.legend(loc='upper right')
            
            # Set reasonable y-axis limits
            ax.set_ylim(-5, 5)  # Adjust based on your data
            
            # Configure x-axis ticks
            tick_spacing = max_frames // (num_xticks - 1)
            ax.set_xticks(np.linspace(0, max_frames, num_xticks))
            
            # Add frame labels to bottom row only
            if idx == len(objects) - 1:
                ax.set_xlabel('Frame')

    # Update plot title
    title_prefix = f'Participant {participant_id}' if participant_id else 'All Participants'
    plt.suptitle(f'{title_prefix} Component-wise Velocities by Body Part\n(Solid: Linear, Dashed: Angular)', 
                 fontsize=14, y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    input_dir = "data/FAB/FAB_B_Modified_Motion"
    output_path = "figures/velocity_timeseries.png"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create the plot
    plot_participant_velocities(
        input_dir=input_dir,
        output_path=output_path,
        max_files=None,
        max_frames=2300,
        participant_id="20552M",
        num_xticks=15  # Example: show 12 ticks on x-axis
    ) 