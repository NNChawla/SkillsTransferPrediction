import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

assembly_A_dir = "/srv/STP/data/FAB/Assembly_A"
assembly_B_dir = "/srv/STP/data/FAB/Assembly_B"
FAB_A_Resampled_dir = "/srv/STP/data/FAB/FAB_A_Resampled"
FAB_B_Resampled_dir = "/srv/STP/data/FAB/FAB_B_Resampled"
assembly_A_files = sorted(os.listdir(assembly_A_dir))
assembly_B_files = sorted(os.listdir(assembly_B_dir))
FAB_A_Resampled_files = sorted(os.listdir(FAB_A_Resampled_dir))
FAB_B_Resampled_files = sorted(os.listdir(FAB_B_Resampled_dir))

cols = ['Head_quat_x', 'Head_quat_y', 'Head_quat_z', 'Head_quat_w', 'RightHand_quat_x', 'RightHand_quat_y', 'RightHand_quat_z', 'RightHand_quat_w', 'LeftHand_quat_x', 'LeftHand_quat_y', 'LeftHand_quat_z', 'LeftHand_quat_w']
marker_colors = ['ro', 'go', 'ko']

# Create a single figure with subplots
fig, axes = plt.subplots(len(FAB_A_Resampled_files), len(cols), 
                         figsize=(90, 5*len(FAB_A_Resampled_files)))

for id in range(len(FAB_A_Resampled_files)):
    # Read CSVs using Polars
    assembly_A_df = pl.read_csv(os.path.join(assembly_A_dir, assembly_A_files[id]))
    assembly_B_df = pl.read_csv(os.path.join(assembly_B_dir, assembly_B_files[id]))
    df_A = pl.read_csv(os.path.join(FAB_A_Resampled_dir, FAB_A_Resampled_files[id]))
    df_B = pl.read_csv(os.path.join(FAB_B_Resampled_dir, FAB_B_Resampled_files[id]))
    participant_id = FAB_A_Resampled_files[id].split('_')[0]
    
    marker_timestamps = assembly_A_df["Timestamp"].drop_nulls()
    delta_t = 1 / 100
    x = df_A["Timestamp"]

    for idx, col in enumerate(cols):
        y = df_A[col].drop_nulls()
        
        # Get the current subplot
        if len(FAB_A_Resampled_files) == 1:
            ax = axes[idx]
        else:
            ax = axes[id, idx]
        
        # Plot the data
        ax.plot(x, y, label=col)
        
        # Add markers at each specified timestamp
        for t_idx, ts in enumerate(marker_timestamps):
            # Calculate the corresponding index. Ensure the index is within range.
            marker_index = int(ts / delta_t)
            if marker_index < len(x):
                ax.plot(x[marker_index], y[marker_index], marker_colors[t_idx % len(marker_colors)], markersize=6)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rads)')
        ax.set_ylim([-1, 1])
        ax.set_title(f'Participant {participant_id} - {col}')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

def plot_angle_axis_smoothed(alpha_min, alpha_max, threshold_lo, threshold_hi, id, smoothed=True):

    assembly_A_dir = "/srv/STP/data/FAB/Assembly_A"
    assembly_B_dir = "/srv/STP/data/FAB/Assembly_B"
    FAB_A_Resampled_dir = "/srv/STP/data/FAB/FAB_A_Resampled"
    FAB_B_Resampled_dir = "/srv/STP/data/FAB/FAB_B_Resampled"
    assembly_A_files = sorted(os.listdir(assembly_A_dir))
    assembly_B_files = sorted(os.listdir(assembly_B_dir))
    FAB_A_Resampled_files = sorted(os.listdir(FAB_A_Resampled_dir))
    FAB_B_Resampled_files = sorted(os.listdir(FAB_B_Resampled_dir))

    assembly_A_df = pl.read_csv(os.path.join(assembly_A_dir, assembly_A_files[id]))
    assembly_B_df = pl.read_csv(os.path.join(assembly_B_dir, assembly_B_files[id]))
    df_A = pl.read_csv(os.path.join(FAB_A_Resampled_dir, FAB_A_Resampled_files[id]))
    df_B = pl.read_csv(os.path.join(FAB_B_Resampled_dir, FAB_B_Resampled_files[id]))
    participant_id = FAB_A_Resampled_files[id].split('_')[0]

    marker_timestamps = assembly_A_df["Timestamp"].drop_nulls()
    marker_colors = ['ro', 'go', 'ko']

    timestamps = df_A['Timestamp'].to_numpy()
    head_quats = df_A[['Head_quat_x', 'Head_quat_y', 'Head_quat_z', 'Head_quat_w']].to_numpy()

    if smoothed:
        head_quats = adaptive_slerp_filter(
                head_quats, 
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                threshold_low=threshold_lo,
                threshold_high=threshold_hi
        )

    angles = []
    axes = []
    for q in head_quats:
        angle, axis = quat_to_axis_angle(q)
        angles.append(angle)
        axes.append(axis)
    angles = np.array(angles)  # feature: rotation angles (radians)
    axes = np.array(axes)      # feature: 3-element vectors per time step

    fig, ax = plt.subplots(figsize=(45, 1))
    ax.plot(timestamps, angles)

    for t_idx, ts in enumerate(marker_timestamps):
        marker_index = int(ts / delta_t)
        if marker_index < len(timestamps):
            # Use a format string with marker (e.g., 'o') and color
            ax.plot(
                timestamps[marker_index],  # x-coordinate
                angles[marker_index],                   # y-coordinate
                marker_colors[t_idx % len(marker_colors)],  # Add marker style
                markersize=6,
                linestyle='none'  # Ensure no line is drawn
            )