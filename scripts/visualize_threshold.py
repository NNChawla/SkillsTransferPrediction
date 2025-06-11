assembly_A_dir = "/srv/STP/data/FAB/Assembly_A"
FAB_A_Resampled_dir = "/srv/STP/data/FAB/FAB_A_Step_Motion_Finite_Std"
assembly_A_files = sorted(os.listdir(assembly_A_dir))
FAB_A_Resampled_files = sorted(os.listdir(FAB_A_Resampled_dir))
index = 92
id = [i.split('_')[0] for i in assembly_A_files].index(pids[index])
assembly_A_df = pd.read_csv(os.path.join(assembly_A_dir, assembly_A_files[id]))
df_A = pd.read_csv(os.path.join(FAB_A_Resampled_dir, FAB_A_Resampled_files[id]))
participant_id = FAB_A_Resampled_files[id].split('_')[0]
print(participant_id)
marker_timestamps = assembly_A_df["Timestamp"]
marker_colors = ['ro', 'go', 'ko']
y = df_A[f'Head_linacc_x_451']
x = np.arange(len(y)) * (1/100)
fig, ax = plt.subplots(figsize=(75, 10))
element = 3
value = vectors[index][element].abs()
ax.plot(value, label=f"Participant_{index}")
threshold = np.mean(value) + 0.75 * np.std(value)
ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.2)
threshold = np.mean(value) + 1.0 * np.std(value)
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.2)
for t_idx, ts in enumerate(marker_timestamps):
    # Calculate the corresponding index. Ensure the index is within range.
    marker_index = int(ts / (1 / 100))
    print(f"{t_idx} | {ts} | {marker_index} | {x[marker_index]} | {value[marker_index]}")
    if marker_index < len(x):
        ax.plot(marker_index, value[marker_index], marker_colors[t_idx % len(marker_colors)], markersize=6)