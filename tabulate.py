import os, sys, time
import numpy as np
import polars as pl
from scripts.analysis import get_linear_features, get_feature_names, calculate_lyapunov_exponent
import tqdm
import multiprocessing as mp

# Directories
assembly_A_dir = "./data/FAB/Assembly_A"
assembly_B_dir = "./data/FAB/Assembly_B"
FAB_A_Complete_dir = "./data/FAB/FAB_A_Complete"
FAB_B_Complete_dir = "./data/FAB/FAB_B_Complete"

# Get all files in the directories
assembly_A_files = sorted(os.listdir(assembly_A_dir))
assembly_B_files = sorted(os.listdir(assembly_B_dir))
FAB_A_Complete_files = sorted(os.listdir(FAB_A_Complete_dir))
FAB_B_Complete_files = sorted(os.listdir(FAB_B_Complete_dir))

# Read CSVs using Polars
assembly_A_csvs = [
    pl.read_csv(os.path.join(assembly_A_dir, file))
    for file in assembly_A_files if file.endswith('.csv')
]
assembly_B_csvs = [
    pl.read_csv(os.path.join(assembly_B_dir, file))
    for file in assembly_B_files if file.endswith('.csv')
]
FAB_A_Complete_csvs = [
    pl.read_csv(os.path.join(FAB_A_Complete_dir, file))
    for file in FAB_A_Complete_files if file.endswith('.csv')
]
FAB_B_Complete_csvs = [
    pl.read_csv(os.path.join(FAB_B_Complete_dir, file))
    for file in FAB_B_Complete_files if file.endswith('.csv')
]

assembly_csvs = [assembly_A_csvs, assembly_B_csvs]
tracking_csvs = [FAB_A_Complete_csvs, FAB_B_Complete_csvs]

file_names = [FAB_A_Complete_files, FAB_B_Complete_files]
participant_ids = [f.split('_')[0] for f in FAB_A_Complete_files]
num_participants = len(participant_ids)

t = time.time()
# Resample the data to 0.01 sec intervals
for idx, dataset in enumerate(tracking_csvs):
    for i in range(len(dataset)):
        df = dataset[i]
        # Remove duplicate Timestamps (keep first occurrence)
        orig_height = df.height
        df = df.unique(subset=["Timestamp"], maintain_order=True)
        if df.height < orig_height:
            print(f"Warning: Found duplicates in {file_names[idx][i]}")
        
        # Get min and max Timestamp values
        min_ts = df.select(pl.col("Timestamp")).min().item()
        max_ts = df.select(pl.col("Timestamp")).max().item()
        
        # Create a new uniform timestamp range (0.01 sec intervals)
        new_index = np.arange(min_ts, max_ts + 0.01, 0.01)
        df_uniform = pl.DataFrame({"Timestamp": new_index})
        
        # Left join the original data to the uniform timestamps
        df_uniform = df_uniform.join(df, on="Timestamp", how="left")
        
        # Interpolate missing values in numeric columns (only for 'position' or 'euler' columns)
        cols_to_interp = [
            col for col in df_uniform.columns
            if col != "Timestamp" and ("position" in col or "euler" in col)
        ]
        for col in cols_to_interp:
            # Using Polars' interpolate method on the Series (requires a recent version)
            interpolated = df_uniform[col].interpolate()
            df_uniform = df_uniform.with_columns(interpolated.alias(col))
        
        # Keep only the Timestamp and relevant columns
        cols = ["Timestamp"] + [col for col in df_uniform.columns if ("position" in col or "euler" in col)]
        df_uniform = df_uniform.select(cols)
        
        dataset[i] = df_uniform

print(f"Time taken to resample the data: {time.time() - t} seconds")

linear_feature_names = get_feature_names()

# Prepare tabulated data storage
tabulated_data = {
    "PID": participant_ids
}

# Determine the columns to tabulate (position and euler)
position_and_euler_columns = [
    col for col in FAB_A_Complete_csvs[0].columns
    if ("position" in col or "euler" in col)
]

# Initialize table for basic stats (mean, std, min, max, median)
for col in position_and_euler_columns:
    for stat in linear_feature_names:
        for assembly in ["A", "B"]:
            tabulated_data[f"{col}_{stat}_{assembly}"] = [np.nan] * num_participants

# Initialize table for velocity and acceleration stats over multiple window sizes
window_sizes = [1, 10, 25, 50, 100, 250]
for window in window_sizes:
    for col in position_and_euler_columns:
        vel_col = f"{col.replace('position', 'linear_velocity').replace('euler', 'angular_velocity')}_{window}"
        acc_col = f"{col.replace('position', 'linear_acceleration').replace('euler', 'angular_acceleration')}_{window}"
        for stat in linear_feature_names:
            for assembly in ["A", "B"]:
                tabulated_data[f"{vel_col}_{stat}_{assembly}"] = [np.nan] * num_participants
                tabulated_data[f"{acc_col}_{stat}_{assembly}"] = [np.nan] * num_participants

# Tabulate position and euler statistics
def process_participant(i):
    results_dict = {}
    df_A = FAB_A_Complete_csvs[i]
    df_B = FAB_B_Complete_csvs[i]
    for col in position_and_euler_columns:
        # Assembly A stats
        results = get_linear_features(df_A[col])
        for name, value in results:
            results_dict[f"{col}_{name}_A"] = value

        # Assembly B stats
        results = get_linear_features(df_B[col])
        for name, value in results:
            results_dict[f"{col}_{name}_B"] = value
    return i, results_dict

# Use multiprocessing to parallelize the computation
if __name__ == "__main__" or "ipykernel" in sys.modules:
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results = list(tqdm.tqdm(
    #         pool.imap(process_participant, range(num_participants)),
    #         total=num_participants,
    #         desc="Tabulating position and euler statistics"
    #     ))

    results = [process_participant(i) for i in tqdm.tqdm(range(num_participants), desc="Tabulating position and euler statistics")]
        
    # Update the tabulated_data dictionary with the results
    for i, result_dict in results:
        for key, value in result_dict.items():
            tabulated_data[key][i] = value

def calculate_velocity_and_acceleration_columns(idx, window):
    df_A = FAB_A_Complete_csvs[idx]
    df_B = FAB_B_Complete_csvs[idx]
    results_dict = {}
    
    for col in position_and_euler_columns:
        # Velocity calculation for assembly A
        vel_A = (df_A[col] - df_A[col].shift(window)) / (df_A["Timestamp"] - df_A["Timestamp"].shift(window))
        vel_col_name = f"{col.replace('position', 'linear_velocity').replace('euler', 'angular_velocity')}_{window}"
        df_A = df_A.with_columns(vel_A.alias(vel_col_name))
        
        # Velocity calculation for assembly B
        vel_B = (df_B[col] - df_B[col].shift(window)) / (df_B["Timestamp"] - df_B["Timestamp"].shift(window))
        df_B = df_B.with_columns(vel_B.alias(vel_col_name))
        
        # Acceleration calculation for assembly A
        acc_A = (vel_A - vel_A.shift(window)) / (df_A["Timestamp"] - df_A["Timestamp"].shift(window))
        acc_col_name = f"{col.replace('position', 'linear_acceleration').replace('euler', 'angular_acceleration')}_{window}"
        df_A = df_A.with_columns(acc_A.alias(acc_col_name))
        
        # Acceleration calculation for assembly B
        acc_B = (vel_B - vel_B.shift(window)) / (df_B["Timestamp"] - df_B["Timestamp"].shift(window))
        df_B = df_B.with_columns(acc_B.alias(acc_col_name))
    
    # Identify the new columns (those containing 'velocity' or 'acceleration')
    vel_acc_columns = [
        col for col in df_A.columns
        if ("velocity" in col or "acceleration" in col)
    ]
    
    # Calculate stats for these new columns
    for col in vel_acc_columns:
        # Assembly A stats
        results = get_linear_features(df_A[col])
        for name, value in results:
            results_dict[f"{col}_{name}_A"] = value
            
        # Assembly B stats
        results = get_linear_features(df_B[col])
        for name, value in results:
            results_dict[f"{col}_{name}_B"] = value
    
    return idx, results_dict

def tabulate_linear_and_angular_velocity_and_acceleration(window):
    if __name__ == "__main__" or "ipykernel" in sys.modules:
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     results = list(tqdm.tqdm(
        #         pool.starmap(
        #             calculate_velocity_and_acceleration_columns, 
        #             [(i, window) for i in range(num_participants)]
        #         ),
        #         total=num_participants,
        #         desc=f"Processing window size {window}"
        #     ))

        results = [calculate_velocity_and_acceleration_columns(i, window) for i in tqdm.tqdm(range(num_participants), desc=f"Processing window size {window}")]
        
        # Update the tabulated_data dictionary with the results
        for i, result_dict in results:
            for key, value in result_dict.items():
                tabulated_data[key][i] = value

# Process each window size with a progress bar
for window in tqdm.tqdm(window_sizes, desc="Tabulating velocity and acceleration"):
    tabulate_linear_and_angular_velocity_and_acceleration(window)

# Convert the tabulated data to a Polars DataFrame and write it to CSV
tabulated_dataframe = pl.DataFrame(tabulated_data)
# tabulated_dataframe.write_csv("tabulated_data.csv")

###############################################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

# for window_size in window_sizes:
#     y = FAB_A_Complete_csvs[43][f'RightHand_linear_velocity_x_{window_size}'].dropna()
#     x = np.linspace(0, len(y), len(y))
#     plt.figure(figsize=(24, 4))
#     plt.plot(x, y, label='Noisy Signal')
#     plt.title(f'Right Hand Linear Velocity X | Window Size = {window_size}')
#     plt.grid(lw=2,ls=':')
#     plt.xlabel('Time Step')
#     plt.ylabel("Value")
#     plt.legend()
#     plt.show()

# window_size = 30
# poly_order = 2
# y_smooth = savgol_filter(y, window_size, poly_order)

# plt.figure(figsize=(24, 4))  # Wider plot (width=12, height=4)
# plt.plot(x, y, label='Noisy Signal')
# plt.plot(x, y_smooth, label='Smoothed Signal', color='red')
# plt.grid(lw=2,ls=':')
# plt.xlabel('Time Step')
# plt.ylabel("Value")
# plt.legend()
# plt.show()
###############################################################################