import os, sys, time, pickle
import numpy as np
import polars as pl
from alt_analysis import *
import tqdm
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R, Slerp
mp.set_start_method("fork", force=True) # "spawn"
from scipy.signal import medfilt, savgol_filter
import numpy as np
from scipy.ndimage import uniform_filter1d  # For moving average smoothing

def truncate_data(df):
    rh_idx = df.with_row_index().filter(pl.col('RightHand_TriggerFloat_value') > 0).select('index').min().item()
    lh_idx = df.with_row_index().filter(pl.col('LeftHand_TriggerFloat_value') > 0).select('index').min().item()
    if rh_idx == None:
        return lh_idx
    elif lh_idx == None:
        return rh_idx
    else:
        return min(rh_idx, lh_idx)

def resample_data():
    # Directories
    assembly_A_dir = "/srv/STP/data/FAB/Assembly_A"
    assembly_B_dir = "/srv/STP/data/FAB/Assembly_B"
    Tracking_A_dir = "/srv/STP/data/FAB/Tracking_A"
    Tracking_B_dir = "/srv/STP/data/FAB/Tracking_B"
    output_dir_A = "/srv/STP/data/FAB/FAB_A_Resampled"
    output_dir_B = "/srv/STP/data/FAB/FAB_B_Resampled"
    # Get all files in the directories
    assembly_A_files = sorted(os.listdir(assembly_A_dir))
    assembly_B_files = sorted(os.listdir(assembly_B_dir))
    Tracking_A_files = sorted(os.listdir(Tracking_A_dir))
    Tracking_B_files = sorted(os.listdir(Tracking_B_dir))

    os.makedirs(output_dir_A, exist_ok=True)
    os.makedirs(output_dir_B, exist_ok=True)

    # Read CSVs using Polars
    assembly_A_csvs = [
        pl.read_csv(os.path.join(assembly_A_dir, file))
        for file in assembly_A_files if file.endswith('.csv')
    ]
    assembly_B_csvs = [
        pl.read_csv(os.path.join(assembly_B_dir, file))
        for file in assembly_B_files if file.endswith('.csv')
    ]
    Tracking_A_csvs = [
        pl.read_csv(os.path.join(Tracking_A_dir, file))
        for file in Tracking_A_files if file.endswith('.csv')
    ]
    Tracking_B_csvs = [
        pl.read_csv(os.path.join(Tracking_B_dir, file))
        for file in Tracking_B_files if file.endswith('.csv')
    ]

    assembly_csvs = [assembly_A_csvs, assembly_B_csvs]
    tracking_csvs = [Tracking_A_csvs, Tracking_B_csvs]

    file_names = [Tracking_A_files, Tracking_B_files]
    participant_ids = [f.split('_')[0] for f in Tracking_A_files]
    num_participants = len(participant_ids)
    
    # Resample the data to 0.01 sec intervals
    for idx, dataset in tqdm.tqdm(enumerate(tracking_csvs), total=len(tracking_csvs), desc="Resampling tracking data"):
        for i in range(len(dataset)):
            df = dataset[i]
            df = df.drop([col_name for col_name in df.columns if "euler" in col_name])
            
            # Truncate the data to the first trigger click
            start_idx = truncate_data(df)
            df = df.slice(start_idx, None)

            # Remove duplicate Timestamps (keep first occurrence)
            orig_height = df.height
            df = df.unique(subset=["Timestamp"], maintain_order=True)
            if df.height < orig_height:
                print(f"Warning: Found duplicates in {file_names[idx][i]}")
            
            # Get min and max Timestamp values
            min_ts = df.select(pl.col("Timestamp")).min().item()
            max_ts = df.select(pl.col("Timestamp")).max().item()
            
            # Create a new uniform timestamp range (0.01 sec intervals)
            new_index = np.arange(min_ts, max_ts + 1/90, 1/90)
            df_uniform = pl.DataFrame({"Timestamp": new_index})
            
            # Round timestamps to avoid floating point precision issues
            df_uniform = df_uniform.with_columns(pl.col("Timestamp").round(3))
            df = df.with_columns(pl.col("Timestamp").round(3))
            
            # Left join the original data to the uniform timestamps
            df_uniform = df_uniform.join(df, on="Timestamp", how="left")
            
            # Interpolate missing values in numeric columns (only for 'position' or 'euler' columns)
            cols_to_interp = [
                col for col in df_uniform.columns
                if ((col != "Timestamp") and (("position" in col) or ("quat" in col) or ("sixD" in col) or ("TriggerFloat_value" in col)))
            ]
            for col in cols_to_interp:
                # Using Polars' interpolate method on the Series (requires a recent version)
                interpolated = df_uniform[col].interpolate()
                df_uniform = df_uniform.with_columns(interpolated.alias(col))
            
            # Keep only the Timestamp and relevant columns
            cols = ["Timestamp"] + [col for col in df_uniform.columns if (("position" in col) or ("quat" in col) or ("sixD" in col) or ("TriggerFloat_value" in col))]
            df_uniform = df_uniform.select(cols)
            
            # Round all numeric columns to 6 decimal places
            numeric_cols = [col for col in df_uniform.columns if col != "Timestamp"]
            df_uniform = df_uniform.with_columns([
                pl.col(col).round(6) for col in numeric_cols
            ])
            
            dataset[i] = df_uniform.drop_nulls()

    for idx, dataset in enumerate(tracking_csvs):
        for i in range(len(dataset)):
            dataset[i].write_csv(os.path.join(output_dir_A if (idx == 0) else output_dir_B, file_names[idx][i]))

def generate_motion_columns(idx):
    assembly_A_dir = "/srv/STP/data/FAB/Assembly_A"
    assembly_B_dir = "/srv/STP/data/FAB/Assembly_B"
    FAB_A_Resampled_dir = "/srv/STP/data/FAB/FAB_A_Resampled"
    FAB_B_Resampled_dir = "/srv/STP/data/FAB/FAB_B_Resampled"
    FAB_A_Output_dir = "/srv/STP/data/FAB/FAB_A_Shift_Central"
    FAB_B_Output_dir = "/srv/STP/data/FAB/FAB_B_Shift_Central"
    assembly_A_files = sorted(os.listdir(assembly_A_dir))
    assembly_B_files = sorted(os.listdir(assembly_B_dir))
    FAB_A_Resampled_files = sorted(os.listdir(FAB_A_Resampled_dir))
    FAB_B_Resampled_files = sorted(os.listdir(FAB_B_Resampled_dir))

    os.makedirs(FAB_A_Output_dir, exist_ok=True)
    os.makedirs(FAB_B_Output_dir, exist_ok=True)

    # Read CSVs using Polars
    df_A = pl.read_csv(os.path.join(FAB_A_Resampled_dir, FAB_A_Resampled_files[idx]))
    df_B = pl.read_csv(os.path.join(FAB_B_Resampled_dir, FAB_B_Resampled_files[idx]))

    window_sizes = [450]
    stride = 450
    mode = "central"
    obj_list = ["Head", "LeftHand", "RightHand"]
    for obj in obj_list:
        pos_df_A = df_A[[f"{obj}_position_x", f"{obj}_position_y", f"{obj}_position_z"]].to_numpy()
        quat_df_A = df_A[[f"{obj}_quat_x", f"{obj}_quat_y", f"{obj}_quat_z", f"{obj}_quat_w"]].to_numpy()
        pos_df_B = df_B[[f"{obj}_position_x", f"{obj}_position_y", f"{obj}_position_z"]].to_numpy()
        quat_df_B = df_B[[f"{obj}_quat_x", f"{obj}_quat_y", f"{obj}_quat_z", f"{obj}_quat_w"]].to_numpy()
        timestamps_A = df_A["Timestamp"].to_numpy()
        timestamps_B = df_B["Timestamp"].to_numpy()
    
        for window in window_sizes:
            vel_x_A, vel_y_A, vel_z_A = finite_difference_linear(pos_df_A, timestamps_A, window, mode, stride)
            acc_x_A, acc_y_A, acc_z_A = finite_difference_linear_accel(pos_df_A, timestamps_A, window, mode, stride)
            jerk_x_A, jerk_y_A, jerk_z_A = finite_difference_linear_jerk(pos_df_A, timestamps_A, window, mode, stride)
            df_A = df_A.with_columns(
                pl.Series(name=f"{obj}_linvel_x_{window}", values=vel_x_A),
                pl.Series(name=f"{obj}_linvel_y_{window}", values=vel_y_A),
                pl.Series(name=f"{obj}_linvel_z_{window}", values=vel_z_A),
                pl.Series(name=f"{obj}_linacc_x_{window}", values=acc_x_A),
                pl.Series(name=f"{obj}_linacc_y_{window}", values=acc_y_A),
                pl.Series(name=f"{obj}_linacc_z_{window}", values=acc_z_A),
                pl.Series(name=f"{obj}_linjerk_x_{window}", values=jerk_x_A),
                pl.Series(name=f"{obj}_linjerk_y_{window}", values=jerk_y_A),
                pl.Series(name=f"{obj}_linjerk_z_{window}", values=jerk_z_A)
            )

            vel_x_A, vel_y_A, vel_z_A = finite_difference_angular(quat_df_A, timestamps_A, window, mode, stride)
            acc_x_A, acc_y_A, acc_z_A = finite_difference_angular_accel(quat_df_A, timestamps_A, window, mode, stride)
            jerk_x_A, jerk_y_A, jerk_z_A = finite_difference_angular_jerk(quat_df_A, timestamps_A, window, mode, stride)

            df_A = df_A.with_columns(
                pl.Series(name=f"{obj}_angvel_x_{window}", values=vel_x_A),
                pl.Series(name=f"{obj}_angvel_y_{window}", values=vel_y_A),
                pl.Series(name=f"{obj}_angvel_z_{window}", values=vel_z_A),
                pl.Series(name=f"{obj}_angacc_x_{window}", values=acc_x_A),
                pl.Series(name=f"{obj}_angacc_y_{window}", values=acc_y_A),
                pl.Series(name=f"{obj}_angacc_z_{window}", values=acc_z_A),
                pl.Series(name=f"{obj}_angjerk_x_{window}", values=jerk_x_A),
                pl.Series(name=f"{obj}_angjerk_y_{window}", values=jerk_y_A),
                pl.Series(name=f"{obj}_angjerk_z_{window}", values=jerk_z_A)
            )

            vel_x_B, vel_y_B, vel_z_B = finite_difference_linear(pos_df_B, timestamps_B, window, mode, stride)
            acc_x_B, acc_y_B, acc_z_B = finite_difference_linear_accel(pos_df_B, timestamps_B, window, mode, stride)
            jerk_x_B, jerk_y_B, jerk_z_B = finite_difference_linear_jerk(pos_df_B, timestamps_B, window, mode, stride)
            df_B = df_B.with_columns(
                pl.Series(name=f"{obj}_linvel_x_{window}", values=vel_x_B),
                pl.Series(name=f"{obj}_linvel_y_{window}", values=vel_y_B),
                pl.Series(name=f"{obj}_linvel_z_{window}", values=vel_z_B),
                pl.Series(name=f"{obj}_linacc_x_{window}", values=acc_x_B),
                pl.Series(name=f"{obj}_linacc_y_{window}", values=acc_y_B),
                pl.Series(name=f"{obj}_linacc_z_{window}", values=acc_z_B),
                pl.Series(name=f"{obj}_linjerk_x_{window}", values=jerk_x_B),
                pl.Series(name=f"{obj}_linjerk_y_{window}", values=jerk_y_B),
                pl.Series(name=f"{obj}_linjerk_z_{window}", values=jerk_z_B)
            )

            vel_x_B, vel_y_B, vel_z_B = finite_difference_angular(quat_df_B, timestamps_B, window, mode, stride)
            acc_x_B, acc_y_B, acc_z_B = finite_difference_angular_accel(quat_df_B, timestamps_B, window, mode, stride)
            jerk_x_B, jerk_y_B, jerk_z_B = finite_difference_angular_jerk(quat_df_B, timestamps_B, window, mode, stride)
            df_B = df_B.with_columns(
                pl.Series(name=f"{obj}_angvel_x_{window}", values=vel_x_B),
                pl.Series(name=f"{obj}_angvel_y_{window}", values=vel_y_B),
                pl.Series(name=f"{obj}_angvel_z_{window}", values=vel_z_B),
                pl.Series(name=f"{obj}_angacc_x_{window}", values=acc_x_B),
                pl.Series(name=f"{obj}_angacc_y_{window}", values=acc_y_B),
                pl.Series(name=f"{obj}_angacc_z_{window}", values=acc_z_B),
                pl.Series(name=f"{obj}_angjerk_x_{window}", values=jerk_x_B),
                pl.Series(name=f"{obj}_angjerk_y_{window}", values=jerk_y_B),
                pl.Series(name=f"{obj}_angjerk_z_{window}", values=jerk_z_B)
            )

    df_A.write_csv(os.path.join(FAB_A_Output_dir, FAB_A_Resampled_files[idx]))
    df_B.write_csv(os.path.join(FAB_B_Output_dir, FAB_B_Resampled_files[idx]))
    return idx

def process_participant(i):
    t = time.time()
    assembly_A_dir = "/srv/STP/data/FAB/Assembly_A"
    assembly_B_dir = "/srv/STP/data/FAB/Assembly_B"
    FAB_A_Motion_dir = "/srv/STP/data/FAB/FAB_A_Shift_Central"
    FAB_B_Motion_dir = "/srv/STP/data/FAB/FAB_B_Shift_Central"
    assembly_A_files = sorted(os.listdir(assembly_A_dir))
    assembly_B_files = sorted(os.listdir(assembly_B_dir))
    FAB_A_Motion_files = sorted(os.listdir(FAB_A_Motion_dir))
    FAB_B_Motion_files = sorted(os.listdir(FAB_B_Motion_dir))

    # Read CSVs using Polars
    assembly_dfs = [pl.read_csv(os.path.join(assembly_A_dir, assembly_A_files[i])), pl.read_csv(os.path.join(assembly_B_dir, assembly_B_files[i]))]
    dfs = [pl.read_csv(os.path.join(FAB_A_Motion_dir, FAB_A_Motion_files[i])), pl.read_csv(os.path.join(FAB_B_Motion_dir, FAB_B_Motion_files[i]))]
    build_letters = ["A", "B"]
    participant_id = FAB_A_Motion_files[i].split('_')[0]

    # print(f"Processing Participant {participant_id}")
    results_dict = {}
    
    for build_letter, df, assembly_df in zip(build_letters, dfs, assembly_dfs):

        # print(f"Processing Assembly A Position Features")
        # results = get_position_features(df, assembly_df)
        # for name, value in results.items():
        #     try:
        #         results_dict[f"{name}_{build_letter}"] = float(value)
        #     except Exception as e:
        #         print(f"Error processing {name} for {participant_id}: Value is {value}")

        # results = get_quat_features(df, assembly_df)
        # for name, value in results.items():
        #     try:
        #         results_dict[f"{name}_{build_letter}"] = float(value)
        #     except Exception as e:
        #         print(f"Error processing {name} for {participant_id}: Value is {value}")

        # results = get_sixD_features(df, assembly_df)
        # for name, value in results.items():
        #     try:
        #         results_dict[f"{name}_{build_letter}"] = float(value)
        #     except Exception as e:
        #         print(f"Error processing {name} for {participant_id}: Value is {value}")

        results = get_motion_features(df, assembly_df, [450])
        for name, value in results.items():
            try:
                results_dict[f"{name}_{build_letter}"] = float(value)
            except Exception as e:
                pass

    print(f"Completed processing {i} in {time.time() - t} seconds")
    return i, results_dict
 
# # Use multiprocessing to parallelize the computation
if __name__ == "__main__" or "ipykernel" in sys.modules:

    # Step 1: Resample the data (Should not need to be run again)
    ###############################################################################
    # resample_data()
    # sys.exit()

    # Step 2: Generate velocity and acceleration columns (If changing the window size, calculation method, or data types)
    ###############################################################################
    participant_ids = [i.split('_')[0] for i in sorted(os.listdir("/srv/STP/data/FAB/FAB_A_Resampled"))]
    num_participants = len(participant_ids)
    # results = [generate_motion_columns(i) for i in tqdm.tqdm(range(num_participants), desc="Generating motion columns")]
    
    num_cpus = 5 # mp.cpu_count()
    with mp.Pool(processes=num_cpus) as pool:
        results = list(tqdm.tqdm(
            pool.imap(generate_motion_columns, range(num_participants), chunksize=1),
            total=num_participants,
            desc="Generating motion columns"
        ))

    # Step 3: Tabulate the data
    # ###############################################################################
    t = time.time()
    # # Prepare tabulated data storage
    participant_ids = [i.split('_')[0] for i in sorted(os.listdir("/srv/STP/data/FAB/FAB_A_Shift_Central"))]
    num_participants = len(participant_ids)

    # sequences = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 105)]
    # counter = 1
    # for start, end in sequences:
    #     results = [process_participant(i) for i in tqdm.tqdm(range(start, end), desc="Tabulating feature statistics")]
    #     with open(f"/srv/STP/results_{counter}.pkl", "wb") as f:
    #         pickle.dump(results, f)
    #     counter += 1

    sequences = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 105)]
    counter = 1
    for start, end in sequences:
        num_cpus = 5
        with mp.Pool(processes=num_cpus) as pool:
            results = list(tqdm.tqdm(
                pool.imap(process_participant, range(start, end), chunksize=1),
                total=end-start,
                desc="Tabulating feature statistics"
            ))
        
        with open(f"/srv/STP/results_{counter}.pkl", "wb") as f:
            pickle.dump(results, f)
        
        counter += 1