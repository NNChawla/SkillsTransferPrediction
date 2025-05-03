import numpy as np
import polars as pl
from pyinform import mutual_info
from nolitsa import dimension
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis
import tsfel
import nolds
import itertools, time
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

def get_step_dfs(tracking_df, assembly_df):
    tracking_step_dfs = []
    tracking_substep_dfs = []
    tracking_substep_type_dfs = {}
    for substep_type in assembly_df['subStep'].unique():
        tracking_substep_type_dfs[substep_type] = []
    
    for step in range(assembly_df['step'].max() + 1):
        assembly_step_df = assembly_df.filter(pl.col('step') == step)
        assembly_previous_step_df = None if (step == 0) else assembly_df.filter(pl.col('step') == (step - 1))
        start_time = tracking_df['Timestamp'].min() if (step == 0) else assembly_previous_step_df['Timestamp'].max()
        end_time = assembly_step_df['Timestamp'].max()
        start_idx = (tracking_df["Timestamp"] - start_time).abs().arg_min()
        end_idx = (tracking_df["Timestamp"] - end_time).abs().arg_min()
        tracking_step_df = tracking_df[start_idx:end_idx]
        tracking_step_dfs.append(tracking_step_df)

        for substep_idx in range(len(assembly_step_df)):
            if (step == 0) and (substep_idx == 0):
                start_time = tracking_df['Timestamp'].min()
            elif (substep_idx == 0):
                start_time = assembly_previous_step_df['Timestamp'].max()
            else:
                start_time = assembly_step_df['Timestamp'][substep_idx - 1]
            end_time = assembly_step_df['Timestamp'][substep_idx]
            start_idx = (tracking_step_df["Timestamp"] - start_time).abs().arg_min()
            end_idx = (tracking_step_df["Timestamp"] - end_time).abs().arg_min()
            tracking_substep_df = tracking_step_df[start_idx:end_idx]
            tracking_substep_dfs.append(tracking_substep_df)
            total_substep_idx = len(assembly_step_df) * step + substep_idx
            tracking_substep_type_dfs[assembly_step_df['subStep'][substep_idx]].append(total_substep_idx)
    
    start_time = assembly_df['Timestamp'].max()
    start_idx = (tracking_df["Timestamp"] - start_time).abs().arg_min()
    end_idx = len(tracking_df)
    tracking_end_df = tracking_df[start_idx:end_idx]

    return tracking_step_dfs, tracking_substep_dfs, tracking_substep_type_dfs, tracking_end_df

def get_duration_features(tracking_df, assembly_df):
    duration_features = {}
    tracking_step_dfs, tracking_substep_dfs, tracking_substep_type_dfs, tracking_end_df = get_step_dfs(tracking_df, assembly_df)
    step_durations = []
    substep_durations = []

    # Variables
    duration_features['session_duration'] = tracking_df["Timestamp"].max()
    duration_features['session_end_duration'] = tracking_end_df["Timestamp"].max() - tracking_end_df["Timestamp"].min()

    for step in range(len(tracking_step_dfs) - 1): # last step is not included as it is imbalanced
        step_duration = tracking_step_dfs[step]["Timestamp"].max() - tracking_step_dfs[step]["Timestamp"].min()
        duration_features[f'step{step}_duration'] = step_duration
        step_durations.append(step_duration)

    duration_features['step_of_max_duration'] = step_durations.index(max(step_durations))
    duration_features['step_of_min_duration'] = step_durations.index(min(step_durations))

    for substep_idx in range(len(tracking_substep_dfs)):
        try:
            substep_duration = tracking_substep_dfs[substep_idx]["Timestamp"].max() - tracking_substep_dfs[substep_idx]["Timestamp"].min()
            duration_features[f'substep{substep_idx}_duration'] = substep_duration
            substep_durations.append(substep_duration)
        except:
            duration_features[f'substep{substep_idx}_duration'] = None
            substep_durations.append(None)

    non_none_substep_durations = [i for i in substep_durations if i is not None]
    
    duration_features['substep_of_max_duration'] = substep_durations.index(max(non_none_substep_durations))
    duration_features['substep_of_min_duration'] = substep_durations.index(min(non_none_substep_durations))
    
    # Feature Vectors
    duration_features.update(get_stat_features(step_durations, prefix='step_duration'))
    duration_features.update(get_stat_features(substep_durations, prefix='substep_duration'))

    for substep_type in sorted(tracking_substep_type_dfs.keys()):
        substep_type_durations = [duration_features[f'substep{substep_idx}_duration'] for substep_idx in tracking_substep_type_dfs[substep_type]]
        # print(f"Substep type: {substep_type} | indices {tracking_substep_type_dfs[substep_type]}")
        non_none_substep_type_durations = [i for i in substep_type_durations if i is not None]
        duration_features.update(get_stat_features(non_none_substep_type_durations, prefix=f'{substep_type}_duration'))
    
    return duration_features

def get_position_features(tracking_df, assembly_df):
    position_features = {}
    tracking_step_dfs, tracking_substep_dfs, tracking_substep_type_dfs, tracking_end_df = get_step_dfs(tracking_df, assembly_df)
    
    obj_list = ['Head', 'LeftHand', 'RightHand']
    obj_pair_list = list(itertools.combinations(obj_list, 2))
    unique_position_features = ['bbox', 'conv_hull', 'total_dist_traveled', 'path_efficiency', 'net_displacement']

    column_parameter_list = []

    for obj in obj_list:
        column_name = f"{obj}_position_axis"
        column_parameter_list.append(column_name)

    # Session level
    for col_name in column_parameter_list:
        axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z')]
        session_df = tracking_df[axis_names]
        position_features.update(get_custom_features_position(session_df, tracking_df['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'session_{obj}', fs=90, stat_features=False))

    for obj_pair in obj_pair_list:
        obj1_df = tracking_df[[f'{obj_pair[0]}_position_x', f'{obj_pair[0]}_position_y', f'{obj_pair[0]}_position_z']]
        obj2_df = tracking_df[[f'{obj_pair[1]}_position_x', f'{obj_pair[1]}_position_y', f'{obj_pair[1]}_position_z']]
        distance_between_objects = calculate_distance_between_objects_3D(obj1_df, obj2_df)
        position_features.update(get_stat_features(distance_between_objects, prefix=f'{obj_pair[0]}_dist_to_{obj_pair[1]}'))

        axes = [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'y'), ('y', 'z'), ('z', 'z')]
        for axis in axes:
            obj1_column_name = f"{obj_pair[0]}_position_{axis[0]}"
            obj2_column_name = f"{obj_pair[1]}_position_{axis[1]}"
            data = calculate_distance_between_objects_1D(tracking_df[obj1_column_name].to_numpy(), tracking_df[obj2_column_name].to_numpy())
            position_features.update(get_stat_features(data, prefix=f'{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'))
    
    # End level
    for col_name in column_parameter_list:
        axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z')]
        end_df = tracking_end_df[axis_names]
        position_features.update(get_custom_features_position(end_df, tracking_end_df['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'end_{obj}', fs=90, stat_features=True))
    
    for obj_pair in obj_pair_list:
        obj1_df = tracking_end_df[[f'{obj_pair[0]}_position_x', f'{obj_pair[0]}_position_y', f'{obj_pair[0]}_position_z']]
        obj2_df = tracking_end_df[[f'{obj_pair[1]}_position_x', f'{obj_pair[1]}_position_y', f'{obj_pair[1]}_position_z']]
        distance_between_objects = calculate_distance_between_objects_3D(obj1_df, obj2_df)
        position_features.update(get_stat_features(distance_between_objects, prefix=f'end_{obj_pair[0]}_dist_to_{obj_pair[1]}'))

        axes = [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'y'), ('y', 'z'), ('z', 'z')]
        for axis in axes:
            obj1_column_name = f"{obj_pair[0]}_position_{axis[0]}"
            obj2_column_name = f"{obj_pair[1]}_position_{axis[1]}"
            data = calculate_distance_between_objects_1D(tracking_end_df[obj1_column_name].to_numpy(), tracking_end_df[obj2_column_name].to_numpy())
            position_features.update(get_stat_features(data, prefix=f'end_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'))

    # Step level
    for step in range(len(tracking_step_dfs) - 1): # last step is not included as it is imbalanced
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z')]
            step_df = tracking_step_dfs[step][axis_names]
            position_features.update(get_custom_features_position(step_df, tracking_step_dfs[step]['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'step{step}_{obj}', fs=90, stat_features=True))

        for obj_pair in obj_pair_list:
            obj1_df = tracking_step_dfs[step][[f'{obj_pair[0]}_position_x', f'{obj_pair[0]}_position_y', f'{obj_pair[0]}_position_z']]
            obj2_df = tracking_step_dfs[step][[f'{obj_pair[1]}_position_x', f'{obj_pair[1]}_position_y', f'{obj_pair[1]}_position_z']]
            distance_between_objects = calculate_distance_between_objects_3D(obj1_df, obj2_df)
            position_features.update(get_stat_features(distance_between_objects, prefix=f'step{step}_{obj_pair[0]}_dist_to_{obj_pair[1]}'))

            axes = [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'y'), ('y', 'z'), ('z', 'z')]
            for axis in axes:
                obj1_column_name = f"{obj_pair[0]}_position_{axis[0]}"
                obj2_column_name = f"{obj_pair[1]}_position_{axis[1]}"
                data = calculate_distance_between_objects_1D(tracking_step_dfs[step][obj1_column_name].to_numpy(), tracking_step_dfs[step][obj2_column_name].to_numpy())
                position_features.update(get_stat_features(data, prefix=f'step{step}_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'))

    for obj in obj_list:
        for feature in unique_position_features:
            step_list = [position_features.get(f'step{step}_{obj}_{feature}', None) for step in range(len(tracking_step_dfs) - 1)]
            non_none_step_list = [i for i in step_list if i is not None]
            if (len(non_none_step_list) == 0):
                continue
            position_features[f'step_of_max_{feature}_{obj}'] = step_list.index(max(non_none_step_list))
            position_features[f'step_of_min_{feature}_{obj}'] = step_list.index(min(non_none_step_list))
            position_features.update(get_stat_features(non_none_step_list, prefix=f'steps_{feature}_{obj}'))

    # Substep level
    for substep_idx in range(len(tracking_substep_dfs)):
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z')]
            substep_df = tracking_substep_dfs[substep_idx][axis_names]
            position_features.update(get_custom_features_position(substep_df, tracking_substep_dfs[substep_idx]['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'substep{substep_idx}_{obj}', fs=90, stat_features=True))

        for obj_pair in obj_pair_list:
            obj1_df = tracking_substep_dfs[substep_idx][[f'{obj_pair[0]}_position_x', f'{obj_pair[0]}_position_y', f'{obj_pair[0]}_position_z']]
            obj2_df = tracking_substep_dfs[substep_idx][[f'{obj_pair[1]}_position_x', f'{obj_pair[1]}_position_y', f'{obj_pair[1]}_position_z']]
            distance_between_objects = calculate_distance_between_objects_3D(obj1_df, obj2_df)
            position_features.update(get_stat_features(distance_between_objects, prefix=f'substep{substep_idx}_{obj_pair[0]}_dist_to_{obj_pair[1]}'))

            axes = [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'y'), ('y', 'z'), ('z', 'z')]
            for axis in axes:
                obj1_column_name = f"{obj_pair[0]}_position_{axis[0]}"
                obj2_column_name = f"{obj_pair[1]}_position_{axis[1]}"
                data = calculate_distance_between_objects_1D(tracking_substep_dfs[substep_idx][obj1_column_name].to_numpy(), tracking_substep_dfs[substep_idx][obj2_column_name].to_numpy())
                position_features.update(get_stat_features(data, prefix=f'substep{substep_idx}_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'))

    for obj in obj_list:
        for feature in unique_position_features:
            substep_list = [position_features.get(f'substep{substep_idx}_{obj}_{feature}', None) for substep_idx in range(len(tracking_substep_dfs))]
            non_none_substep_list = [i for i in substep_list if i is not None]
            if (len(non_none_substep_list) == 0):
                continue
            position_features[f'substep_of_max_{feature}_{obj}'] = substep_list.index(max(non_none_substep_list))
            position_features[f'substep_of_min_{feature}_{obj}'] = substep_list.index(min(non_none_substep_list))
            position_features.update(get_stat_features(non_none_substep_list, prefix=f'substeps_{feature}_{obj}'))
    
    # Substep type level
    for substep_type in sorted(tracking_substep_type_dfs.keys()):
        for obj in obj_list:
            for feature in unique_position_features:
                substep_list = [position_features.get(f'substep{substep_idx}_{obj}_{feature}', None) for substep_idx in tracking_substep_type_dfs[substep_type]]
                non_none_substep_list = [i for i in substep_list if i is not None]
                if (len(non_none_substep_list) == 0):
                    continue
                position_features.update(get_stat_features(non_none_substep_list, prefix=f'{substep_type}_{feature}_{obj}'))

    return position_features

def get_quat_features(tracking_df, assembly_df):
    quat_features = {}
    tracking_step_dfs, tracking_substep_dfs, tracking_substep_type_dfs, tracking_end_df = get_step_dfs(tracking_df, assembly_df)

    obj_list = ['Head', 'LeftHand', 'RightHand']
    unique_quat_features = ['total_rotation_traveled']#, 'quaternion_dispersion']

    column_parameter_list = []

    for obj in obj_list:
        column_name = f"{obj}_quat_axis"
        column_parameter_list.append(column_name)

    # Session level
    for col_name in column_parameter_list:
        axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z'), col_name.replace('axis', 'w')]
        session_df = tracking_df[axis_names].to_numpy()
        quat_features.update(get_custom_features_quat(session_df, tracking_df['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'session_{obj}', fs=90, stat_features=False))

    # End level
    for col_name in column_parameter_list:
        axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z'), col_name.replace('axis', 'w')]
        obj_df = tracking_end_df[axis_names].to_numpy()
        quat_features.update(get_custom_features_quat(obj_df, tracking_end_df['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'end_{obj}', fs=90, stat_features=True))

    # Step level
    for step in range(len(tracking_step_dfs) - 1): # last step is not included as it is imbalanced
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z'), col_name.replace('axis', 'w')]
            obj_df = tracking_step_dfs[step][axis_names].to_numpy()
            quat_features.update(get_custom_features_quat(obj_df, tracking_step_dfs[step]['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'step{step}_{obj}', fs=90, stat_features=True))

    for obj in obj_list:
        for feature in unique_quat_features:
            step_list = [quat_features.get(f'step{step}_{obj}_{feature}', None) for step in range(len(tracking_step_dfs) - 1)]
            non_none_step_list = [i for i in step_list if i is not None]
            if (len(non_none_step_list) == 0):
                continue
            quat_features[f'step_of_max_{feature}_{obj}'] = step_list.index(max(non_none_step_list))
            quat_features[f'step_of_min_{feature}_{obj}'] = step_list.index(min(non_none_step_list))
            quat_features.update(get_stat_features(non_none_step_list, prefix=f'steps_{feature}_{obj}'))
                
    # Substep level
    for substep_idx in range(len(tracking_substep_dfs)):
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z'), col_name.replace('axis', 'w')]
            obj_df = tracking_substep_dfs[substep_idx][axis_names].to_numpy()
            quat_features.update(get_custom_features_quat(obj_df, tracking_substep_dfs[substep_idx]['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'substep{substep_idx}_{obj}', fs=90, stat_features=True))
        
    for obj in obj_list:
        for feature in unique_quat_features:
            substep_list = [quat_features.get(f'substep{substep_idx}_{obj}_{feature}', None) for substep_idx in range(len(tracking_substep_dfs))]
            non_none_substep_list = [i for i in substep_list if i is not None]
            if (len(non_none_substep_list) == 0):
                continue
            quat_features[f'substep_of_max_{feature}_{obj}'] = substep_list.index(max(non_none_substep_list))
            quat_features[f'substep_of_min_{feature}_{obj}'] = substep_list.index(min(non_none_substep_list))
            quat_features.update(get_stat_features(non_none_substep_list, prefix=f'substeps_{feature}_{obj}'))
    
    # Substep type level
    for substep_type in sorted(tracking_substep_type_dfs.keys()):
        for obj in obj_list:
            for feature in unique_quat_features:
                substep_list = [quat_features.get(f'substep{substep_idx}_{obj}_{feature}', None) for substep_idx in tracking_substep_type_dfs[substep_type]]
                non_none_substep_list = [i for i in substep_list if i is not None]
                if (len(non_none_substep_list) == 0):
                    continue
                quat_features.update(get_stat_features(non_none_substep_list, prefix=f'{substep_type}_{feature}_{obj}'))

    return quat_features

def get_sixD_features(tracking_df, assembly_df):
    sixD_features = {}
    tracking_step_dfs, tracking_substep_dfs, tracking_substep_type_dfs, tracking_end_df = get_step_dfs(tracking_df, assembly_df)

    obj_list = ['Head', 'LeftHand', 'RightHand']
    column_parameter_list = []

    for obj in obj_list:
        column_name = f"{obj}_sixD_axis"
        column_parameter_list.append(column_name)

    # Session level
    for col_name in column_parameter_list:
        axis_names = [col_name.replace('axis', 'a'), col_name.replace('axis', 'b'), col_name.replace('axis', 'c'), col_name.replace('axis', 'd'), col_name.replace('axis', 'e'), col_name.replace('axis', 'f')]
        for axis_name in axis_names:
            sixD_features.update(get_custom_features(tracking_df[axis_name].to_numpy(), tracking_df['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'session_{axis_name}', fs=90))

    # End level
    for col_name in column_parameter_list:
        axis_names = [col_name.replace('axis', 'a'), col_name.replace('axis', 'b'), col_name.replace('axis', 'c'), col_name.replace('axis', 'd'), col_name.replace('axis', 'e'), col_name.replace('axis', 'f')]
        for axis_name in axis_names:
            sixD_features.update(get_stat_features(tracking_end_df[axis_name].to_numpy(), prefix=f'end_{axis_name}'))

    # Step level
    for step in range(len(tracking_step_dfs) - 1): # last step is not included as it is imbalanced
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'a'), col_name.replace('axis', 'b'), col_name.replace('axis', 'c'), col_name.replace('axis', 'd'), col_name.replace('axis', 'e'), col_name.replace('axis', 'f')]
            for axis_name in axis_names:
                sixD_features.update(get_stat_features(tracking_step_dfs[step][axis_name].to_numpy(), prefix=f'step{step}_{axis_name}'))
                
    # Substep level
    for substep_idx in range(len(tracking_substep_dfs)): # last step is not included as it is imbalanced
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'a'), col_name.replace('axis', 'b'), col_name.replace('axis', 'c'), col_name.replace('axis', 'd'), col_name.replace('axis', 'e'), col_name.replace('axis', 'f')]
            for axis_name in axis_names:
                sixD_features.update(get_stat_features(tracking_substep_dfs[substep_idx][axis_name].to_numpy(), prefix=f'substep{substep_idx}_{axis_name}'))

    for substep_type in sorted(tracking_substep_type_dfs.keys()):
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'a'), col_name.replace('axis', 'b'), col_name.replace('axis', 'c'), col_name.replace('axis', 'd'), col_name.replace('axis', 'e'), col_name.replace('axis', 'f')]
            for axis_name in axis_names:
                substep_list = [sixD_features.get(f'substep{substep_idx}_{axis_name}', None) for substep_idx in tracking_substep_type_dfs[substep_type]]
                non_none_substep_list = [i for i in substep_list if i is not None]
                if (len(non_none_substep_list) == 0):
                    continue
                sixD_features.update(get_stat_features(non_none_substep_list, prefix=f'{substep_type}_{axis_name}'))

    return sixD_features

def get_trigger_features(tracking_df):
    trigger_features = {}
    column_parameter_list = ['LeftHand_TriggerFloat_value', 'RightHand_TriggerFloat_value']

    # Session level
    for col_name in column_parameter_list:
        trigger_active_intervals = find_nonzero_trigger_intervals(tracking_df[col_name].to_numpy())
        trigger_inactive_intervals = find_zero_trigger_intervals(tracking_df[col_name].to_numpy())
        timestamps = tracking_df['Timestamp'].to_numpy()
        trigger_features.update(get_custom_features_interval(trigger_active_intervals, timestamps, prefix=f'{col_name}_active'))
        trigger_features.update(get_custom_features_interval(trigger_inactive_intervals, timestamps, prefix=f'{col_name}_inactive'))
    return trigger_features

def get_motion_features(tracking_df, assembly_df):
    motion_features = {}
    tracking_step_dfs, tracking_substep_dfs, tracking_substep_type_dfs, tracking_end_df = get_step_dfs(tracking_df, assembly_df)

    obj_list = ['Head', 'LeftHand', 'RightHand']
    unique_motion_features = ['directional_reversal_frequency', 'num_reversals', 'cumulative_opposing_displacement']

    column_parameter_list = []
    window_sizes = [9, 91, 181, 271, 361, 451, 541]
    feature_types = ['linvel', 'linacc', 'angvel', 'angacc']
    for obj in obj_list:
        for feature_type in feature_types:
            for window_size in window_sizes:
                column_name = f"{obj}_{feature_type}_axis_{window_size}"
                column_parameter_list.append(column_name)

    # Session level
    for col_name in column_parameter_list:
        axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z')]
        pos_axis_names = [axis_name.split('_')[0] + '_position_' + axis_name.split('_')[2] for axis_name in axis_names]
        session_df = tracking_df[axis_names].to_numpy()
        session_pos_df = tracking_df[pos_axis_names].to_numpy()
        motion_features.update(get_custom_features_motion(session_df, session_pos_df, tracking_df['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'session_{col_name}', fs=90, stat_features=False))

    # End level
    for col_name in column_parameter_list:
        axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z')]
        pos_axis_names = [axis_name.split('_')[0] + '_position_' + axis_name.split('_')[2] for axis_name in axis_names]
        obj_df = tracking_end_df[axis_names].to_numpy()
        obj_pos_df = tracking_end_df[pos_axis_names].to_numpy()
        motion_features.update(get_custom_features_motion(obj_df, obj_pos_df, tracking_end_df['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'end_{col_name}', fs=90, stat_features=True))

    # Step level
    for step in range(len(tracking_step_dfs) - 1): # last step is not included as it is imbalanced
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z')]
            pos_axis_names = [axis_name.split('_')[0] + '_position_' + axis_name.split('_')[2] for axis_name in axis_names]
            obj_df = tracking_step_dfs[step][axis_names].to_numpy()
            obj_pos_df = tracking_step_dfs[step][pos_axis_names].to_numpy()
            motion_features.update(get_custom_features_motion(obj_df, obj_pos_df, tracking_step_dfs[step]['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'step{step}_{col_name}', fs=90, stat_features=True))

    for col_name in column_parameter_list:
        for feature in unique_motion_features:
            step_list = [motion_features.get(f'step{step}_{col_name}_{feature}', None) for step in range(len(tracking_step_dfs) - 1)]
            non_none_step_list = [i for i in step_list if i is not None]
            if (len(non_none_step_list) == 0):
                continue
            motion_features[f'step_of_max_{feature}_{col_name}'] = step_list.index(max(non_none_step_list))
            motion_features[f'step_of_min_{feature}_{col_name}'] = step_list.index(min(non_none_step_list))
            motion_features.update(get_stat_features(non_none_step_list, prefix=f'steps_{feature}_{col_name}'))
                
    # Substep level
    for substep_idx in range(len(tracking_substep_dfs)):
        for col_name in column_parameter_list:
            axis_names = [col_name.replace('axis', 'x'), col_name.replace('axis', 'y'), col_name.replace('axis', 'z')]
            pos_axis_names = [axis_name.split('_')[0] + '_position_' + axis_name.split('_')[2] for axis_name in axis_names]
            obj_df = tracking_substep_dfs[substep_idx][axis_names].to_numpy()
            obj_pos_df = tracking_substep_dfs[substep_idx][pos_axis_names].to_numpy()
            motion_features.update(get_custom_features_motion(obj_df, obj_pos_df, tracking_substep_dfs[substep_idx]['Timestamp'].to_numpy(), assembly_df['Timestamp'].to_numpy(), assembly_df['step'].to_numpy(), assembly_df['subStep'].to_numpy(), prefix=f'substep{substep_idx}_{col_name}', fs=90, stat_features=True))
        
    for col_name in column_parameter_list:
        for feature in unique_motion_features:
            substep_list = [motion_features.get(f'substep{substep_idx}_{col_name}_{feature}', None) for substep_idx in range(len(tracking_substep_dfs))]
            non_none_substep_list = [i for i in substep_list if i is not None]
            if (len(non_none_substep_list) == 0):
                continue
            motion_features[f'substep_of_max_{feature}_{col_name}'] = substep_list.index(max(non_none_substep_list))
            motion_features[f'substep_of_min_{feature}_{col_name}'] = substep_list.index(min(non_none_substep_list))
            motion_features.update(get_stat_features(non_none_substep_list, prefix=f'substeps_{feature}_{col_name}'))
    
    # Substep type level
    for substep_type in sorted(tracking_substep_type_dfs.keys()):
        for col_name in column_parameter_list:
            for feature in unique_motion_features:
                substep_list = [motion_features.get(f'substep{substep_idx}_{col_name}_{feature}', None) for substep_idx in tracking_substep_type_dfs[substep_type]]
                non_none_substep_list = [i for i in substep_list if i is not None]
                if (len(non_none_substep_list) == 0):
                    continue
                motion_features.update(get_stat_features(non_none_substep_list, prefix=f'{substep_type}_{feature}_{col_name}'))

    return motion_features

def get_linear_features(input_data, prefix='', fs=90):
    cfg_file = tsfel.get_features_by_domain()
    try:
        features = tsfel.time_series_features_extractor(cfg_file, input_data, fs=fs, verbose=0, n_jobs=1)
        features = features.to_dict()
        features = [(f"{prefix}_{i.replace('0_', '').replace(' ', '_')}", j[0]) for i,j in features.items()]
    except Exception as e:
        features = tsfel.time_series_features_extractor(cfg_file, np.arange(1000), fs=fs, verbose=0, n_jobs=1)
        features = features.to_dict()
        features = [(f"{prefix}_{i.replace('0_', '').replace(' ', '_')}", np.nan) for i,j in features.items()]
        print(f"Error: {e}")
    return features

def calculate_coefficient_of_variation(data):
    result = np.std(data) / np.mean(data)
    return result

def slope_of_linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))  # Covariance
    denominator = np.sum((x - x_mean) ** 2)           # Variance

    if denominator == 0:
        return np.nan

    slope = numerator / denominator
    return slope

def calculate_total_distance_traveled_1D(data):
    return np.sum(np.abs(np.diff(data)))

def calculate_total_distance_traveled_3D(data):
    return np.sum(np.linalg.norm(np.diff(data, axis=0), axis=1))

def calculate_bounding_box_volume(points):
    """Compute the axis-aligned bounding box volume of a set of 3D points."""
    min_vals = points.min()  # Minimum along x, y, z
    max_vals = points.max()  # Maximum along x, y, z
    ranges = max_vals - min_vals       # Range for each dimension
    return np.prod(ranges)             # Volume = (x_range * y_range * z_range)

def calculate_convex_hull_volume(points):
    """Compute the volume of the convex hull of a set of 3D points."""
    try:
        hull = ConvexHull(points, qhull_options='QJ')
        return hull.volume
    except Exception as e:
        return np.nan

def calculate_distance_between_objects_3D(obj1_df, obj2_df):
    return np.linalg.norm(obj2_df - obj1_df, axis=1)

def calculate_distance_between_objects_1D(obj1_df, obj2_df):
    return np.abs(obj2_df - obj1_df)

def get_finite_differences(df, col, window):
    finite_difference = pl.Series(df[col] - df[col].shift(window)) / (df["Timestamp"] - df["Timestamp"].shift(window))
    return finite_difference

def get_vector_magnitude(vec_x, vec_y, vec_z):
    vec_3D = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2)
    return vec_3D

def get_extremum_of_vector(vector, timestamps, extremum_type='max'):
    if extremum_type == 'max':
        ex_idx = int(np.argmax(vector))
    elif extremum_type == 'min':
        ex_idx = int(np.argmin(vector))
    ex_val = vector[ex_idx]
    ex_time = timestamps[ex_idx]
    return ex_val, ex_time

def duration_extremum_vector(vector, timestamps, threshold: float, extremum_type: str):
    """
    Returns the total duration (in time units) during which vector exceeds the threshold.
    
    This function detects contiguous segments where the velocity is above `threshold` and
    sums the durations of these segments.
    """
    if extremum_type == 'max':
        extremum_mask = vector > threshold
    elif extremum_type == 'min':
        extremum_mask = vector < threshold

    if not np.any(extremum_mask):
        return 0.0

    # Compute transitions in the high_mask.
    diff = np.diff(extremum_mask.astype(int))
    # Start indices: where mask goes from False to True.
    starts = np.where(diff == 1)[0] + 1
    # If the very first point is high, include index 0.
    if extremum_mask[0]:
        starts = np.insert(starts, 0, 0)
    # End indices: where mask goes from True to False.
    ends = np.where(diff == -1)[0] + 1
    # If the last point is high, include the final index.
    if extremum_mask[-1]:
        ends = np.append(ends, len(vector))
    
    # Sum the duration for each high-velocity segment.
    # (Assumes that the Timestamp column is sorted.)
    durations = timestamps[ends - 1] - timestamps[starts]
    total_duration = np.sum(durations)
    return total_duration

def find_threshold_intervals(vector, threshold_high: float, threshold_low: float):
    """
    Finds intervals where vector is between threshold_low and threshold_high.
    
    Args:
        vector: numpy array of values to analyze
        threshold_high: upper threshold value
        threshold_low: lower threshold value
        
    Returns:
        List of tuples containing (start_index, end_index) for each interval
        where vector is between thresholds
    """
    # Find points between thresholds
    between_mask = (vector >= threshold_low) & (vector <= threshold_high)
    
    if not np.any(between_mask):
        return []

    # Find transitions
    transitions = np.diff(between_mask.astype(int))
    
    # Start indices: where mask goes from False to True
    starts = np.where(transitions == 1)[0] + 1
    
    # If first point is between thresholds, add index 0
    if between_mask[0]:
        starts = np.insert(starts, 0, 0)
        
    # End indices: where mask goes from True to False
    ends = np.where(transitions == -1)[0] + 1
    
    # If last point is between thresholds, add final index
    if between_mask[-1]:
        ends = np.append(ends, len(vector))

    # Return list of (start,end) index tuples
    return list(zip(starts, ends))


def find_extrema(vector, threshold: float, extremum_type: str):
    """
    A simple peak finder: returns indices of local maxima that are above the given threshold.
    
    A peak is defined as a point that is greater than its immediate neighbors.
    """
    # For indices 1 through len(v)-2, check if v[i] is a local maximum.
    if extremum_type == 'max':
        extrema = np.where((vector[1:-1] > vector[:-2]) & (vector[1:-1] > vector[2:]) & (vector[1:-1] > threshold))[0] + 1
    elif extremum_type == 'min':
        extrema = np.where((vector[1:-1] < vector[:-2]) & (vector[1:-1] < vector[2:]) & (vector[1:-1] < threshold))[0] + 1
    return extrema

def frequency_of_extrema_crossing_threshold(vector, timestamps, threshold: float, extremum_type: str):
    """
    Returns the frequency of peaks (peaks per time unit) above a given threshold,
    and also returns the indices of the detected peaks.
    """
    extrema = find_extrema(vector, threshold, extremum_type)
    
    if len(timestamps) < 2:
        return 0, extrema
    total_time = timestamps[-1] - timestamps[0]
    frequency = len(extrema) / total_time if total_time > 0 else 0
    return frequency, extrema

def inter_extrema_intervals(timestamps, extrema: np.ndarray):
    """
    Computes the array of time differences between consecutive peaks and the average interval.
    Returns (intervals, average_interval). If fewer than two peaks are found, returns (None, None).
    """
    if len(extrema) < 2:
        return None, None
    intervals = np.diff(timestamps[extrema])
    avg_interval = np.mean(intervals)
    return intervals, avg_interval

def time_to_extrema(vector, timestamps, extremum_type: str):
    """
    Computes the time from the start (first timestamp) to the maximum velocity.
    """
    if extremum_type == 'max':
        max_idx = int(np.argmax(vector))
    elif extremum_type == 'min':
        max_idx = int(np.argmin(vector))
    time_to_extrema = timestamps[max_idx] - timestamps[0]
    return time_to_extrema

def decay_time(vector, timestamps, decay_threshold: float, extremum_type: str):
    """
    Computes the time it takes for the velocity to fall below a specified decay_threshold
    after the peak.
    
    Returns the difference between the time at the peak and the first subsequent time
    when velocity falls below decay_threshold. If no such time is found, returns None.
    """
    if extremum_type == 'max':
        max_idx = int(np.argmax(vector))
    elif extremum_type == 'min':
        max_idx = int(np.argmin(vector))
    # Search from the peak forward.
    for i in range(max_idx, len(vector)):
        if extremum_type == 'max' and vector[i] < decay_threshold:
            return timestamps[i] - timestamps[max_idx]
        elif extremum_type == 'min' and vector[i] > decay_threshold:
            return timestamps[i] - timestamps[max_idx]
    return None

# -----------------------------------------------------------------------------
# 1. Directional Reversal Frequency
# -----------------------------------------------------------------------------
def directional_reversal_frequency(velocity_x, velocity_y, velocity_z, timestamps, time_window: float = None):
    """
    Computes the number of directional reversals based on the cosine similarity 
    between consecutive velocity vectors. A reversal is detected when the cosine 
    similarity is negative (i.e. the angle between consecutive velocity vectors is > 90°).

    Parameters:
        df: A Polars DataFrame with columns "Timestamp", "Velocity_x", "Velocity_y", "Velocity_z"
        time_window (optional): If provided, only reversals within the first `time_window` seconds are counted.

    Returns:
        reversals: The total count of directional reversals.
        reversal_frequency: Reversals per unit time (if a time window is defined, otherwise over the whole duration).
    """
    # Extract velocity vectors and timestamps
    v = pl.DataFrame({'x': velocity_x, 'y': velocity_y, 'z': velocity_z}).to_numpy()
    t = timestamps

    # If a time window is specified, only use the data within that window
    if time_window is not None:
        mask = t <= t[0] + time_window
        v = v[mask]
        t = t[mask]
        if len(t) < 2:
            return 0, 0

    # Compute cosine similarity between consecutive velocity vectors:
    # cos_sim = (v_i dot v_{i+1}) / (||v_i|| * ||v_{i+1}||)
    dot_products = np.sum(v[:-1] * v[1:], axis=1)
    norms = np.linalg.norm(v[:-1], axis=1) * np.linalg.norm(v[1:], axis=1)
    cos_sim = dot_products / (norms + 1e-12)  # avoid division by zero

    # A reversal occurs when the cosine similarity is negative.
    reversals = np.sum(cos_sim < 0)

    # Compute total time span
    total_time = t[-1] - t[0] if len(t) > 1 else 1
    reversal_frequency = reversals / total_time
    return int(reversals), reversal_frequency

# -----------------------------------------------------------------------------
# 2. Path Efficiency (Tortuosity)
# -----------------------------------------------------------------------------
def calculate_path_efficiency(df):
    """
    Computes the path efficiency (or tortuosity) as the ratio of the net displacement
    to the total distance traveled.

    Parameters:
        df: A Polars DataFrame with columns "Position_x", "Position_y", "Position_z"
    
    Returns:
        efficiency: Ratio of net displacement (straight-line distance from start to end)
                    to total distance traveled.
        net_displacement: Euclidean distance between start and end points.
        total_distance: Sum of Euclidean distances between consecutive positions.
    """
    pos = df.to_numpy()
    # Total distance traveled is the sum of distances between consecutive points.
    diffs = np.diff(pos, axis=0)
    segment_distances = np.linalg.norm(diffs, axis=1)
    total_distance = np.sum(segment_distances)
    
    # Net displacement is the straight-line distance from start to end.
    net_displacement = np.linalg.norm(pos[-1] - pos[0])
    
    efficiency = net_displacement / total_distance if total_distance > 0 else 0
    return efficiency, net_displacement, total_distance

# -----------------------------------------------------------------------------
# 3. Cumulative Opposing Displacement
# -----------------------------------------------------------------------------
def cumulative_opposing_displacement(px, py, pz, vx, vy, vz):
    """
    Calculates the cumulative displacement in the direction opposite to an intended movement.
    
    The "intended" direction is defined as the vector from the start position to the 
    position at which the velocity magnitude is maximum (i.e., at the peak velocity point).
    Then, for all displacements after the peak, the projection onto this intended direction 
    is computed. If the projection is negative (i.e. movement opposite to the intended direction),
    its magnitude is added to the cumulative opposing displacement.

    Parameters:
        df: A Polars DataFrame with columns "Position_x", "Position_y", "Position_z" and 
            "Velocity_x", "Velocity_y", "Velocity_z".
    
    Returns:
        cumulative_opposing: The sum of displacements opposing the intended direction.
    """
    # Extract positions and velocities
    pos = pl.DataFrame({'x': px, 'y': py, 'z': pz}).to_numpy()
    vel = pl.DataFrame({'x': vx, 'y': vy, 'z': vz}).to_numpy()

    # Compute velocity magnitudes and determine peak index.
    vmag = np.linalg.norm(vel, axis=1)
    peak_idx = int(np.argmax(vmag))

    # Intended direction is from the start to the peak position.
    primary_direction = pos[peak_idx] - pos[0]
    norm_primary = np.linalg.norm(primary_direction)
    if norm_primary == 0:
        return 0.0
    primary_unit = primary_direction / norm_primary

    # For each displacement after the peak, compute its projection on the primary direction.
    opposing_displacement = 0.0
    for i in range(peak_idx, len(pos) - 1):
        disp = pos[i + 1] - pos[i]
        proj = np.dot(disp, primary_unit)
        if proj < 0:
            opposing_displacement += abs(proj)
    return opposing_displacement

def quaternion_geodesic_distance(q1, q2):
    """
    Compute the geodesic (angular) distance between two quaternions.
    Given two unit quaternions q1 and q2 (in [x, y, z, w] format), the rotation
    difference is given by:
    
         d = 2 * arccos(|<q1, q2>|)
         
    which is the minimal rotation needed to align them.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    # Ensure unit quaternions.
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, 0, 1)
    return 2 * np.arccos(dot)

def total_rotation_traveled(quat_array):
    """
    Given an array of quaternions (shape (N, 4)) representing a time series,
    compute the total rotation traveled by summing the geodesic distances
    between consecutive quaternions.
    
    Returns:
        total_angle (in radians)
    """
    total = 0.0
    for i in range(1, quat_array.shape[0]):
        total += quaternion_geodesic_distance(quat_array[i-1], quat_array[i])
    return total

def quaternion_dispersion(quat_array):
    """
    Compute a simple measure of dispersion for a set of quaternions.
    One approach is to compute the average geodesic distance between all pairs.
    (This is O(N^2) and may be expensive for long sequences.)
    
    Returns:
        The mean geodesic distance among all pairs.
    """
    N = quat_array.shape[0]
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            dists.append(quaternion_geodesic_distance(quat_array[i], quat_array[j]))
    return np.mean(dists)

def quat_to_axis_angle(quat):
    """
    Convert a quaternion [x, y, z, w] to an axis–angle representation.
    Returns:
        angle: rotation angle in radians
        axis: a unit 3-vector representing the axis of rotation
    """
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Zero norm quaternion")
    q = quat / norm  # Ensure unit quaternion.
    w = q[3]
    angle = 2 * np.arccos(w)
    sin_half_angle = np.sqrt(max(0, 1 - w*w))
    if sin_half_angle < 1e-8:
        # If the angle is close to zero, the axis can be arbitrary.
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = q[:3] / sin_half_angle
    return angle, axis

def find_nonzero_trigger_intervals(arr):
    # Find indices of nonzero elements
    nonzero_indices = np.nonzero(arr)[0]
    
    if len(nonzero_indices) == 0:
        return []
    
    # Find breaks in consecutive indices
    breaks = np.where(np.diff(nonzero_indices) > 1)[0]
    
    # Create start indices by including the first nonzero index
    # and the indices just after the breaks
    start_indices = np.concatenate(([nonzero_indices[0]], nonzero_indices[breaks + 1]))
    
    # Create end indices by including the indices at the breaks
    # and the last nonzero index
    end_indices = np.concatenate((nonzero_indices[breaks], [nonzero_indices[-1]]))
    
    # Return list of (start, end) tuples (inclusive ranges)
    return list(zip(start_indices, end_indices))

def find_zero_trigger_intervals(arr):
    # Find indices of nonzero elements
    nonzero_indices = np.nonzero(arr)[0]
    
    if len(nonzero_indices) == 0:
        # If no nonzero elements, the entire array is zeros
        return [(0, len(arr) - 1)] if len(arr) > 0 else []
    
    intervals = []
    
    # Check if there are zeros at the beginning
    if nonzero_indices[0] > 0:
        intervals.append((0, nonzero_indices[0] - 1))
    
    # Find breaks in consecutive nonzero indices
    breaks = np.where(np.diff(nonzero_indices) > 1)[0]
    
    # For each break, get the interval of zeros
    for i in breaks:
        intervals.append((nonzero_indices[i] + 1, nonzero_indices[i + 1] - 1))
    
    # Check if there are zeros at the end
    if nonzero_indices[-1] < len(arr) - 1:
        intervals.append((nonzero_indices[-1] + 1, len(arr) - 1))
    
    return intervals

# -----------------------------
# SLERP implementation between two quaternions
# -----------------------------
def slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions q1 and q2.
    
    Parameters:
        q1, q2: numpy arrays of shape (4,)
        t: interpolation factor between 0 and 1.
    
    Returns:
        A normalized quaternion as a numpy array of shape (4,).
    """
    dot = np.dot(q1, q2)
    # Ensure the dot product is non-negative to take the shortest path.
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # The quaternions are nearly parallel; use linear interpolation and renormalize.
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)        # angle between q1 and q2
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t             # angle for interpolation
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q1 + s1 * q2
    return result / np.linalg.norm(result)

# -----------------------------
# Compute angular distance between two quaternions
# -----------------------------
def quaternion_angular_distance(q1, q2):
    """
    Compute the angular difference (in radians) between two quaternions.
    
    This is done by computing the relative rotation and taking the norm of its rotation vector.
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_rel = r1.inv() * r2
    angle = np.linalg.norm(r_rel.as_rotvec())
    return angle

# -----------------------------
# Adaptive SLERP-based low-pass filter for quaternions
# -----------------------------
def adaptive_slerp_filter(quat_array, alpha_min=0.05, alpha_max=0.5, 
                           threshold_low=np.deg2rad(0.5), threshold_high=np.deg2rad(5)):
    N = quat_array.shape[0]
    filtered_quats = np.zeros_like(quat_array)
    filtered_quats[0] = quat_array[0]  # Initialize with the first measurement.
    for i in range(1, N):
        q_prev = filtered_quats[i-1]
        q_current = quat_array[i]
        
        # Check for zero norm in q_current. If found, replace it with the previous filtered quaternion.
        if np.linalg.norm(q_current) == 0:
            q_current = q_prev
        
        delta = quaternion_angular_distance(q_prev, q_current)
        if delta <= threshold_low:
            alpha = alpha_min
        elif delta >= threshold_high:
            alpha = alpha_max
        else:
            alpha = alpha_min + (alpha_max - alpha_min) * ((delta - threshold_low) / (threshold_high - threshold_low))
        filtered_quats[i] = slerp(q_prev, q_current, alpha)
    return filtered_quats

def low_pass_filter(obj, df, lbl):

    alpha_min = 100000 # 0.05
    alpha_max = 100000 # 0.5
    threshold_low = np.deg2rad(0.5)  # 0.5° in radians
    threshold_high = np.deg2rad(5)   # 5° in radians

    # Apply the adaptive SLERP filter.
    smoothed_quat = adaptive_slerp_filter(
        df.select([f"{obj}_quat_x", f"{obj}_quat_y", f"{obj}_quat_z", f"{obj}_quat_w"]).to_numpy(), 
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        threshold_low=threshold_low,
        threshold_high=threshold_high
    )

    df = df.drop([f"{obj}_quat_x", f"{obj}_quat_y", f"{obj}_quat_z", f"{obj}_quat_w"])
    df = df.with_columns(
        pl.Series(f"{obj}_quat_x", smoothed_quat[:, 0]),
        pl.Series(f"{obj}_quat_y", smoothed_quat[:, 1]),
        pl.Series(f"{obj}_quat_z", smoothed_quat[:, 2]),
        pl.Series(f"{obj}_quat_w", smoothed_quat[:, 3])
    )
    
    rot = R.from_quat(df.select([f"{obj}_quat_x", f"{obj}_quat_y", f"{obj}_quat_z", f"{obj}_quat_w"]).to_numpy())
    rot_euler = rot.as_euler('xyz', degrees=True)
    euler_df = pl.DataFrame(rot_euler)
    df = df.with_columns(
        euler_df['column_0'].alias(f"{obj}_euler_x"),
        euler_df['column_1'].alias(f"{obj}_euler_y"),
        euler_df['column_2'].alias(f"{obj}_euler_z")
    )

    return df

def compute_linear_derivatives(positions, timestamps, window_length=11, polyorder=2, deriv=1):
    """
    Compute smoothed linear derivatives for each axis using the Savitzky–Golay filter.
    
    Parameters:
        positions : numpy.ndarray of shape (N, 3)
            Array of x, y, z positions.
        timestamps : numpy.ndarray of shape (N,)
            Array of corresponding timestamps.
        window_length : int
            The length of the filter window (must be odd).
        polyorder : int
            The order of the polynomial to fit.
            
    Returns:
        velocities : numpy.ndarray of shape (N, 3)
            Smoothed derivatives for each axis.
    """
    # Ensure window_length is valid.
    if window_length % 2 == 0:
        raise ValueError("window_length must be an odd number.")
    
    # Compute the time differences for scaling the derivative.
    dt = np.gradient(timestamps)
    
    derivatives = np.zeros_like(positions)
    # Apply the Savitzky–Golay filter for each axis.
    for axis in range(positions.shape[1]):
        # Use the deriv=1 option to compute the first derivative.
        derivatives[:, axis] = savgol_filter(positions[:, axis], window_length, polyorder, deriv=deriv) / dt
    deriv_x = derivatives[:, 0]
    deriv_y = derivatives[:, 1]
    deriv_z = derivatives[:, 2]
    return deriv_x, deriv_y, deriv_z

def compute_angular_derivatives(quaternions, timestamps, window_length=11, polyorder=2, deriv=1):
    rotations = R.from_quat(quaternions)
    angular_derivatives = []
    for i in range(1, len(rotations)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0:
            raise ValueError("Timestamps must be strictly increasing.")
        relative_rotation = rotations[i-1].inv() * rotations[i]
        rotvec = relative_rotation.as_rotvec()
        angular_derivatives.append(rotvec / dt)
    angular_derivatives = np.array(angular_derivatives)
    smoothed_ang_deriv = np.zeros_like(angular_derivatives)
    for j in range(3):
        smoothed_ang_deriv[:, j] = savgol_filter(angular_derivatives[:, j], window_length, polyorder, deriv=deriv)
    ang_deriv_x = smoothed_ang_deriv[:, 0]
    ang_deriv_y = smoothed_ang_deriv[:, 1]
    ang_deriv_z = smoothed_ang_deriv[:, 2]
    
    # Before returning, pad the derivatives by duplicating the first value
    ang_deriv_x = np.concatenate(([ang_deriv_x[0]], ang_deriv_x))
    ang_deriv_y = np.concatenate(([ang_deriv_y[0]], ang_deriv_y))
    ang_deriv_z = np.concatenate(([ang_deriv_z[0]], ang_deriv_z))
    
    return ang_deriv_x, ang_deriv_y, ang_deriv_z

def substep_type_to_int(substep_type):
    if substep_type == 'AttachPiece':
        return 0
    elif substep_type == 'InsertScrew':
        return 1
    elif substep_type == 'UseKey':
        return 2
    else:
        raise ValueError(f"Invalid substep type: {substep_type}")

def get_stat_features(input_data, prefix=""):
    features = {}
    
    if (input_data is None) or (len(input_data) == 0):
        # print(f"Stat is None or empty: {prefix}")
        return features
    elif None in input_data:
        # print(f"Stat has None: {prefix}")
        input_data = [i for i in input_data if i is not None]
    
    features[f'{prefix}_coeff_of_var'] = calculate_coefficient_of_variation(input_data)
    features[f'{prefix}_slope'] = slope_of_linear_regression(np.arange(len(input_data)), input_data)
    features[f'{prefix}_mean'] = np.mean(input_data)
    features[f'{prefix}_median'] = np.median(input_data)
    features[f'{prefix}_std'] = np.std(input_data)
    features[f'{prefix}_var'] = np.var(input_data)
    features[f'{prefix}_skewness'] = skew(input_data)
    features[f'{prefix}_kurtosis'] = kurtosis(input_data)
    features[f'{prefix}_iqr'] = np.percentile(input_data, 75) - np.percentile(input_data, 25)
    features[f'{prefix}_range'] = np.max(input_data) - np.min(input_data)
    features[f'{prefix}_min'] = np.min(input_data)
    features[f'{prefix}_max'] = np.max(input_data)
    features[f'{prefix}_sum'] = np.sum(input_data)   

    return features

def get_custom_features(input_data, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix="", fs=90):
    features = {}

    if (input_data is None) or (len(input_data) == 0):
        # print(f"Custom is None or empty: {prefix}")
        return features
    elif None in input_data:
        # print(f"Custom has None: {prefix}")
        input_data = [i for i in input_data if i is not None]
        
    features[f'{prefix}_coeff_of_var'] = calculate_coefficient_of_variation(input_data)
    features[f'{prefix}_slope'] = slope_of_linear_regression(np.arange(len(input_data)), input_data)
    features.update(get_linear_features(input_data, prefix=prefix, fs=fs))

    max_val, max_time = get_extremum_of_vector(input_data, timestamps, extremum_type='max')
    features[f'{prefix}_max_value'] = max_val
    features[f'{prefix}_time_of_max_value'] = max_time
    features[f'{prefix}_percent_through_session_of_max_value'] = max_time / timestamps.max()
    min_idx = np.argmin(np.abs(asbly_timestamps - max_time))
    features[f'{prefix}_step_of_max_value'] = asbly_steps[min_idx]
    features[f'{prefix}_substep_of_max_value'] = min_idx
    features[f'{prefix}_substep_type_of_max_value'] = substep_type_to_int(asbly_substeps[min_idx])

    min_val, min_time = get_extremum_of_vector(input_data, timestamps, extremum_type='min')
    features[f'{prefix}_min_value'] = min_val
    features[f'{prefix}_time_of_min_value'] = min_time
    features[f'{prefix}_percent_through_session_of_min_value'] = min_time / timestamps.max()
    min_idx = np.argmin(np.abs(asbly_timestamps - min_time))
    features[f'{prefix}_step_of_min_value'] = asbly_steps[min_idx]
    features[f'{prefix}_substep_of_min_value'] = min_idx
    features[f'{prefix}_substep_type_of_min_value'] = substep_type_to_int(asbly_substeps[min_idx])

    median_val = np.median(input_data)
    mad = np.median(np.abs(input_data - median_val))
    threshold_high_above = median_val + 3 * mad
    threshold_mid_above = median_val + 2 * mad
    threshold_low_above = median_val + 1.5 * mad
    threshold_base_above = median_val + mad
    threshold_high_below = median_val - 3 * mad
    threshold_mid_below = median_val - 2 * mad
    threshold_low_below = median_val - 1.5 * mad
    threshold_base_below = median_val - mad

    duration_high_above_threshold = duration_extremum_vector(input_data, timestamps, threshold=threshold_high_above, extremum_type='max')
    duration_mid_above_threshold = duration_extremum_vector(input_data, timestamps, threshold=threshold_mid_above, extremum_type='max')
    duration_low_above_threshold = duration_extremum_vector(input_data, timestamps, threshold=threshold_low_above, extremum_type='max')
    duration_base_above_threshold = duration_extremum_vector(input_data, timestamps, threshold=threshold_base_above, extremum_type='max')
    duration_high_below_threshold = duration_extremum_vector(input_data, timestamps, threshold=threshold_high_below, extremum_type='min')
    duration_mid_below_threshold = duration_extremum_vector(input_data, timestamps, threshold=threshold_mid_below, extremum_type='min')
    duration_low_below_threshold = duration_extremum_vector(input_data, timestamps, threshold=threshold_low_below, extremum_type='min')
    duration_base_below_threshold = duration_extremum_vector(input_data, timestamps, threshold=threshold_base_below, extremum_type='min')

    features[f'{prefix}_duration_high_above_threshold'] = duration_high_above_threshold
    features[f'{prefix}_duration_mid_above_threshold'] = duration_mid_above_threshold
    features[f'{prefix}_duration_low_above_threshold'] = duration_low_above_threshold
    features[f'{prefix}_duration_base_above_threshold'] = duration_base_above_threshold
    features[f'{prefix}_duration_high_below_threshold'] = duration_high_below_threshold
    features[f'{prefix}_duration_mid_below_threshold'] = duration_mid_below_threshold
    features[f'{prefix}_duration_low_below_threshold'] = duration_low_below_threshold
    features[f'{prefix}_duration_base_below_threshold'] = duration_base_below_threshold

    freq_high_above_threshold, points_high_above_extrema = frequency_of_extrema_crossing_threshold(input_data, timestamps, threshold=threshold_high_above, extremum_type='max')
    freq_mid_above_threshold, points_mid_above_extrema = frequency_of_extrema_crossing_threshold(input_data, timestamps, threshold=threshold_mid_above, extremum_type='max')
    freq_low_above_threshold, points_low_above_extrema = frequency_of_extrema_crossing_threshold(input_data, timestamps, threshold=threshold_low_above, extremum_type='max')
    freq_high_below_threshold, points_high_below_extrema = frequency_of_extrema_crossing_threshold(input_data, timestamps, threshold=threshold_high_below, extremum_type='min')
    freq_mid_below_threshold, points_mid_below_extrema = frequency_of_extrema_crossing_threshold(input_data, timestamps, threshold=threshold_mid_below, extremum_type='min')
    freq_low_below_threshold, points_low_below_extrema = frequency_of_extrema_crossing_threshold(input_data, timestamps, threshold=threshold_low_below, extremum_type='min')
    freq_base_above_threshold, points_base_above_extrema = frequency_of_extrema_crossing_threshold(input_data, timestamps, threshold=threshold_base_above, extremum_type='max')
    freq_base_below_threshold, points_base_below_extrema = frequency_of_extrema_crossing_threshold(input_data, timestamps, threshold=threshold_base_below, extremum_type='min')
    
    features[f'{prefix}_freq_high_above_threshold'] = freq_high_above_threshold
    features[f'{prefix}_freq_mid_above_threshold'] = freq_mid_above_threshold
    features[f'{prefix}_freq_low_above_threshold'] = freq_low_above_threshold
    features[f'{prefix}_freq_base_above_threshold'] = freq_base_above_threshold
    features[f'{prefix}_freq_high_below_threshold'] = freq_high_below_threshold
    features[f'{prefix}_freq_mid_below_threshold'] = freq_mid_below_threshold
    features[f'{prefix}_freq_low_below_threshold'] = freq_low_below_threshold
    features[f'{prefix}_freq_base_below_threshold'] = freq_base_below_threshold

    features[f'{prefix}_num_high_above_extrema'] = len(points_high_above_extrema)
    features[f'{prefix}_num_mid_above_extrema'] = len(points_mid_above_extrema)
    features[f'{prefix}_num_low_above_extrema'] = len(points_low_above_extrema)
    features[f'{prefix}_num_base_above_extrema'] = len(points_base_above_extrema)
    features[f'{prefix}_num_high_below_extrema'] = len(points_high_below_extrema)
    features[f'{prefix}_num_mid_below_extrema'] = len(points_mid_below_extrema)
    features[f'{prefix}_num_low_below_extrema'] = len(points_low_below_extrema)
    features[f'{prefix}_num_base_below_extrema'] = len(points_base_below_extrema)

    inter_extrema_intervals_high_above_threshold, _ = inter_extrema_intervals(timestamps, points_high_above_extrema)
    inter_extrema_intervals_mid_above_threshold, _ = inter_extrema_intervals(timestamps, points_mid_above_extrema)
    inter_extrema_intervals_low_above_threshold, _ = inter_extrema_intervals(timestamps, points_low_above_extrema)
    inter_extrema_intervals_base_above_threshold, _ = inter_extrema_intervals(timestamps, points_base_above_extrema)
    inter_extrema_intervals_high_below_threshold, _ = inter_extrema_intervals(timestamps, points_high_below_extrema)
    inter_extrema_intervals_mid_below_threshold, _ = inter_extrema_intervals(timestamps, points_mid_below_extrema)
    inter_extrema_intervals_low_below_threshold, _ = inter_extrema_intervals(timestamps, points_low_below_extrema)
    inter_extrema_intervals_base_below_threshold, _ = inter_extrema_intervals(timestamps, points_base_below_extrema)

    features.update(get_stat_features(inter_extrema_intervals_high_above_threshold, prefix=f'{prefix}_inter_extrema_intervals_high_above_threshold'))
    features.update(get_stat_features(inter_extrema_intervals_mid_above_threshold, prefix=f'{prefix}_inter_extrema_intervals_mid_above_threshold'))
    features.update(get_stat_features(inter_extrema_intervals_low_above_threshold, prefix=f'{prefix}_inter_extrema_intervals_low_above_threshold'))
    features.update(get_stat_features(inter_extrema_intervals_base_above_threshold, prefix=f'{prefix}_inter_extrema_intervals_base_above_threshold'))
    features.update(get_stat_features(inter_extrema_intervals_high_below_threshold, prefix=f'{prefix}_inter_extrema_intervals_high_below_threshold'))
    features.update(get_stat_features(inter_extrema_intervals_mid_below_threshold, prefix=f'{prefix}_inter_extrema_intervals_mid_below_threshold'))
    features.update(get_stat_features(inter_extrema_intervals_low_below_threshold, prefix=f'{prefix}_inter_extrema_intervals_low_below_threshold'))
    features.update(get_stat_features(inter_extrema_intervals_base_below_threshold, prefix=f'{prefix}_inter_extrema_intervals_base_below_threshold'))
    
    time_to_extrema_high_above_threshold = time_to_extrema(input_data, timestamps, extremum_type='max')
    time_to_extrema_mid_above_threshold = time_to_extrema(input_data, timestamps, extremum_type='max')
    time_to_extrema_low_above_threshold = time_to_extrema(input_data, timestamps, extremum_type='max')
    time_to_extrema_base_above_threshold = time_to_extrema(input_data, timestamps, extremum_type='max')
    time_to_extrema_high_below_threshold = time_to_extrema(input_data, timestamps, extremum_type='min')
    time_to_extrema_mid_below_threshold = time_to_extrema(input_data, timestamps, extremum_type='min')
    time_to_extrema_low_below_threshold = time_to_extrema(input_data, timestamps, extremum_type='min')
    time_to_extrema_base_below_threshold = time_to_extrema(input_data, timestamps, extremum_type='min')
    
    features[f'{prefix}_time_to_extrema_high_above_threshold'] = time_to_extrema_high_above_threshold
    features[f'{prefix}_time_to_extrema_mid_above_threshold'] = time_to_extrema_mid_above_threshold
    features[f'{prefix}_time_to_extrema_low_above_threshold'] = time_to_extrema_low_above_threshold
    features[f'{prefix}_time_to_extrema_base_above_threshold'] = time_to_extrema_base_above_threshold
    features[f'{prefix}_time_to_extrema_high_below_threshold'] = time_to_extrema_high_below_threshold
    features[f'{prefix}_time_to_extrema_mid_below_threshold'] = time_to_extrema_mid_below_threshold
    features[f'{prefix}_time_to_extrema_low_below_threshold'] = time_to_extrema_low_below_threshold
    features[f'{prefix}_time_to_extrema_base_below_threshold'] = time_to_extrema_base_below_threshold
    
    decay_time_high_above_threshold = decay_time(input_data, timestamps, decay_threshold=threshold_high_above, extremum_type='max')
    decay_time_mid_above_threshold = decay_time(input_data, timestamps, decay_threshold=threshold_mid_above, extremum_type='max')
    decay_time_low_above_threshold = decay_time(input_data, timestamps, decay_threshold=threshold_low_above, extremum_type='max')
    decay_time_base_above_threshold = decay_time(input_data, timestamps, decay_threshold=threshold_base_above, extremum_type='max')
    decay_time_high_below_threshold = decay_time(input_data, timestamps, decay_threshold=threshold_high_below, extremum_type='min')
    decay_time_mid_below_threshold = decay_time(input_data, timestamps, decay_threshold=threshold_mid_below, extremum_type='min')
    decay_time_low_below_threshold = decay_time(input_data, timestamps, decay_threshold=threshold_low_below, extremum_type='min')
    decay_time_base_below_threshold = decay_time(input_data, timestamps, decay_threshold=threshold_base_below, extremum_type='min')
    
    features[f'{prefix}_decay_time_high_above_threshold'] = decay_time_high_above_threshold
    features[f'{prefix}_decay_time_mid_above_threshold'] = decay_time_mid_above_threshold
    features[f'{prefix}_decay_time_low_above_threshold'] = decay_time_low_above_threshold
    features[f'{prefix}_decay_time_base_above_threshold'] = decay_time_base_above_threshold
    features[f'{prefix}_decay_time_high_below_threshold'] = decay_time_high_below_threshold
    features[f'{prefix}_decay_time_mid_below_threshold'] = decay_time_mid_below_threshold
    features[f'{prefix}_decay_time_low_below_threshold'] = decay_time_low_below_threshold
    features[f'{prefix}_decay_time_base_below_threshold'] = decay_time_base_below_threshold

    return features

def get_custom_features_position(input_data, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix="", fs=90, stat_features=True):
    features = {}
    if len(input_data) == 0:
        return features
    features[f'{prefix}_bbox'] = calculate_bounding_box_volume(input_data)
    features[f'{prefix}_conv_hull'] = calculate_convex_hull_volume(input_data)
    features[f'{prefix}_total_dist_traveled'] = calculate_total_distance_traveled_3D(input_data)
    efficiency, net_displacement, _ = calculate_path_efficiency(input_data)
    features[f'{prefix}_path_efficiency'] = efficiency
    features[f'{prefix}_net_displacement'] = net_displacement
    pos_x = []
    pos_y = []
    pos_z = []

    np_input_data = input_data.to_numpy()

    for idx in range(len(np_input_data)):
        pos_x.append(np_input_data[idx][0])
        pos_y.append(np_input_data[idx][1])
        pos_z.append(np_input_data[idx][2])
    if stat_features:
        features.update(get_stat_features(pos_x, prefix=f'{prefix}_pos_x'))
        features.update(get_stat_features(pos_y, prefix=f'{prefix}_pos_y'))
        features.update(get_stat_features(pos_z, prefix=f'{prefix}_pos_z'))
    else:
        features.update(get_custom_features(pos_x, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_pos_x', fs=fs))
        features.update(get_custom_features(pos_y, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_pos_y', fs=fs))
        features.update(get_custom_features(pos_z, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_pos_z', fs=fs))
    features[f'{prefix}_pos_x_total_dist_traveled'] = calculate_total_distance_traveled_1D(pos_x)
    features[f'{prefix}_pos_y_total_dist_traveled'] = calculate_total_distance_traveled_1D(pos_y)
    features[f'{prefix}_pos_z_total_dist_traveled'] = calculate_total_distance_traveled_1D(pos_z)

    return features

def get_custom_features_quat(input_data, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix="", fs=90, stat_features=True):
    features = {}

    if len(input_data) == 0:
        return features

    features[f'{prefix}_total_rotation_traveled'] = total_rotation_traveled(input_data)
    # features[f'{prefix}_quaternion_dispersion'] = quaternion_dispersion(input_data)
    quat_x = []
    quat_y = []
    quat_z = []
    quat_w = []
    axis_x = []
    axis_y = []
    axis_z = []
    angles = []
    for idx in range(len(input_data)):
        quat_x.append(input_data[idx][0])
        quat_y.append(input_data[idx][1])
        quat_z.append(input_data[idx][2])
        quat_w.append(input_data[idx][3])
        angle, axis = quat_to_axis_angle(input_data[idx])
        axis_x.append(axis[0])
        axis_y.append(axis[1])
        axis_z.append(axis[2])
        angles.append(angle)

    if stat_features:
        features.update(get_stat_features(quat_x, prefix=f'{prefix}_quat_x'))
        features.update(get_stat_features(quat_y, prefix=f'{prefix}_quat_y'))
        features.update(get_stat_features(quat_z, prefix=f'{prefix}_quat_z'))
        features.update(get_stat_features(quat_w, prefix=f'{prefix}_quat_w'))
        features.update(get_stat_features(axis_x, prefix=f'{prefix}_axis_x'))
        features.update(get_stat_features(axis_y, prefix=f'{prefix}_axis_y'))
        features.update(get_stat_features(axis_z, prefix=f'{prefix}_axis_z'))
        features.update(get_stat_features(angles, prefix=f'{prefix}_axis_angle'))
    else:
        features.update(get_custom_features(quat_x, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_quat_x', fs=fs))
        features.update(get_custom_features(quat_y, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_quat_y', fs=fs))
        features.update(get_custom_features(quat_z, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_quat_z', fs=fs))
        features.update(get_custom_features(quat_w, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_quat_w', fs=fs))
        features.update(get_custom_features(angles, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_axis_angle', fs=fs))
        features.update(get_custom_features(axis_x, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_axis_x', fs=fs))
        features.update(get_custom_features(axis_y, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_axis_y', fs=fs))
        features.update(get_custom_features(axis_z, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=f'{prefix}_axis_z', fs=fs))
    return features

def get_custom_features_interval(intervals, timestamps, prefix=""):
    interval_features = {}
    durations = []
    for indices in intervals:
        end_index = int(indices[1])
        start_index = int(indices[0])
        if end_index >= len(timestamps):
            end_index = len(timestamps) - 1
        durations.append(timestamps[end_index] - timestamps[start_index])

    if len(durations) == 0:
        return interval_features
    
    interval_features[f'{prefix}_duration_frequency'] = len(durations) / (timestamps.max() - timestamps.min())
    interval_features[f'{prefix}_duration_count'] = len(durations)
    interval_features.update(get_stat_features(durations, prefix=f'{prefix}_durations'))

    min_idx = np.argmin(durations)
    max_idx = np.argmax(durations)
    interval_features[f'{prefix}_time_to_min_duration'] = timestamps[int(intervals[min_idx][0])] - timestamps.min()
    interval_features[f'{prefix}_time_to_max_duration'] = timestamps[int(intervals[max_idx][0])] - timestamps.min()

    return interval_features

def get_custom_features_motion(input_data, positions, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix="", fs=90, stat_features=True):
    motion_features = {}

    if len(input_data) == 0:
        return motion_features
    
    deriv_x = input_data[:, 0]
    deriv_y = input_data[:, 1]
    deriv_z = input_data[:, 2]
    magnitude = [get_vector_magnitude(deriv_x[idx], deriv_y[idx], deriv_z[idx]) for idx in range(len(input_data))]
    pos_x = positions[:, 0]
    pos_y = positions[:, 1]
    pos_z = positions[:, 2]

    num_reversals, reversal_frequency = directional_reversal_frequency(deriv_x, deriv_y, deriv_z, timestamps)
    motion_features[f'{prefix}_directional_reversal_frequency'] = reversal_frequency
    motion_features[f'{prefix}_num_reversals'] = num_reversals
    motion_features[f'{prefix}_cumulative_opposing_displacement'] = cumulative_opposing_displacement(pos_x, pos_y, pos_z, deriv_x, deriv_y, deriv_z)
    
    vectors = [deriv_x, deriv_y, deriv_z, magnitude]
    prefixes = [prefix.replace('_axis_', '_x_'), prefix.replace('_axis_', '_y_'), prefix.replace('_axis_', '_z_'), prefix.replace('_axis_', '_magnitude_')]
    for vec, pfix in zip(vectors, prefixes):
        if stat_features:
            motion_features.update(get_stat_features(vec, prefix=pfix))
        else:
            motion_features.update(get_custom_features(vec, timestamps, asbly_timestamps, asbly_steps, asbly_substeps, prefix=pfix, fs=fs))

        median_val = np.median(vec)
        mad = np.median(np.abs(vec - median_val))
        thresholds = [1.25, 1, 0.75, 0.5]

        for multiplier in thresholds:
            threshold_high = median_val + multiplier * mad
            threshold_low = median_val - multiplier * mad
            intervals = find_threshold_intervals(vec, threshold_high=threshold_high, threshold_low=threshold_low)
            motion_features.update(get_custom_features_interval(intervals, timestamps, prefix=f"{pfix}_pause_{multiplier}"))

    return motion_features
