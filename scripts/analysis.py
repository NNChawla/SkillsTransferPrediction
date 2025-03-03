import numpy as np
import polars as pl
from pyinform import mutual_info
from nolitsa import dimension
from scipy.spatial import ConvexHull
import tsfel
import nolds
import itertools, time
from scipy.spatial.transform import Rotation as R

def get_step_dfs(tracking_df, assembly_df):
    tracking_step_dfs = []
    tracking_substep_dfs = []
    tracking_substep_type_dfs = {}
    for substep_type in assembly_df['subStep'].unique():
        tracking_substep_type_dfs[substep_type] = []
    
    for step in range(assembly_df['step'].max()+1):
        assembly_step_df = assembly_df.filter(pl.col('step') == step)
        assembly_previous_step_df = None if (step == 0) else assembly_df.filter(pl.col('step') == (step - 1))
        start_time = 0 if (step == 0) else assembly_previous_step_df['Timestamp'].max()
        end_time = assembly_step_df['Timestamp'].max()
        start_idx = (tracking_df["Timestamp"] - start_time).abs().arg_min()
        end_idx = (tracking_df["Timestamp"] - end_time).abs().arg_min()
        tracking_step_df = tracking_df[start_idx:end_idx]
        tracking_step_dfs.append(tracking_step_df)

        for substep_idx in range(len(assembly_step_df)):
            if (step == 0) and (substep_idx == 0):
                start_time = 0
            elif (substep_idx == 0):
                start_time = assembly_previous_step_df['Timestamp'].max()
            else:
                start_time = assembly_step_df['Timestamp'][substep_idx - 1]
            end_time = assembly_step_df['Timestamp'][substep_idx]
            start_idx = (tracking_step_df["Timestamp"] - start_time).abs().arg_min()
            end_idx = (tracking_step_df["Timestamp"] - end_time).abs().arg_min()
            tracking_substep_df = tracking_step_df[start_idx:end_idx]
            tracking_substep_dfs.append(tracking_substep_df)
            tracking_substep_type_dfs[assembly_step_df['subStep'][substep_idx]].append(tracking_substep_df)
    
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
    start_time = time.time()
    print(f'Starting duration features')
    duration_features['session_duration'] = tracking_df["Timestamp"].max()
    duration_features['session_end_duration'] = tracking_end_df["Timestamp"].max() - tracking_end_df["Timestamp"].min()

    for step in range(len(tracking_step_dfs) - 1): # last step is not included as it is imbalanced
        step_duration = tracking_step_dfs[step]["Timestamp"].max() - tracking_step_dfs[step]["Timestamp"].min()
        duration_features[f'step{step}_duration'] = step_duration
        step_durations.append(step_duration)

    duration_features['step_of_max_duration'] = step_durations.index(max(step_durations))
    duration_features['step_of_min_duration'] = step_durations.index(min(step_durations))

    for substep_idx in range(len(tracking_substep_dfs)):
        substep_duration = tracking_substep_dfs[substep_idx]["Timestamp"].max() - tracking_substep_dfs[substep_idx]["Timestamp"].min()
        duration_features[f'substep{substep_idx}_duration'] = substep_duration
        substep_durations.append(substep_duration)
    
    duration_features['substep_of_max_duration'] = substep_durations.index(max(substep_durations))
    duration_features['substep_of_min_duration'] = substep_durations.index(min(substep_durations))
    
    # Feature Vectors
    duration_features['coeff_of_var_of_step_duration'] = calculate_coefficient_of_variation(step_durations)
    duration_features['coeff_of_var_of_substep_duration'] = calculate_coefficient_of_variation(substep_durations)    
    duration_features['slope_of_step_duration'] = slope_of_linear_regression(np.arange(len(step_durations)), step_durations)
    duration_features['slope_of_substep_duration'] = slope_of_linear_regression(np.arange(len(substep_durations)), substep_durations)
    duration_features.update(get_linear_features(step_durations, prefix='step_duration', fs=None))
    duration_features.update(get_linear_features(substep_durations, prefix='substep_duration', fs=None))
    for substep_type in sorted(tracking_substep_type_dfs.keys()):
        subset_type_durations = []
        for substep_idx in range(len(tracking_substep_type_dfs[substep_type])):
            substep_df = tracking_substep_type_dfs[substep_type][substep_idx]
            subset_type_durations.append(substep_df["Timestamp"].max() - substep_df["Timestamp"].min())
        duration_features[f'coeff_of_var_of_{substep_type}_duration'] = calculate_coefficient_of_variation(subset_type_durations)
        duration_features[f'slope_of_{substep_type}_duration'] = slope_of_linear_regression(np.arange(len(subset_type_durations)), subset_type_durations)    
        duration_features.update(get_linear_features(subset_type_durations, prefix=f'{substep_type}_duration', fs=None))
    
    print(f'Duration features completed in {time.time() - start_time} seconds')
    return duration_features

def get_position_features(tracking_df, assembly_df):
    position_features = {}
    tracking_step_dfs, tracking_substep_dfs, tracking_substep_type_dfs, tracking_end_df = get_step_dfs(tracking_df, assembly_df)

    start_time = time.time()
    print(f'Starting session level features')
    # Session level
    obj_list = ['Head', 'LeftHand', 'RightHand']
    obj_pair_list = list(itertools.combinations(obj_list, 2))
    for obj in obj_list:
        obj_df = tracking_df[[f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']]
        position_features[f'{obj}_bbox'] = calculate_bounding_box_volume(obj_df)
        position_features[f'{obj}_conv_hull'] = calculate_convex_hull_volume(obj_df)
        position_features[f'{obj}_total_dist_traveled'] = calculate_total_distance_traveled_3D(obj_df)

        for axis in ['x', 'y', 'z']:
            column_name = f"{obj}_position_{axis}"
            data = tracking_df[column_name].to_numpy()
            position_features[f'coeff_of_var_of_{obj}_position_{axis}'] = calculate_coefficient_of_variation(data)
            position_features[f'slope_of_{obj}_position_{axis}'] = slope_of_linear_regression(np.arange(len(data)), data)
            position_features[f'total_dist_traveled_{obj}_position_{axis}'] = calculate_total_distance_traveled_1D(data)
            position_features.update(get_linear_features(data, prefix=f'{obj}_position_{axis}', fs=100))
    
    for obj_pair in obj_pair_list:
        obj1_df = tracking_df[[f'{obj_pair[0]}_position_x', f'{obj_pair[0]}_position_y', f'{obj_pair[0]}_position_z']]
        obj2_df = tracking_df[[f'{obj_pair[1]}_position_x', f'{obj_pair[1]}_position_y', f'{obj_pair[1]}_position_z']]
        distance_between_objects = calculate_distance_between_objects_3D(obj1_df, obj2_df)
        position_features[f'coeff_of_var_of_{obj_pair[0]}_dist_to_{obj_pair[1]}'] = calculate_coefficient_of_variation(distance_between_objects)
        position_features[f'slope_of_{obj_pair[0]}_dist_to_{obj_pair[1]}'] = slope_of_linear_regression(np.arange(len(distance_between_objects)), distance_between_objects)
        position_features.update(get_linear_features(distance_between_objects, prefix=f'{obj_pair[0]}_dist_to_{obj_pair[1]}', fs=100))

        axes = [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'y'), ('y', 'z'), ('z', 'z')]
        for axis in axes:
            obj1_column_name = f"{obj_pair[0]}_position_{axis[0]}"
            obj2_column_name = f"{obj_pair[1]}_position_{axis[1]}"
            data = calculate_distance_between_objects_1D(tracking_df[obj1_column_name].to_numpy(), tracking_df[obj2_column_name].to_numpy())
            position_features[f'coeff_of_var_of_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'] = calculate_coefficient_of_variation(data)
            position_features[f'slope_of_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'] = slope_of_linear_regression(np.arange(len(data)), data)
            position_features.update(get_linear_features(data, prefix=f'{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}', fs=100))
    
    print(f'Session level features completed in {time.time() - start_time} seconds')

    start_time = time.time()
    print(f'Starting end level features')
    # End level
    for obj in obj_list:
        obj_df = tracking_end_df[[f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']]
        position_features[f'end_{obj}_bbox'] = calculate_bounding_box_volume(obj_df)
        position_features[f'end_{obj}_conv_hull'] = calculate_convex_hull_volume(obj_df)
        position_features[f'end_{obj}_total_dist_traveled'] = calculate_total_distance_traveled_3D(obj_df)

        for axis in ['x', 'y', 'z']:
            column_name = f"{obj}_position_{axis}"
            data = tracking_end_df[column_name].to_numpy()
            position_features[f'coeff_of_var_of_end_{obj}_position_{axis}'] = calculate_coefficient_of_variation(data)
            position_features[f'slope_of_end_{obj}_position_{axis}'] = slope_of_linear_regression(np.arange(len(data)), data)
            position_features[f'total_dist_traveled_end_{obj}_position_{axis}'] = calculate_total_distance_traveled_1D(data)
            position_features.update(get_linear_features(data, prefix=f'end_{obj}_position_{axis}', fs=100))
    
    for obj_pair in obj_pair_list:
        obj1_df = tracking_end_df[[f'{obj_pair[0]}_position_x', f'{obj_pair[0]}_position_y', f'{obj_pair[0]}_position_z']]
        obj2_df = tracking_end_df[[f'{obj_pair[1]}_position_x', f'{obj_pair[1]}_position_y', f'{obj_pair[1]}_position_z']]
        distance_between_objects = calculate_distance_between_objects_3D(obj1_df, obj2_df)
        position_features[f'coeff_of_var_of_end_{obj_pair[0]}_dist_to_{obj_pair[1]}'] = calculate_coefficient_of_variation(distance_between_objects)
        position_features[f'slope_of_end_{obj_pair[0]}_dist_to_{obj_pair[1]}'] = slope_of_linear_regression(np.arange(len(distance_between_objects)), distance_between_objects)
        position_features.update(get_linear_features(distance_between_objects, prefix=f'end_{obj_pair[0]}_dist_to_{obj_pair[1]}', fs=100))

        axes = [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'y'), ('y', 'z'), ('z', 'z')]
        for axis in axes:
            obj1_column_name = f"{obj_pair[0]}_position_{axis[0]}"
            obj2_column_name = f"{obj_pair[1]}_position_{axis[1]}"
            data = calculate_distance_between_objects_1D(tracking_end_df[obj1_column_name].to_numpy(), tracking_end_df[obj2_column_name].to_numpy())
            position_features[f'coeff_of_var_of_end_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'] = calculate_coefficient_of_variation(data)
            position_features[f'slope_of_end_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'] = slope_of_linear_regression(np.arange(len(data)), data)
            position_features.update(get_linear_features(data, prefix=f'end_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}', fs=100))

    print(f'End level features completed in {time.time() - start_time} seconds')

    start_time = time.time()
    print(f'Starting step level features')
    # Step level
    step_bbox_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    step_conv_hull_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    step_total_dist_traveled_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    for step in range(len(tracking_step_dfs) - 1): # last step is not included as it is imbalanced
        for obj in obj_list:
            obj_df = tracking_step_dfs[step][[f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']]
            bbox_volume = calculate_bounding_box_volume(obj_df)
            conv_hull_volume = calculate_convex_hull_volume(obj_df)
            total_dist_traveled = calculate_total_distance_traveled_3D(obj_df)
            step_bbox_dict[obj].append(bbox_volume)
            step_conv_hull_dict[obj].append(conv_hull_volume)
            step_total_dist_traveled_dict[obj].append(total_dist_traveled)
            position_features[f'step{step}_{obj}_bbox'] = bbox_volume
            position_features[f'step{step}_{obj}_conv_hull'] = conv_hull_volume
            position_features[f'step{step}_{obj}_total_dist_traveled'] = total_dist_traveled

            for axis in ['x', 'y', 'z']:
                column_name = f"{obj}_position_{axis}"
                data = tracking_step_dfs[step][column_name].to_numpy()
                position_features[f'coeff_of_var_of_step{step}_{obj}_position_{axis}'] = calculate_coefficient_of_variation(data)
                position_features[f'slope_of_step{step}_{obj}_position_{axis}'] = slope_of_linear_regression(np.arange(len(data)), data)
                position_features[f'total_dist_traveled_step{step}_{obj}_position_{axis}'] = calculate_total_distance_traveled_1D(data)
                position_features.update(get_linear_features(data, prefix=f'step{step}_{obj}_position_{axis}', fs=100))
        
        for obj_pair in obj_pair_list:
            obj1_df = tracking_step_dfs[step][[f'{obj_pair[0]}_position_x', f'{obj_pair[0]}_position_y', f'{obj_pair[0]}_position_z']]
            obj2_df = tracking_step_dfs[step][[f'{obj_pair[1]}_position_x', f'{obj_pair[1]}_position_y', f'{obj_pair[1]}_position_z']]
            distance_between_objects = calculate_distance_between_objects_3D(obj1_df, obj2_df)
            position_features[f'coeff_of_var_of_step{step}_{obj_pair[0]}_dist_to_{obj_pair[1]}'] = calculate_coefficient_of_variation(distance_between_objects)
            position_features[f'slope_of_step{step}_{obj_pair[0]}_dist_to_{obj_pair[1]}'] = slope_of_linear_regression(np.arange(len(distance_between_objects)), distance_between_objects)
            position_features.update(get_linear_features(distance_between_objects, prefix=f'step{step}_{obj_pair[0]}_dist_to_{obj_pair[1]}', fs=100))

            axes = [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'y'), ('y', 'z'), ('z', 'z')]
            for axis in axes:
                obj1_column_name = f"{obj_pair[0]}_position_{axis[0]}"
                obj2_column_name = f"{obj_pair[1]}_position_{axis[1]}"
                data = calculate_distance_between_objects_1D(tracking_step_dfs[step][obj1_column_name].to_numpy(), tracking_step_dfs[step][obj2_column_name].to_numpy())
                position_features[f'coeff_of_var_of_step{step}_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'] = calculate_coefficient_of_variation(data)
                position_features[f'slope_of_step{step}_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'] = slope_of_linear_regression(np.arange(len(data)), data)
                position_features.update(get_linear_features(data, prefix=f'step{step}_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}', fs=100))

    # Substep level
    for obj in obj_list:
        position_features[f'step_of_max_bbox_{obj}'] = step_bbox_dict[obj].index(max(step_bbox_dict[obj]))
        position_features[f'step_of_min_bbox_{obj}'] = step_bbox_dict[obj].index(min(step_bbox_dict[obj]))
        position_features[f'step_of_max_conv_hull_{obj}'] = step_conv_hull_dict[obj].index(max(step_conv_hull_dict[obj]))
        position_features[f'step_of_min_conv_hull_{obj}'] = step_conv_hull_dict[obj].index(min(step_conv_hull_dict[obj]))
        position_features[f'step_of_max_total_dist_traveled_{obj}'] = step_total_dist_traveled_dict[obj].index(max(step_total_dist_traveled_dict[obj]))
        position_features[f'step_of_min_total_dist_traveled_{obj}'] = step_total_dist_traveled_dict[obj].index(min(step_total_dist_traveled_dict[obj]))

        position_features[f'coeff_of_var_of_step_total_dist_traveled_{obj}'] = calculate_coefficient_of_variation(step_total_dist_traveled_dict[obj])
        position_features[f'slope_of_step_total_dist_traveled_{obj}'] = slope_of_linear_regression(np.arange(len(step_total_dist_traveled_dict[obj])), step_total_dist_traveled_dict[obj])
        position_features.update(get_linear_features(step_total_dist_traveled_dict[obj], prefix=f'step_total_dist_traveled_{obj}', fs=None))
        position_features[f'coeff_of_var_of_step_bbox_{obj}'] = calculate_coefficient_of_variation(step_bbox_dict[obj])
        position_features[f'slope_of_step_bbox_{obj}'] = slope_of_linear_regression(np.arange(len(step_bbox_dict[obj])), step_bbox_dict[obj])
        position_features.update(get_linear_features(step_bbox_dict[obj], prefix=f'step_bbox_{obj}', fs=None))
        position_features[f'coeff_of_var_of_step_conv_hull_{obj}'] = calculate_coefficient_of_variation(step_conv_hull_dict[obj])
        position_features[f'slope_of_step_conv_hull_{obj}'] = slope_of_linear_regression(np.arange(len(step_conv_hull_dict[obj])), step_conv_hull_dict[obj])
        position_features.update(get_linear_features(step_conv_hull_dict[obj], prefix=f'step_conv_hull_{obj}', fs=None))
    
    print(f'Step level features completed in {time.time() - start_time} seconds')

    start_time = time.time()
    print(f'Starting substep level features')
    # Substep level
    substep_bbox_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    substep_conv_hull_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    substep_total_dist_traveled_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    for substep_idx in range(len(tracking_substep_dfs)): # last step is not included as it is imbalanced
        for obj in obj_list:
            obj_df = tracking_substep_dfs[substep_idx][[f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']]
            bbox_volume = calculate_bounding_box_volume(obj_df)
            conv_hull_volume = calculate_convex_hull_volume(obj_df)
            total_dist_traveled = calculate_total_distance_traveled_3D(obj_df)
            substep_bbox_dict[obj].append(bbox_volume)
            substep_conv_hull_dict[obj].append(conv_hull_volume)
            substep_total_dist_traveled_dict[obj].append(total_dist_traveled)
            position_features[f'substep{substep_idx}_{obj}_bbox'] = bbox_volume
            position_features[f'substep{substep_idx}_{obj}_conv_hull'] = conv_hull_volume
            position_features[f'substep{substep_idx}_{obj}_total_dist_traveled'] = total_dist_traveled

            for axis in ['x', 'y', 'z']:
                column_name = f"{obj}_position_{axis}"
                data = tracking_substep_dfs[substep_idx][column_name].to_numpy()
                position_features[f'coeff_of_var_of_substep{substep_idx}_{obj}_position_{axis}'] = calculate_coefficient_of_variation(data)
                position_features[f'slope_of_substep{substep_idx}_{obj}_position_{axis}'] = slope_of_linear_regression(np.arange(len(data)), data)
                position_features[f'total_dist_traveled_substep{substep_idx}_{obj}_position_{axis}'] = calculate_total_distance_traveled_1D(data)
                position_features.update(get_linear_features(data, prefix=f'substep{substep_idx}_{obj}_position_{axis}', fs=100))
        
        for obj_pair in obj_pair_list:
            obj1_df = tracking_substep_dfs[substep_idx][[f'{obj_pair[0]}_position_x', f'{obj_pair[0]}_position_y', f'{obj_pair[0]}_position_z']]
            obj2_df = tracking_substep_dfs[substep_idx][[f'{obj_pair[1]}_position_x', f'{obj_pair[1]}_position_y', f'{obj_pair[1]}_position_z']]
            distance_between_objects = calculate_distance_between_objects_3D(obj1_df, obj2_df)
            position_features[f'coeff_of_var_of_substep{substep_idx}_{obj_pair[0]}_dist_to_{obj_pair[1]}'] = calculate_coefficient_of_variation(distance_between_objects)
            position_features[f'slope_of_substep{substep_idx}_{obj_pair[0]}_dist_to_{obj_pair[1]}'] = slope_of_linear_regression(np.arange(len(distance_between_objects)), distance_between_objects)
            position_features.update(get_linear_features(distance_between_objects, prefix=f'substep{substep_idx}_{obj_pair[0]}_dist_to_{obj_pair[1]}', fs=100))

            axes = [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'y'), ('y', 'z'), ('z', 'z')]
            for axis in axes:
                obj1_column_name = f"{obj_pair[0]}_position_{axis[0]}"
                obj2_column_name = f"{obj_pair[1]}_position_{axis[1]}"
                data = calculate_distance_between_objects_1D(tracking_substep_dfs[substep_idx][obj1_column_name].to_numpy(), tracking_substep_dfs[substep_idx][obj2_column_name].to_numpy())
                position_features[f'coeff_of_var_of_substep{substep_idx}_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'] = calculate_coefficient_of_variation(data)
                position_features[f'slope_of_substep{substep_idx}_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}'] = slope_of_linear_regression(np.arange(len(data)), data)
                position_features.update(get_linear_features(data, prefix=f'substep{substep_idx}_{obj_pair[0]}_dist_to_{obj_pair[1]}_{axis[0]}_{axis[1]}', fs=100))

    for obj in obj_list:
        position_features[f'substep_of_max_bbox_{obj}'] = substep_bbox_dict[obj].index(max(substep_bbox_dict[obj]))
        position_features[f'substep_of_min_bbox_{obj}'] = substep_bbox_dict[obj].index(min(substep_bbox_dict[obj]))
        position_features[f'substep_of_max_conv_hull_{obj}'] = substep_conv_hull_dict[obj].index(max(substep_conv_hull_dict[obj]))
        position_features[f'substep_of_min_conv_hull_{obj}'] = substep_conv_hull_dict[obj].index(min(substep_conv_hull_dict[obj]))
        position_features[f'substep_of_max_total_dist_traveled_{obj}'] = substep_total_dist_traveled_dict[obj].index(max(substep_total_dist_traveled_dict[obj]))
        position_features[f'substep_of_min_total_dist_traveled_{obj}'] = substep_total_dist_traveled_dict[obj].index(min(substep_total_dist_traveled_dict[obj]))
        
        position_features[f'coeff_of_var_of_substep_total_dist_traveled_{obj}'] = calculate_coefficient_of_variation(substep_total_dist_traveled_dict[obj])
        position_features[f'slope_of_substep_total_dist_traveled_{obj}'] = slope_of_linear_regression(np.arange(len(substep_total_dist_traveled_dict[obj])), substep_total_dist_traveled_dict[obj])
        position_features.update(get_linear_features(substep_total_dist_traveled_dict[obj], prefix=f'substep_total_dist_traveled_{obj}', fs=None))
        position_features[f'coeff_of_var_of_substep_bbox_{obj}'] = calculate_coefficient_of_variation(substep_bbox_dict[obj])
        position_features[f'slope_of_substep_bbox_{obj}'] = slope_of_linear_regression(np.arange(len(substep_bbox_dict[obj])), substep_bbox_dict[obj])
        position_features.update(get_linear_features(substep_bbox_dict[obj], prefix=f'substep_bbox_{obj}', fs=None))
        position_features[f'coeff_of_var_of_substep_conv_hull_{obj}'] = calculate_coefficient_of_variation(substep_conv_hull_dict[obj])
        position_features[f'slope_of_substep_conv_hull_{obj}'] = slope_of_linear_regression(np.arange(len(substep_conv_hull_dict[obj])), substep_conv_hull_dict[obj])
        position_features.update(get_linear_features(substep_conv_hull_dict[obj], prefix=f'substep_conv_hull_{obj}', fs=None))
    
    print(f'Substep level features completed in {time.time() - start_time} seconds')

    start_time = time.time()
    print(f'Starting substep type level features')
    # Substep type level
    for substep_type in sorted(tracking_substep_type_dfs.keys()):
        for obj in obj_list:
            bbox_volumes = []
            conv_hull_volumes = []
            total_dists_traveled = []
            for substep_idx in range(len(tracking_substep_type_dfs[substep_type])):
                obj_df = tracking_substep_type_dfs[substep_type][substep_idx][[f'{obj}_position_x', f'{obj}_position_y', f'{obj}_position_z']]
                bbox_volumes.append(calculate_bounding_box_volume(obj_df))
                conv_hull_volumes.append(calculate_convex_hull_volume(obj_df))
                total_dists_traveled.append(calculate_total_distance_traveled_3D(obj_df))
            position_features[f'coeff_of_var_of_{substep_type}_conv_hull_{obj}'] = calculate_coefficient_of_variation(conv_hull_volumes)
            position_features[f'slope_of_{substep_type}_conv_hull_{obj}'] = slope_of_linear_regression(np.arange(len(conv_hull_volumes)), conv_hull_volumes)
            position_features.update(get_linear_features(conv_hull_volumes, prefix=f'{substep_type}_conv_hull_{obj}', fs=None))
            position_features[f'coeff_of_var_of_{substep_type}_bbox_{obj}'] = calculate_coefficient_of_variation(bbox_volumes)
            position_features[f'slope_of_{substep_type}_bbox_{obj}'] = slope_of_linear_regression(np.arange(len(bbox_volumes)), bbox_volumes)
            position_features.update(get_linear_features(bbox_volumes, prefix=f'{substep_type}_bbox_{obj}', fs=None))
            position_features[f'coeff_of_var_of_{substep_type}_total_dist_traveled_{obj}'] = calculate_coefficient_of_variation(total_dists_traveled)
            position_features[f'slope_of_{substep_type}_total_dist_traveled_{obj}'] = slope_of_linear_regression(np.arange(len(total_dists_traveled)), total_dists_traveled)
            position_features.update(get_linear_features(total_dists_traveled, prefix=f'{substep_type}_total_dist_traveled_{obj}', fs=None))

    print(f'Substep type level features completed in {time.time() - start_time} seconds')

    return position_features

def get_euler_features(tracking_df, assembly_df):
    euler_features = {}
    tracking_step_dfs, tracking_substep_dfs, tracking_substep_type_dfs, tracking_end_df = get_step_dfs(tracking_df, assembly_df)

    # Session level
    obj_list = ['Head', 'LeftHand', 'RightHand']
    obj_pair_list = list(itertools.combinations(obj_list, 2))
    for obj in obj_list:
        obj_df = tracking_df[[f'{obj}_euler_x', f'{obj}_euler_y', f'{obj}_euler_z']]
        total_rotation_traveled_deg, total_rotation_traveled_rad = calculate_total_rotation_traveled_3D(obj_df)
        euler_features[f'{obj}_total_rotation_traveled_deg'] = total_rotation_traveled_deg
        euler_features[f'{obj}_total_rotation_traveled_rad'] = total_rotation_traveled_rad

        for axis in ['x', 'y', 'z']:
            column_name = f"{obj}_euler_{axis}"
            data = tracking_df[column_name].to_numpy()
            euler_features[f'coeff_of_var_of_{obj}_euler_{axis}'] = calculate_coefficient_of_variation(data)
            euler_features[f'slope_of_{obj}_euler_{axis}'] = slope_of_linear_regression(np.arange(len(data)), data)
            axis_rotation_traveled_deg, axis_rotation_traveled_rad = calculate_total_rotation_traveled_1D(data)
            euler_features[f'total_rotation_traveled_{obj}_euler_{axis}_deg'] = axis_rotation_traveled_deg
            euler_features[f'total_rotation_traveled_{obj}_euler_{axis}_rad'] = axis_rotation_traveled_rad
            euler_features.update(get_linear_features(data, prefix=f'{obj}_euler_{axis}', fs=100))

    # End level
    for obj in obj_list:
        obj_df = tracking_end_df[[f'{obj}_euler_x', f'{obj}_euler_y', f'{obj}_euler_z']]
        total_rotation_traveled_deg, total_rotation_traveled_rad = calculate_total_rotation_traveled_3D(obj_df)
        euler_features[f'end_{obj}_total_rotation_traveled_deg'] = total_rotation_traveled_deg
        euler_features[f'end_{obj}_total_rotation_traveled_rad'] = total_rotation_traveled_rad

        for axis in ['x', 'y', 'z']:
            column_name = f"{obj}_euler_{axis}"
            data = tracking_end_df[column_name].to_numpy()
            euler_features[f'coeff_of_var_of_end_{obj}_euler_{axis}'] = calculate_coefficient_of_variation(data)
            euler_features[f'slope_of_end_{obj}_euler_{axis}'] = slope_of_linear_regression(np.arange(len(data)), data)
            axis_rotation_traveled_deg, axis_rotation_traveled_rad = calculate_total_rotation_traveled_1D(data)
            euler_features[f'end_total_rotation_traveled_{obj}_euler_{axis}_deg'] = axis_rotation_traveled_deg
            euler_features[f'end_total_rotation_traveled_{obj}_euler_{axis}_rad'] = axis_rotation_traveled_rad
            euler_features.update(get_linear_features(data, prefix=f'end_{obj}_euler_{axis}', fs=100))

    # Step level
    step_total_rotation_traveled_deg_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    step_total_rotation_traveled_rad_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    for step in range(len(tracking_step_dfs) - 1): # last step is not included as it is imbalanced
        for obj in obj_list:
            obj_df = tracking_step_dfs[step][[f'{obj}_euler_x', f'{obj}_euler_y', f'{obj}_euler_z']]
            total_rotation_traveled_deg, total_rotation_traveled_rad = calculate_total_rotation_traveled_3D(obj_df)
            step_total_rotation_traveled_deg_dict[obj].append(total_rotation_traveled_deg)
            step_total_rotation_traveled_rad_dict[obj].append(total_rotation_traveled_rad)
            euler_features[f'step{step}_{obj}_total_rotation_traveled_deg'] = total_rotation_traveled_deg
            euler_features[f'step{step}_{obj}_total_rotation_traveled_rad'] = total_rotation_traveled_rad

            for axis in ['x', 'y', 'z']:
                column_name = f"{obj}_euler_{axis}"
                data = tracking_step_dfs[step][column_name].to_numpy()
                euler_features[f'coeff_of_var_of_step{step}_{obj}_euler_{axis}'] = calculate_coefficient_of_variation(data)
                euler_features[f'slope_of_step{step}_{obj}_euler_{axis}'] = slope_of_linear_regression(np.arange(len(data)), data)
                axis_rotation_traveled_deg, axis_rotation_traveled_rad = calculate_total_rotation_traveled_1D(data)
                euler_features[f'total_rotation_traveled_step{step}_{obj}_euler_{axis}_deg'] = axis_rotation_traveled_deg
                euler_features[f'total_rotation_traveled_step{step}_{obj}_euler_{axis}_rad'] = axis_rotation_traveled_rad
                euler_features.update(get_linear_features(data, prefix=f'step{step}_{obj}_euler_{axis}', fs=100))

    for obj in obj_list:
        euler_features[f'step_of_max_total_rotation_traveled_deg_{obj}'] = step_total_rotation_traveled_deg_dict[obj].index(max(step_total_rotation_traveled_deg_dict[obj]))
        euler_features[f'step_of_min_total_rotation_traveled_deg_{obj}'] = step_total_rotation_traveled_deg_dict[obj].index(min(step_total_rotation_traveled_deg_dict[obj]))
        euler_features[f'step_of_max_total_rotation_traveled_rad_{obj}'] = step_total_rotation_traveled_rad_dict[obj].index(max(step_total_rotation_traveled_rad_dict[obj]))
        euler_features[f'step_of_min_total_rotation_traveled_rad_{obj}'] = step_total_rotation_traveled_rad_dict[obj].index(min(step_total_rotation_traveled_rad_dict[obj]))
        euler_features[f'coeff_of_var_of_step_total_rotation_traveled_deg_{obj}'] = calculate_coefficient_of_variation(step_total_rotation_traveled_deg_dict[obj])
        euler_features[f'slope_of_step_total_rotation_traveled_deg_{obj}'] = slope_of_linear_regression(np.arange(len(step_total_rotation_traveled_deg_dict[obj])), step_total_rotation_traveled_deg_dict[obj])
        euler_features.update(get_linear_features(step_total_rotation_traveled_deg_dict[obj], prefix=f'step_total_rotation_traveled_deg_{obj}', fs=None))
        euler_features[f'coeff_of_var_of_step_total_rotation_traveled_rad_{obj}'] = calculate_coefficient_of_variation(step_total_rotation_traveled_rad_dict[obj])
        euler_features[f'slope_of_step_total_rotation_traveled_rad_{obj}'] = slope_of_linear_regression(np.arange(len(step_total_rotation_traveled_rad_dict[obj])), step_total_rotation_traveled_rad_dict[obj])
        euler_features.update(get_linear_features(step_total_rotation_traveled_rad_dict[obj], prefix=f'step_total_rotation_traveled_rad_{obj}', fs=None))
    
    # Substep level
    substep_total_rotation_traveled_deg_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    substep_total_rotation_traveled_rad_dict = {'Head': [], 'LeftHand': [], 'RightHand': []}
    for substep_idx in range(len(tracking_substep_dfs)): # last step is not included as it is imbalanced
        for obj in obj_list:
            obj_df = tracking_substep_dfs[substep_idx][[f'{obj}_euler_x', f'{obj}_euler_y', f'{obj}_euler_z']]
            total_rotation_traveled_deg, total_rotation_traveled_rad = calculate_total_rotation_traveled_3D(obj_df)
            substep_total_rotation_traveled_deg_dict[obj].append(total_rotation_traveled_deg)
            substep_total_rotation_traveled_rad_dict[obj].append(total_rotation_traveled_rad)
            euler_features[f'substep{substep_idx}_{obj}_total_rotation_traveled_deg'] = total_rotation_traveled_deg
            euler_features[f'substep{substep_idx}_{obj}_total_rotation_traveled_rad'] = total_rotation_traveled_rad

            for axis in ['x', 'y', 'z']:
                column_name = f"{obj}_euler_{axis}"
                data = tracking_substep_dfs[substep_idx][column_name].to_numpy()
                euler_features[f'coeff_of_var_of_substep{substep_idx}_{obj}_euler_{axis}'] = calculate_coefficient_of_variation(data)
                euler_features[f'slope_of_substep{substep_idx}_{obj}_euler_{axis}'] = slope_of_linear_regression(np.arange(len(data)), data)
                axis_rotation_traveled_deg, axis_rotation_traveled_rad = calculate_total_rotation_traveled_1D(data)
                euler_features[f'total_rotation_traveled_substep{substep_idx}_{obj}_euler_{axis}_deg'] = axis_rotation_traveled_deg
                euler_features[f'total_rotation_traveled_substep{substep_idx}_{obj}_euler_{axis}_rad'] = axis_rotation_traveled_rad
                euler_features.update(get_linear_features(data, prefix=f'substep{substep_idx}_{obj}_euler_{axis}', fs=100))
        
    for obj in obj_list:
        euler_features[f'substep_of_max_total_rotation_traveled_deg_{obj}'] = substep_total_rotation_traveled_deg_dict[obj].index(max(substep_total_rotation_traveled_deg_dict[obj]))
        euler_features[f'substep_of_min_total_rotation_traveled_deg_{obj}'] = substep_total_rotation_traveled_deg_dict[obj].index(min(substep_total_rotation_traveled_deg_dict[obj]))
        euler_features[f'substep_of_max_total_rotation_traveled_rad_{obj}'] = substep_total_rotation_traveled_rad_dict[obj].index(max(substep_total_rotation_traveled_rad_dict[obj]))
        euler_features[f'substep_of_min_total_rotation_traveled_rad_{obj}'] = substep_total_rotation_traveled_rad_dict[obj].index(min(substep_total_rotation_traveled_rad_dict[obj]))
        euler_features[f'coeff_of_var_of_substep_total_rotation_traveled_deg_{obj}'] = calculate_coefficient_of_variation(substep_total_rotation_traveled_deg_dict[obj])
        euler_features[f'slope_of_substep_total_rotation_traveled_deg_{obj}'] = slope_of_linear_regression(np.arange(len(substep_total_rotation_traveled_deg_dict[obj])), substep_total_rotation_traveled_deg_dict[obj])
        euler_features.update(get_linear_features(substep_total_rotation_traveled_deg_dict[obj], prefix=f'substep_total_rotation_traveled_deg_{obj}', fs=None))
        euler_features[f'coeff_of_var_of_substep_total_rotation_traveled_rad_{obj}'] = calculate_coefficient_of_variation(substep_total_rotation_traveled_rad_dict[obj])
        euler_features[f'slope_of_substep_total_rotation_traveled_rad_{obj}'] = slope_of_linear_regression(np.arange(len(substep_total_rotation_traveled_rad_dict[obj])), substep_total_rotation_traveled_rad_dict[obj])
        euler_features.update(get_linear_features(substep_total_rotation_traveled_rad_dict[obj], prefix=f'substep_total_rotation_traveled_rad_{obj}', fs=None))
    
    # Substep type level
    for substep_type in sorted(tracking_substep_type_dfs.keys()):
        for obj in obj_list:
            total_rotation_traveled_deg = []
            total_rotation_traveled_rad = []
            for substep_idx in range(len(tracking_substep_type_dfs[substep_type])):
                obj_df = tracking_substep_type_dfs[substep_type][substep_idx][[f'{obj}_euler_x', f'{obj}_euler_y', f'{obj}_euler_z']]
                total_rotation_traveled_deg, total_rotation_traveled_rad = calculate_total_rotation_traveled_3D(obj_df)
                total_rotation_traveled_deg.append(total_rotation_traveled_deg)
                total_rotation_traveled_rad.append(total_rotation_traveled_rad)
            euler_features[f'coeff_of_var_of_{substep_type}_total_rotation_traveled_deg_{obj}'] = calculate_coefficient_of_variation(total_rotation_traveled_deg)
            euler_features[f'slope_of_{substep_type}_total_rotation_traveled_deg_{obj}'] = slope_of_linear_regression(np.arange(len(total_rotation_traveled_deg)), total_rotation_traveled_deg)
            euler_features.update(get_linear_features(total_rotation_traveled_deg, prefix=f'{substep_type}_total_rotation_traveled_deg_{obj}', fs=None))
            euler_features[f'coeff_of_var_of_{substep_type}_total_rotation_traveled_rad_{obj}'] = calculate_coefficient_of_variation(total_rotation_traveled_rad)
            euler_features[f'slope_of_{substep_type}_total_rotation_traveled_rad_{obj}'] = slope_of_linear_regression(np.arange(len(total_rotation_traveled_rad)), total_rotation_traveled_rad)
            euler_features.update(get_linear_features(total_rotation_traveled_rad, prefix=f'{substep_type}_total_rotation_traveled_rad_{obj}', fs=None))

    return euler_features

def get_feature_names():
    cfg_file = tsfel.get_features_by_domain()
    features = tsfel.time_series_features_extractor(cfg_file, np.arange(0, 100, 1), fs=100, verbose=0)
    names = features.to_dict().keys()
    names = [i.replace('0_', '').replace(' ', '_') for i in names]
    return names

def get_linear_features(input_data, prefix='', fs=100):
    cfg_file = tsfel.get_features_by_domain()
    features = tsfel.time_series_features_extractor(cfg_file, input_data, fs=fs, verbose=0)
    features = features.to_dict()
    features = [(f"{prefix}_{i.replace('0_', '').replace(' ', '_')}", j[0]) for i,j in features.items()]
    return features

def calculate_coefficient_of_variation(data):
    return np.std(data) / np.mean(data)

def slope_of_linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))  # Covariance
    denominator = np.sum((x - x_mean) ** 2)           # Variance
    return numerator / denominator

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
    hull = ConvexHull(points, qhull_options='QJ')
    return hull.volume

def calculate_distance_between_objects_3D(obj1_df, obj2_df):
    return np.linalg.norm(obj2_df - obj1_df, axis=1)

def calculate_distance_between_objects_1D(obj1_df, obj2_df):
    return np.abs(obj2_df - obj1_df)

def calculate_total_rotation_traveled_1D(euler_array):
    angles_rad = np.deg2rad(euler_array)
    angles_unwrapped = np.unwrap(angles_rad)
    total_rotations_radians = np.sum(np.abs(np.diff(angles_unwrapped)))
    total_rotations_deg = np.rad2deg(total_rotations_radians)
    return total_rotations_deg, total_rotations_radians

def calculate_total_rotation_traveled_3D(euler_array):
    rotations = R.from_euler('xyz', euler_array, degrees=True)
    total_rotation_radians = 0.0
    for i in range(len(rotations) - 1):
        # Relative rotation from i to i+1
        rel_rotation = rotations[i].inv() * rotations[i+1]
        
        # Convert to rotation vector. Its length is the rotation angle in radians.
        rotvec = rel_rotation.as_rotvec()  # shape (3,)
        angle = np.linalg.norm(rotvec)     # angle in radians
        
        # Accumulate
        total_rotation_radians += angle

    # (Optional) Convert to degrees
    total_rotation_degs = np.degrees(total_rotation_radians)
    return total_rotation_degs, total_rotation_radians

# def calculate_lyapunov_exponent(velocity_series):
#     data = velocity_series.drop_nulls().to_numpy()
#     max_lag = 100
#     n = len(data)
#     mean = np.mean(data)
#     var = np.var(data)
#     ac = np.array([np.sum((data[:n - lag] - mean) * (data[lag:] - mean)) / ((n - lag) * var)
#                 for lag in range(max_lag)])
#     threshold_ac = 1 / np.e
#     min_tstep = np.argmax(ac < threshold_ac)
#     return nolds.lyap_r(data, min_tsep=min_tstep)

# from .NONANLibrary.LyE_R import *
# class NonlinearAnalysis:
#     def __init__(self, velocity_series):
#         self.velocity_series = velocity_series.drop_nulls().to_numpy()
#         self.optimal_lag = None
#         self.optimal_emb_dim = None
#         self.min_tstep = None
#         # self.init_parameters()

#     def average_mutual_information(self, max_lag):
#         ami = []
#         # Convert velocity series to non-negative integers
#         min_val = np.min(self.velocity_series)
#         shifted_series = self.velocity_series - min_val  # Shift to make all values non-negative
#         # Convert to integers (required by pyinform)
#         discretized_series = np.floor(shifted_series * 1000).astype(int)  # Scale and convert to integers
        
#         for lag in range(1, max_lag + 1):
#             mi = mutual_info(discretized_series[:-lag], discretized_series[lag:], local=False)
#             ami.append(mi)
#         return np.array(ami)
    
#     def autocorrelation(self, max_lag):
#         n = len(self.velocity_series)
#         mean = np.mean(self.velocity_series)
#         var = np.var(self.velocity_series)
#         ac = np.array([np.sum((self.velocity_series[:n - lag] - mean) * (self.velocity_series[lag:] - mean)) / ((n - lag) * var)
#                     for lag in range(max_lag)])
#         return ac
    
#     def init_parameters(self):
#         # max_lag = 100
#         # ami_values = self.average_mutual_information(max_lag)
#         # self.optimal_lag = np.argmin(ami_values) + 1

#         # max_dim = 10
#         # dims = np.arange(1, max_dim + 1)
#         # fnn_percent = dimension.fnn(self.velocity_series, tau=self.optimal_lag, dim=dims)[0]
#         # threshold = 10.0
#         # self.optimal_emb_dim = dims[np.where(fnn_percent < threshold)[0][0]]

#         max_ac_lag = 100
#         ac_values = self.autocorrelation(max_ac_lag)
#         threshold_ac = 1 / np.e
#         self.min_tstep = np.argmax(ac_values < threshold_ac)

#     def calculate_lyapunov_exponent(self):
#         # return LyE_R(self.velocity_series, 100, self.optimal_lag, self.optimal_emb_dim)
#         return nolds.lyap_r(self.velocity_series, min_tsep=self.min_tstep)
    