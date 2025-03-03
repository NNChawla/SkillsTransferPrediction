import pandas as pd
import os, sys

assembly_dir = '/srv/STP/data/FAST/Assembly'
tracking_dir = '/srv/STP/data/FAST/Tracking'
output_dir = '/srv/STP/data/FAB/Assembly_'

os.makedirs(output_dir + 'A', exist_ok=True)
os.makedirs(output_dir + 'B', exist_ok=True)

for file_name in os.listdir(tracking_dir):
    build_letter = file_name.split('_')[1][-1]
    if build_letter == 'l':
        continue
    
    df = pd.read_csv(f'{tracking_dir}/{file_name}', header=0)
    df2 = pd.read_csv(f'{assembly_dir}/{file_name}', header=0)

    step_time_in_seconds = (pd.to_datetime(df2['Timestamp']) - pd.to_datetime(df['Timestamp'].iloc[0])).dt.total_seconds()
    df2['Timestamp'] = step_time_in_seconds
    df2.set_index('Timestamp', inplace=True)

    df2.to_csv(f'{output_dir}{build_letter}/{file_name}', index=True)