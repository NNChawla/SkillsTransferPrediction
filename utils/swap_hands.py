import pandas as pd
import os, sys

assembly_dir = '/srv/STP/data/FAST/Assembly'
tracking_dir = '/srv/STP/data/FAST/Tracking'
assembly_output_dir = '/srv/STP/data/FAB/Assembly_'
tracking_output_dir = '/srv/STP/data/FAB/Tracking_'

valid_pids = ['FAB10068F', 'FAB11629F', 'FAB12013F', 'FAB12101F', 'FAB13229M', 'FAB15389M', 'FAB15605M', 'FAB15800F', 'FAB16401F', 'FAB16803F', 'FAB17745M', 'FAB19823F', 'FAB20077M', 'FAB20552M', 'FAB21337F', 'FAB21952F', 'FAB22008F', 'FAB26097M', 'FAB26282M', 'FAB28076M', 'FAB28505M', 'FAB30237M', 'FAB31369F', 'FAB31608M', 'FAB33054M', 'FAB34621M', 'FAB34961M', 'FAB35565M', 'FAB36376F', 'FAB36517M', 'FAB36603M', 'FAB37469M', 'FAB37970F', 'FAB41088M', 'FAB43500F', 'FAB43889M', 'FAB45085F', 'FAB45306F', 'FAB45958M', 'FAB45975F', 'FAB46373M', 'FAB48763F', 'FAB50464F', 'FAB50469M', 'FAB52729M', 'FAB54272M', 'FAB55695M', 'FAB56743M', 'FAB57723M', 'FAB58430M', 'FAB59020M', 'FAB60184F', 'FAB60610M', 'FAB60686M', 'FAB61038F', 'FAB62159M', 'FAB62905F', 'FAB63558M', 'FAB64182M', 'FAB66558M', 'FAB66731F', 'FAB67564F', 'FAB67616M', 'FAB68232F', 'FAB68275M', 'FAB68308M', 'FAB68627M', 'FAB70206M', 'FAB71291M', 'FAB71520F', 'FAB71966M', 'FAB72162F', 'FAB75290M', 'FAB77684F', 'FAB78353M', 'FAB78531F', 'FAB79396F', 'FAB79662M', 'FAB81762M', 'FAB82909M', 'FAB83960M', 'FAB85470F', 'FAB85810F', 'FAB86414M', 'FAB90081F', 'FAB90539M', 'FAB90971F', 'FAB91936F', 'FAB92577F', 'FAB92623F', 'FAB92674F', 'FAB93490F', 'FAB94722F', 'FAB95060F', 'FAB95225M', 'FAB95341F', 'FAB95381F', 'FAB95446F', 'FAB95536F', 'FAB96284M', 'FAB96418F', 'FAB96814M', 'FAB97562M', 'FAB97837F', 'FAB99585F']
lrs_pids = ["FAB12013F_BuildA", "FAB12101F_BuildA", "FAB16803F_BuildA", "FAB20077M_BuildA", "FAB20552M_BuildA", "FAB21952F_BuildA", "FAB22008F_BuildA", "FAB26097M_BuildA", "FAB26146M_BuildA", "FAB34621M_BuildA", "FAB34961M_BuildA", "FAB36517M_BuildA", "FAB37970F_BuildA", "FAB41088M_BuildA", "FAB43500F_BuildA", "FAB45958M_BuildA", "FAB45975F_BuildA", "FAB46373M_BuildA", "FAB52866F_BuildA", "FAB54272M_BuildA", "FAB56743M_BuildA", "FAB59020M_BuildA", "FAB60610M_BuildA", "FAB60686M_BuildA", "FAB62905F_BuildA", "FAB64182M_BuildA", "FAB67616M_BuildA", "FAB68627M_BuildA", "FAB71291M_BuildA", "FAB72162F_BuildA", "FAB77684F_BuildA", "FAB82909M_BuildA", "FAB85470F_BuildA", "FAB91936F_BuildA", "FAB93490F_BuildA", "FAB95060F_BuildA", "FAB95341F_BuildA"]
lrs_pids.extend(["FAB16401F_BuildB", "FAB21337F_BuildB", "FAB26097M_BuildB", "FAB26282M_BuildB", "FAB36376F_BuildB", "FAB36603M_BuildB", "FAB41088M_BuildB", "FAB43889M_BuildB", "FAB45306F_BuildB", "FAB45975F_BuildB", "FAB48763F_BuildB", "FAB50464F_BuildB", "FAB59020M_BuildB", "FAB62159M_BuildB", "FAB63558M_BuildB", "FAB66731F_BuildB", "FAB67616M_BuildB", "FAB68232F_BuildB", "FAB70206M_BuildB", "FAB71520F_BuildB", "FAB77684F_BuildB", "FAB79662M_BuildB", "FAB86414M_BuildB", "FAB90539M_BuildB", "FAB91936F_BuildB", "FAB92674F_BuildB", "FAB94722F_BuildB", "FAB95225M_BuildB", "FAB95341F_BuildB", "FAB95381F_BuildB", "FAB95536F_BuildB", "FAB96814M_BuildB", "FAB97562M_BuildB", "FAB97837F_BuildB"])

os.makedirs(assembly_output_dir + 'A', exist_ok=True)
os.makedirs(assembly_output_dir + 'B', exist_ok=True)
os.makedirs(tracking_output_dir + 'A', exist_ok=True)
os.makedirs(tracking_output_dir + 'B', exist_ok=True)

files = sorted(os.listdir(tracking_dir))

for file_name in files:
    build_letter = file_name.split('_')[1][-1]
    if 'Tutorial' in file_name:
        continue
    if not (file_name.split('_')[0] in valid_pids):
        continue
    
    tracking_df = pd.read_csv(f'{tracking_dir}/{file_name}', header=0)
    assembly_df = pd.read_csv(f'{assembly_dir}/{file_name}', header=0)

    step_time_in_seconds = (pd.to_datetime(assembly_df['Timestamp']) - pd.to_datetime(tracking_df['Timestamp'].iloc[0])).dt.total_seconds()
    assembly_df['Timestamp'] = step_time_in_seconds
    assembly_df.set_index('Timestamp', inplace=True)

    session_time_in_seconds = (pd.to_datetime(tracking_df['Timestamp']) - pd.to_datetime(tracking_df['Timestamp'].iloc[0])).dt.total_seconds()
    tracking_df['Timestamp'] = session_time_in_seconds
    tracking_df.set_index('Timestamp', inplace=True)

    tracking_file_part = "_".join(file_name.split('_')[:2])
    if tracking_file_part in lrs_pids:
        column_names = [i.replace('Right', 'Wight') for i in tracking_df.columns]
        column_names = [i.replace('Left', 'Right') for i in column_names]
        column_names = [i.replace('Wight', 'Left') for i in column_names]
        tracking_df.columns = column_names

    assembly_df.to_csv(f'{assembly_output_dir}{build_letter}/{file_name}', index=True)
    tracking_df.to_csv(f'{tracking_output_dir}{build_letter}/{file_name}', index=True)