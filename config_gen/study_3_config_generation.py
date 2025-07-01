import json, pickle, pandas as pd

useIBT = False

with open("/srv/STP/experimentData/pos_quat_step_df.pkl", "rb") as f:
    pqdf = pickle.load(f)

with open("/srv/STP/experimentData/450shift1_central_noIBT_step_df.pkl", "rb") as f:
    noIBT = pickle.load(f)

# with open("/srv/STP/experimentData/450shift9_central_IBT_step_df.pkl", "rb") as f:
#     IBT = pickle.load(f)

noIBTdf = pqdf.merge(noIBT, on=['PID', 'task_id'], how='outer')
# IBTdf = pqdf.merge(IBT, on=['PID', 'task_id'], how='outer')

device_combos = [('Head', 'LeftHand', 'RightHand')]

mt_combos = [
    # Linear Angular
    ('linjerk', 'angjerk'), 
    ('pos', 'quat', 'linjerk', 'angjerk'), 
    ('linvel', 'angvel', 'linjerk', 'angjerk'),
    ('linacc', 'angacc', 'linjerk', 'angjerk'),
    ('pos', 'quat', 'linvel', 'angvel', 'linjerk', 'angjerk'),
    ('pos', 'quat', 'linacc', 'angacc', 'linjerk', 'angjerk'),
    ('linvel', 'angvel', 'linacc', 'angacc', 'linjerk', 'angjerk'),
    ('pos', 'quat', 'linvel', 'angvel', 'linacc', 'angacc', 'linjerk', 'angjerk'),

    # Linear
    ('linjerk',),
    ('pos', 'linjerk'),
    ('linvel', 'linjerk'),
    ('linacc', 'linjerk'),
    ('pos', 'linvel', 'linjerk'),
    ('pos', 'linacc', 'linjerk'),
    ('linvel', 'linacc', 'linjerk'),
    ('pos', 'linvel', 'linacc', 'linjerk'),

    # Angular
    ('angjerk',),
    ('quat', 'angjerk'),
    ('angvel', 'angjerk'),
    ('angacc', 'angjerk'),
    ('quat', 'angvel', 'angjerk'),
    ('quat', 'angacc', 'angjerk'),
    ('angvel', 'angacc', 'angjerk'),
    ('quat', 'angvel', 'angacc', 'angjerk'),
    ]

# if useIBT:
#     features = IBTdf.columns.to_list()
# else:
features = noIBTdf.columns.to_list()

configs = {}
for dcmb in device_combos:
    for mt in mt_combos:
        subset = []
        for device in dcmb:
            for mt_term in mt:
                subset.extend([i for i in features if device in i and mt_term in i])
        configs[f"{'_'.join(dcmb)}_{'_'.join(mt)}_{'IBT' if useIBT else 'noIBT'}"] = subset

for name, feats in configs.items():
    json.dump(feats, open(f"/srv/STP/study_3_configs_{'IBT' if useIBT else 'noIBT'}/{name}.json", "w"))