import json, pickle, pandas as pd

useIBT = True

with open("/srv/STP/experimentData/pos_quat_step_df.pkl", "rb") as f:
    pqdf = pickle.load(f)

with open("/srv/STP/experimentData/450shift1_central_step_df.pkl", "rb") as f:
    noIBT = pickle.load(f)

# with open("/srv/STP/experimentData/450shift9_central_IBT_step_df.pkl", "rb") as f:
#     IBT = pickle.load(f)

noIBTdf = pqdf.merge(noIBT, on=['PID', 'task_id'], how='outer')
# IBTdf = pqdf.merge(IBT, on=['PID', 'task_id'], how='outer')

device_combos = [('Head', 'LeftHand', 'RightHand')]

mt_combos = [('pos',), ('quat',), ('pos', 'quat'),
             ('linvel',), ('angvel',), ('linvel', 'angvel'),
             ('pos', 'linvel'), ('quat', 'angvel'), ('pos', 'quat', 'linvel', 'angvel')]

if useIBT:
    features = IBTdf.columns.to_list()
else:
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
    json.dump(feats, open(f"/srv/STP/study_1_configs_{'IBT' if useIBT else 'noIBT'}/{name}.json", "w"))