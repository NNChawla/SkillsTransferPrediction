import pickle, os, gc
import numpy as np
import pandas as pd

participant_ids = [i.split('_')[0] for i in sorted(os.listdir("/srv/STP/data/FAB/FAB_A_Step_Motion"))]
num_participants = len(participant_ids)
tabulated_data = {
    "PID": participant_ids
}

with open("/srv/STP/results_0.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

with open("/srv/STP/results_1.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

with open("/srv/STP/results_2.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

with open("/srv/STP/results_3.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

with open("/srv/STP/results_4.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

tabulated_dataframe = pd.DataFrame(tabulated_data)
del tabulated_data
gc.collect()

with open("/srv/STP/tabulated_dataframe.pkl", "wb") as f:
    pickle.dump(tabulated_dataframe, f)