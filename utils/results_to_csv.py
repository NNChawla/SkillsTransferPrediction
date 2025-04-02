import pandas as pd
import pickle, os, gc
import numpy as np

with open("/srv/STP/results_1.pkl", "rb") as f:
    results = pickle.load(f)
participant_ids = [i.split('_')[0] for i in sorted(os.listdir("/srv/STP/data/FAB/FAB_A_Motion"))]
num_participants = len(participant_ids)
tabulated_data = {
    "PID": participant_ids
}

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value

del results
del result_dict
del f
gc.collect()