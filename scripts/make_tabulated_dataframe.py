import pickle, os, gc
import numpy as np
import pandas as pd

participant_ids = [i.split('_')[0] for i in sorted(os.listdir("/srv/STP/data/FAB/FAB_A_Step_Motion_Finite_Std"))]
num_participants = len(participant_ids)
tabulated_data = {
    "PID": participant_ids
}

import time
start_time = time.time()
print("Processing results_1.pkl")

with open("/srv/STP/results_1.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()
print(f"Processing results_1.pkl took {time.time() - start_time} seconds")

start_time = time.time()
print("Processing results_2.pkl")

with open("/srv/STP/results_2.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

print(f"Processing results_2.pkl took {time.time() - start_time} seconds")

start_time = time.time()
print("Processing results_3.pkl")

with open("/srv/STP/results_3.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

print(f"Processing results_3.pkl took {time.time() - start_time} seconds")

start_time = time.time()
print("Processing results_4.pkl")

with open("/srv/STP/results_4.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

print(f"Processing results_4.pkl took {time.time() - start_time} seconds")

start_time = time.time()
print("Processing results_5.pkl")

with open("/srv/STP/results_5.pkl", "rb") as f:
    results = pickle.load(f)

for i, result_dict in results:
    for key, value in result_dict.items():
        if key not in tabulated_data:
            tabulated_data[key] = [np.nan] * num_participants
        tabulated_data[key][i] = value
del results
gc.collect()

print(f"Processing results_5.pkl took {time.time() - start_time} seconds")

start_time = time.time()
print("Creating tabulated dataframe")

tabulated_dataframe = pd.DataFrame(tabulated_data)
del tabulated_data
gc.collect()
print(f"Creating tabulated dataframe took {time.time() - start_time} seconds")
with open("/srv/STP/450shift450_tabulated_dataframe.pkl", "wb") as f:
    pickle.dump(tabulated_dataframe, f)