import pickle
import pandas as pd

with open('./results/all_experiments_results.pkl', 'rb') as f:
    results = pickle.load(f)

labels = [i for i in results.keys()]
labels = sorted(labels)
mcc_results = [results[i]['feature_selection_measure'].iloc[0,0] for i in labels]
mcc_std_results = [results[i]['feature_selection_measure_std'].iloc[0,0] for i in labels]

results_df = pd.DataFrame(data={'label': labels, 'mcc': mcc_results, 'mcc_std': mcc_std_results})

sorted_cross_task_labels = [j for i, j in enumerate(labels) if (i % 2 == 0) and (j.split('random_forest_')[1].split('_')[0] == 'True')]
sorted_mcc_results = [(results[i]['feature_selection_measure'].iloc[0,0],results[f"{'_'.join(i.split('_')[:-1])}_{'B'}"]['feature_selection_measure'].iloc[0,0]) for i in sorted_cross_task_labels]
sorted_mcc_std_results = [(results[i]['feature_selection_measure_std'].iloc[0,0],results[f"{'_'.join(i.split('_')[:-1])}_{'B'}"]['feature_selection_measure_std'].iloc[0,0]) for i in sorted_cross_task_labels]

sorted_results_df = pd.DataFrame(data={'label': sorted_cross_task_labels, 'mcc': sorted_mcc_results, 'mcc_std': sorted_mcc_std_results})

#top_indices = [69, 197, 152, 24, 56, 184, 85, 213]
#top_labels = [labels[i] for i in top_indices]