from itertools import combinations
import pickle, os

dir = "./results_pca_001-10C_04-1K_07P_07SP_gt02P_0.271_0.875"

files = sorted(os.listdir(dir))

files = [i for i in files if "final" in i]

results = []
for name in files:
    with open(f"{dir}/{name}", "rb") as f:
        results.append((name.replace("271_0.875_Step_", "").replace("_final_results.pkl", ""), pickle.load(f)))

results_dict = dict(results)

for key in results_dict.keys():
    results_dict[key] = results_dict[key][0]

anova_data = {}
for key in results_dict.keys():
    anova_data[key] = [i["mcc"] for i in results_dict[key]["kfold_results"]]

anova_data_correct = {}

device_types = ["Head", "RightHand", "LeftHand"]
motion_types = ["linvel", "linacc", "angvel", "angacc"]

dvc_cmbs = []
for i in range(1, len(device_types)+1):
    dvc_cmbs.extend(list(combinations(device_types, i)))

mt_cmbs = []
for i in range(1, len(motion_types)+1):
    mt_cmbs.extend(list(combinations(motion_types, i)))

all_cmbs = []
for i in dvc_cmbs:
    for j in mt_cmbs:
        all_cmbs.append((i, j))

for cmb in all_cmbs:
    key = "_".join(cmb[0])+"_"+"_".join(cmb[1])
    anova_data_correct[cmb] = anova_data[key]