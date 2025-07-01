import os, json, shutil

if os.path.exists("./experimentConfigs"):
    shutil.rmtree("./experimentConfigs")
os.makedirs("./experimentConfigs", exist_ok=True)

pairs = []
for i in [450]:
    for j in [1.0]:
        pairs.append((i, j))

step_configs = sorted(os.listdir("./step_configs"))
for window, threshold in pairs:
    for config in step_configs:
        with open(f"./step_configs/{config}", "r") as f:
            loaded_config = json.load(f)
        with open(f"./experimentConfigs/{window}_{threshold}_{config}", "w") as f:
            tmp_config = [i.replace("_451_", f"_{window}_").replace("_1_", f"_{threshold}_") for i in loaded_config]
            json.dump(tmp_config, f)

# device_combos = []
# for i in range(1, len(devices)+1):
#     device_combos.extend(list(combinations(devices, i)))

# mt_combos = [('pos',), ('quat',), ('pos', 'quat')]

# configs = {}
# for dcmb in device_combos:
#     for mt in mt_combos:
#         subset = []
#         for device in dcmb:
#             for mt_term in mt:
#                 subset.extend([i for i in features if device in i and mt_term in i])
#         configs[f"{'_'.join(dcmb)}_{'_'.join(mt)}"] = subset

# print(configs)

# for name, feats in configs.items():
#     json.dump(feats, open(f"{name}.json", "w"))