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