import json

with open("./experimentConfigs/pause_step_features.json", "r") as f:
    features = json.load(f)

windows = ["9", "91", "181", "271", "361", "451", "541"]
thresholds = ["0.5", "0.75", "1", "1.25"]
for wd in windows:
    for thd in thresholds:
        mad_window_features = []
        for sublist in features:
            selected_features = [i for i in sublist if (f"_{thd}_" in i) and (f"{wd}" in i)]
            if (len(selected_features) > 0):
                mad_window_features.append(["prod"] + selected_features)
        with open(f"./experimentConfigs/MAD_{thd}_Window_{wd}.json", "w") as f:
            json.dump(mad_window_features, f)