import numpy as np
import pandas as pd
import pickle
import os, json, sys, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def drop_high_na_columns(input_df, na_threshold):
    df = input_df.copy()
    na_counts = df.isna().sum()
    columns_to_drop = na_counts[na_counts > na_threshold].index
    return df.drop(columns=columns_to_drop)

def variance_filter(input_df):
    df = input_df.copy()
    variances = df.var()
    low_variance_features = variances[variances < 0.01].index
    na_variance_features = variances[variances.isna()].index
    df = df.drop(columns=low_variance_features)
    df = df.drop(columns=na_variance_features)
    return df

def scale_impute_df(input_df, scaler, imputer):
    df = input_df.copy()
    df = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(df)), columns=df.columns)
    return df

# Preprocessing
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
random_state = 25
na_threshold = 0.2

tab_path = "./experimentData/tabulated_dataframe.pkl"
with open(tab_path, "rb") as f:
    tabulated_dataframe = pickle.load(f)

PID = tabulated_dataframe["PID"]
tdf = tabulated_dataframe.drop(columns=["PID"])
tdf = drop_high_na_columns(tdf, int(tdf.shape[0] * na_threshold))
tdf = variance_filter(tdf)
tdf = pd.concat([PID, tdf], axis=1)
tdf = tdf.iloc[:,1:]
all_feature_names = tdf.columns.to_list()
all_feature_names_set = set(all_feature_names)
unique_features = sorted(list(set(["_".join(i.split('_')[:-1]) for i in all_feature_names])))
unique_features = [i for i in unique_features if ((f"{i}_A" in all_feature_names_set) and (f"{i}_B" in all_feature_names_set))]
features_A = [f"{i}_A" for i in unique_features]
features_B = [f"{i}_B" for i in unique_features]
tdf_A = tdf[features_A]
tdf_B = tdf[features_B]

within_point_biserial_features = pd.read_csv("./_within_task_statistic_results/threshold_0/top_valid_within_point_biserial_features.csv")
within_point_biserial_features = within_point_biserial_features['Feature'].tolist()

pbdf_A = tdf_A[[f"{i}_A" for i in within_point_biserial_features]]
pbdf_B = tdf_B[[f"{i}_B" for i in within_point_biserial_features]]

pbdf_A.columns = within_point_biserial_features
pbdf_B.columns = within_point_biserial_features

pbdf_A = scale_impute_df(pbdf_A, scaler, imputer)
pbdf_B = scale_impute_df(pbdf_B, scaler, imputer)

corr_A = pbdf_A.corr().abs()
corr_B = pbdf_B.corr().abs()

corr = corr_A + corr_B
corr = corr / 2

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
pbdf = pbdf_A.drop(to_drop, axis=1)
remaining_features = pbdf.columns.to_list()

print(f"Number of remaining features: {len(remaining_features)}")

with open("/srv/STP/filtered_point_biserial_features.json", "w") as f:
    json.dump(remaining_features, f)

devices = ["Head", "LeftHand", "RightHand"]
motion_types = ["linvel", "angvel", "linacc", "angacc"]