from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     StratifiedShuffleSplit, cross_val_score)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from wrappers import *
import pickle, time, gc, sys, os, json
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
from JointBorutaShap import *
import optuna
import random

os.makedirs("./results_server", exist_ok=True)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

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
total_run_time = 60
trial_run_time = None
trial_num = 1000
na_threshold = 0.2

tab_path = "./experimentData/tabulated_dataframe.pkl"
md_path = "./experimentData/metadata.csv"
with open(tab_path, "rb") as f:
    tabulated_dataframe = pickle.load(f)
metadata_dataframe = pd.read_csv(md_path)

include_set_feature_A, include_set_feature_B = ["RW_Build_Time_A"], ["RW_Build_Time_B"]
exclude_set_feature_A, exclude_set_feature_B = ["RW_Build_Time_B"], ["RW_Build_Time_A"]
drop_columns = ["Score_A", "Score_B", "Score_A_Linear", "Score_B_Linear"]
drop_columns_A = drop_columns + exclude_set_feature_A
drop_columns_B = drop_columns + exclude_set_feature_B
scalable_metadata_features = ["hour_of_day", "time_sin", "time_cos", "TotalDuration"]
scalable_metadata_features_A = scalable_metadata_features + include_set_feature_A
scalable_metadata_features_B = scalable_metadata_features + include_set_feature_B

metadata_df_A = metadata_dataframe.copy()
metadata_df_B = metadata_dataframe.copy()
metadata_df_A = metadata_df_A.drop(columns=drop_columns_A)
metadata_df_B = metadata_df_B.drop(columns=drop_columns_B)
smdf_A = metadata_df_A[scalable_metadata_features_A].copy()
smdf_B = metadata_df_B[scalable_metadata_features_B].copy()
mdf_A = metadata_df_A.drop(columns=scalable_metadata_features_A + ["PID"])
mdf_B = metadata_df_B.drop(columns=scalable_metadata_features_B + ["PID"])

metadata_binary_features = [i for i in mdf_A.columns if ("-" in i)]
metadata_binary_features = list(set([i.replace("A-", "X-").replace("B-", "X-").replace("_A", "_X").replace("_B", "_X") for i in metadata_binary_features]))
metadata_unique_features = [i for i in mdf_A.columns if not ("-" in i)]
metadata_all_unique_features = metadata_unique_features + metadata_binary_features
metadata_A_features, metadata_B_features = list(zip(*[(i.replace("X-", "A-").replace("_X", "_A"), i.replace("X-", "B-").replace("_X", "_B")) for i in metadata_binary_features]))
metadata_A_features = metadata_unique_features + list(metadata_A_features)
metadata_B_features = metadata_unique_features + list(metadata_B_features)
mdf_A = mdf_A[metadata_A_features]
mdf_B = mdf_B[metadata_B_features]

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
random.shuffle(unique_features)
features_A = [f"{i}_A" for i in unique_features]
features_B = [f"{i}_B" for i in unique_features]
tdf_A = tdf[features_A]
tdf_B = tdf[features_B]

score_A = metadata_dataframe["Score_A_Linear"].apply(lambda x: 0 if x > 0 else 1)
score_B = metadata_dataframe["Score_B_Linear"].apply(lambda x: 0 if x > 0 else 1)
sample_weight_A = compute_sample_weight(class_weight="balanced", y=score_A)
sample_weight_B = compute_sample_weight(class_weight="balanced", y=score_B)

# def get_strings_with_substrings(string_list, target_strings):
#     valid_strings = []
#     for target_string in target_strings:
#         if all(s in target_string for s in string_list):
#             valid_strings.append(target_string)
#     valid_strings = sorted(valid_strings)
#     return valid_strings

for cols in range(0, len(unique_features), 5000):
    unique_features_subset = unique_features[cols:min(cols+5000, len(unique_features))]
    subset_features_A = tdf_A[[f"{i}_A" for i in unique_features_subset]]
    subset_features_B = tdf_B[[f"{i}_B" for i in unique_features_subset]]
    subset_features_A = scale_impute_df(subset_features_A, scaler, imputer)
    subset_features_B = scale_impute_df(subset_features_B, scaler, imputer)

    inner_cv = StratifiedShuffleSplit(n_splits = 2, test_size=0.3, random_state=25)
    xgb = XGBClassifier(random_state=25, n_jobs=-1)
    
    fs = JointBorutaShap(model=xgb, cv=inner_cv, importance_measure='gini', classification=True)
    _ = fs.fit(subset_features_A, score_A, subset_features_B, score_B, sample_weight_A, sample_weight_B, use_sampling=True)

    with open(f"./boruta_results/fs_{cols}.pkl", "wb") as f:
        pickle.dump(fs, f)