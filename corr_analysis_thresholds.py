import pickle, os, gc, time, csv
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pointbiserialr
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

def safe_harmonic_mean(x, y):
    if x < 0 or y < 0:
        return 0
    if x == 0 or y == 0:
        return 0
    return 2 / ((1 / x) + (1 / y))

def metric_harmonic_mean(x, y):
    return 2 / ((1 / abs(x)) + (1 / abs(y)))

tab_path = "/srv/STP/experimentData/tabulated_dataframe.pkl"
md_path = "/srv/STP/experimentData/metadata.csv"
with open(tab_path, "rb") as f:
    tabulated_dataframe = pickle.load(f)
all_features = tabulated_dataframe.columns
metadata_dataframe = pd.read_csv(md_path)
score_A_Linear = metadata_dataframe["Score_A_Linear"]
score_B_Linear = metadata_dataframe["Score_B_Linear"]

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

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

###########################################################################################
# Preprocessing (in line with experiment setup)
###########################################################################################

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
metadata_df_A = metadata_df_A.rename(columns={"ID": "PID"})
metadata_df_B = metadata_df_B.rename(columns={"ID": "PID"})
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
unscaled_tabulated_dataframe = tabulated_dataframe.drop(columns=["PID"])
unscaled_tabulated_dataframe = drop_high_na_columns(unscaled_tabulated_dataframe, int(unscaled_tabulated_dataframe.shape[0] * 0.2))
unscaled_tabulated_dataframe = variance_filter(unscaled_tabulated_dataframe)
unscaled_tabulated_dataframe = pd.DataFrame(imputer.fit_transform(unscaled_tabulated_dataframe), columns=unscaled_tabulated_dataframe.columns)
unscaled_tabulated_dataframe = pd.concat([PID, unscaled_tabulated_dataframe], axis=1)
utdf = unscaled_tabulated_dataframe.iloc[:,1:].copy()

PID = tabulated_dataframe["PID"]
tabulated_dataframe = tabulated_dataframe.drop(columns=["PID"])
tabulated_dataframe = drop_high_na_columns(tabulated_dataframe, int(tabulated_dataframe.shape[0] * 0.2))
tabulated_dataframe = variance_filter(tabulated_dataframe)
tabulated_dataframe = scale_impute_df(tabulated_dataframe, scaler, imputer)
tabulated_dataframe = pd.concat([PID, tabulated_dataframe], axis=1)
tdf = tabulated_dataframe.iloc[:,1:].copy()
all_feature_names = tdf.columns.to_list()
all_feature_names_set = set(all_feature_names)
unique_features = sorted(list(set(["_".join(i.split('_')[:-1]) for i in all_feature_names])))
unique_features = [i for i in unique_features if ((f"{i}_A" in all_feature_names_set) and (f"{i}_B" in all_feature_names_set))]
features_A = [f"{i}_A" for i in unique_features]
features_B = [f"{i}_B" for i in unique_features]
tdf_A = tdf[features_A]
tdf_B = tdf[features_B]
utdf_A = utdf[features_A]
utdf_B = utdf[features_B]

thresholds = [2, 4, 6, 8, 10, 12]
for threshold in thresholds:
    score_A = score_A_Linear.apply(lambda x: 1 if x > threshold else 0)
    score_B = score_B_Linear.apply(lambda x: 1 if x > threshold else 0)
    pth_cross = f"/srv/STP/_cross_task_statistic_results/threshold_{threshold}"
    pth_within = f"/srv/STP/_within_task_statistic_results/threshold_{threshold}"
    os.makedirs(pth_cross, exist_ok=True)
    os.makedirs(pth_within, exist_ok=True)

    ###########################################################################################
    # F-Statistic
    ###########################################################################################
    print(f"Processing F-Statistic for threshold {threshold}")
    print(f"Processing Cross-task F-Statistic")
    t = time.time()
    f_statistic_cross_A, fstat_p_values_cross_A = f_classif(tdf_A, score_B)
    f_statistic_cross_B, fstat_p_values_cross_B = f_classif(tdf_B, score_A)
    filter_fstat_p_values_cross_A_indices = set([i for i in range(len(fstat_p_values_cross_A)) if fstat_p_values_cross_A[i] <= 0.05])
    filter_fstat_p_values_cross_B_indices = set([i for i in range(len(fstat_p_values_cross_B)) if fstat_p_values_cross_B[i] <= 0.05])
    filter_fstat_p_values_cross_indices = sorted(list(filter_fstat_p_values_cross_A_indices.intersection(filter_fstat_p_values_cross_B_indices)))
    top_valid_cross_fstat_features = []
    for i in filter_fstat_p_values_cross_indices:
        tpl = [unique_features[i], f_statistic_cross_A[i], f_statistic_cross_B[i], safe_harmonic_mean(f_statistic_cross_A[i], f_statistic_cross_B[i])]
        top_valid_cross_fstat_features.append(tpl)
    sorted_top_valid_cross_fstat_features = sorted(top_valid_cross_fstat_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_cross}/top_valid_cross_fstat_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "F-Statistic_A", "F-Statistic_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_cross_fstat_features)
    print(f"Cross-task F-Statistic completed in {time.time() - t} seconds")
    print(f"Processing Within-task F-Statistic")
    t = time.time()
    f_statistic_within_A, fstat_p_values_within_A = f_classif(tdf_A, score_A)
    f_statistic_within_B, fstat_p_values_within_B = f_classif(tdf_B, score_B)
    filter_fstat_p_values_within_A_indices = set([i for i in range(len(fstat_p_values_within_A)) if fstat_p_values_within_A[i] <= 0.05])
    filter_fstat_p_values_within_B_indices = set([i for i in range(len(fstat_p_values_within_B)) if fstat_p_values_within_B[i] <= 0.05])
    filter_fstat_p_values_within_indices = sorted(list(filter_fstat_p_values_within_A_indices.intersection(filter_fstat_p_values_within_B_indices)))
    top_valid_within_fstat_features = []
    for i in filter_fstat_p_values_within_indices:
        tpl = [unique_features[i], f_statistic_within_A[i], f_statistic_within_B[i], safe_harmonic_mean(f_statistic_within_A[i], f_statistic_within_B[i])]
        top_valid_within_fstat_features.append(tpl)
    sorted_top_valid_within_fstat_features = sorted(top_valid_within_fstat_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_within}/top_valid_within_fstat_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "F-Statistic_A", "F-Statistic_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_within_fstat_features)
    print(f"Within-task F-Statistic completed in {time.time() - t} seconds")

    ###########################################################################################
    # Spearman Correlation
    ###########################################################################################
    print(f"Processing Spearman Correlation for threshold {threshold}")
    print(f"Processing Cross-task Spearman Correlation")
    t = time.time()
    spearman_correlation_cross_A = tdf_A.corrwith(score_B, method="spearman")
    spearman_correlation_cross_B = tdf_B.corrwith(score_A, method="spearman")
    filter_spearman_cross_A_indices = set([i for i in range(len(spearman_correlation_cross_A)) if abs(spearman_correlation_cross_A[i]) >= 0.2])
    filter_spearman_cross_B_indices = set([i for i in range(len(spearman_correlation_cross_B)) if abs(spearman_correlation_cross_B[i]) >= 0.2])
    filter_spearman_cross_indices = sorted(list(filter_spearman_cross_A_indices.intersection(filter_spearman_cross_B_indices)))
    top_valid_cross_spearman_features = []
    for i in filter_spearman_cross_indices:
        tpl = [unique_features[i], spearman_correlation_cross_A[i], spearman_correlation_cross_B[i], metric_harmonic_mean(spearman_correlation_cross_A[i], spearman_correlation_cross_B[i])]
        top_valid_cross_spearman_features.append(tpl)
    sorted_top_valid_cross_spearman_features = sorted(top_valid_cross_spearman_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_cross}/top_valid_cross_spearman_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Spearman Correlation_A", "Spearman Correlation_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_cross_spearman_features)
    print(f"Cross-task Spearman Correlation completed in {time.time() - t} seconds")
    print(f"Processing Within-task Spearman Correlation")
    t = time.time()
    spearman_correlation_within_A = tdf_A.corrwith(score_A, method="spearman")
    spearman_correlation_within_B = tdf_B.corrwith(score_B, method="spearman")
    filter_spearman_within_A_indices = set([i for i in range(len(spearman_correlation_within_A)) if abs(spearman_correlation_within_A[i]) >= 0.2])
    filter_spearman_within_B_indices = set([i for i in range(len(spearman_correlation_within_B)) if abs(spearman_correlation_within_B[i]) >= 0.2])
    filter_spearman_within_indices = sorted(list(filter_spearman_within_A_indices.intersection(filter_spearman_within_B_indices)))
    top_valid_within_spearman_features = []
    for i in filter_spearman_within_indices:
        tpl = [unique_features[i], spearman_correlation_within_A[i], spearman_correlation_within_B[i], metric_harmonic_mean(spearman_correlation_within_A[i], spearman_correlation_within_B[i])]
        top_valid_within_spearman_features.append(tpl)
    sorted_top_valid_within_spearman_features = sorted(top_valid_within_spearman_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_within}/top_valid_within_spearman_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Spearman Correlation_A", "Spearman Correlation_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_within_spearman_features)
    print(f"Within-task Spearman Correlation completed in {time.time() - t} seconds")
    ###########################################################################################
    # Pearson Correlation
    ###########################################################################################
    print(f"Processing Pearson Correlation for threshold {threshold}")
    print(f"Processing Cross-task Pearson Correlation")
    t = time.time()
    pearson_correlation_cross_A = tdf_A.corrwith(score_B, method="pearson")
    pearson_correlation_cross_B = tdf_B.corrwith(score_A, method="pearson")
    filter_pearson_cross_A_indices = set([i for i in range(len(pearson_correlation_cross_A)) if abs(pearson_correlation_cross_A[i]) >= 0.2])
    filter_pearson_cross_B_indices = set([i for i in range(len(pearson_correlation_cross_B)) if abs(pearson_correlation_cross_B[i]) >= 0.2])
    filter_pearson_cross_indices = sorted(list(filter_pearson_cross_A_indices.intersection(filter_pearson_cross_B_indices)))
    top_valid_cross_pearson_features = []
    for i in filter_pearson_cross_indices:
        tpl = [unique_features[i], pearson_correlation_cross_A[i], pearson_correlation_cross_B[i], metric_harmonic_mean(pearson_correlation_cross_A[i], pearson_correlation_cross_B[i])]
        top_valid_cross_pearson_features.append(tpl)
    sorted_top_valid_cross_pearson_features = sorted(top_valid_cross_pearson_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_cross}/top_valid_cross_pearson_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Pearson Correlation_A", "Pearson Correlation_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_cross_pearson_features)
    print(f"Cross-task Pearson Correlation completed in {time.time() - t} seconds")
    print(f"Processing Within-task Pearson Correlation")
    t = time.time()
    pearson_correlation_within_A = tdf_A.corrwith(score_A, method="pearson")
    pearson_correlation_within_B = tdf_B.corrwith(score_B, method="pearson")
    filter_pearson_within_A_indices = set([i for i in range(len(pearson_correlation_within_A)) if abs(pearson_correlation_within_A[i]) >= 0.2])
    filter_pearson_within_B_indices = set([i for i in range(len(pearson_correlation_within_B)) if abs(pearson_correlation_within_B[i]) >= 0.2])
    filter_pearson_within_indices = sorted(list(filter_pearson_within_A_indices.intersection(filter_pearson_within_B_indices)))
    top_valid_within_pearson_features = []
    for i in filter_pearson_within_indices:
        tpl = [unique_features[i], pearson_correlation_within_A[i], pearson_correlation_within_B[i], metric_harmonic_mean(pearson_correlation_within_A[i], pearson_correlation_within_B[i])]
        top_valid_within_pearson_features.append(tpl)
    sorted_top_valid_within_pearson_features = sorted(top_valid_within_pearson_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_within}/top_valid_within_pearson_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Pearson Correlation_A", "Pearson Correlation_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_within_pearson_features)
    print(f"Within-task Pearson Correlation completed in {time.time() - t} seconds")
    ###########################################################################################
    # Mutual Information
    ###########################################################################################
    print(f"Processing Mutual Information for threshold {threshold}")
    print(f"Processing Cross-task Mutual Information")
    t = time.time()
    mutual_information_cross_A = mutual_info_classif(utdf_A, score_B, discrete_features='auto', n_neighbors=3, n_jobs=-1)
    mutual_information_cross_B = mutual_info_classif(utdf_B, score_A, discrete_features='auto', n_neighbors=3, n_jobs=-1)
    filter_mutual_information_cross_A_indices = set([i for i in range(len(mutual_information_cross_A)) if mutual_information_cross_A[i] >= 0.05])
    filter_mutual_information_cross_B_indices = set([i for i in range(len(mutual_information_cross_B)) if mutual_information_cross_B[i] >= 0.05])
    filter_mutual_information_cross_indices = sorted(list(filter_mutual_information_cross_A_indices.intersection(filter_mutual_information_cross_B_indices)))
    top_valid_cross_mutual_information_features = []
    for i in filter_mutual_information_cross_indices:
        tpl = [unique_features[i], mutual_information_cross_A[i], mutual_information_cross_B[i], safe_harmonic_mean(mutual_information_cross_A[i], mutual_information_cross_B[i])]
        top_valid_cross_mutual_information_features.append(tpl)
    sorted_top_valid_cross_mutual_information_features = sorted(top_valid_cross_mutual_information_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_cross}/top_valid_cross_mutual_information_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Mutual Information_A", "Mutual Information_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_cross_mutual_information_features)
    print(f"Cross-task Mutual Information completed in {time.time() - t} seconds")
    print(f"Processing Within-task Mutual Information")
    t = time.time()
    mutual_information_within_A = mutual_info_classif(utdf_A, score_A, discrete_features='auto', n_neighbors=3, n_jobs=-1)
    mutual_information_within_B = mutual_info_classif(utdf_B, score_B, discrete_features='auto', n_neighbors=3, n_jobs=-1)
    filter_mutual_information_within_A_indices = set([i for i in range(len(mutual_information_within_A)) if mutual_information_within_A[i] >= 0.05])
    filter_mutual_information_within_B_indices = set([i for i in range(len(mutual_information_within_B)) if mutual_information_within_B[i] >= 0.05])
    filter_mutual_information_within_indices = sorted(list(filter_mutual_information_within_A_indices.intersection(filter_mutual_information_within_B_indices)))
    top_valid_within_mutual_information_features = []
    for i in filter_mutual_information_within_indices:
        tpl = [unique_features[i], mutual_information_within_A[i], mutual_information_within_B[i], safe_harmonic_mean(mutual_information_within_A[i], mutual_information_within_B[i])]
        top_valid_within_mutual_information_features.append(tpl)
    sorted_top_valid_within_mutual_information_features = sorted(top_valid_within_mutual_information_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_within}/top_valid_within_mutual_information_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Mutual Information_A", "Mutual Information_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_within_mutual_information_features)
    print(f"Within-task Mutual Information completed in {time.time() - t} seconds")
    ###########################################################################################
    # Point-Biserial Correlation
    ###########################################################################################
    print(f"Processing Point-Biserial Correlation for threshold {threshold}")
    print(f"Processing Cross-task Point-Biserial Correlation")
    t = time.time()
    point_biserial_cross_A, point_biserial_p_values_cross_A = list(zip(*[pointbiserialr(tdf_A[col], score_B) for col in tdf_A.columns]))
    point_biserial_cross_B, point_biserial_p_values_cross_B = list(zip(*[pointbiserialr(tdf_B[col], score_A) for col in tdf_B.columns]))
    filter_point_biserial_p_values_cross_A_indices = set([i for i in range(len(point_biserial_p_values_cross_A)) if point_biserial_p_values_cross_A[i] <= 0.05])
    filter_point_biserial_p_values_cross_B_indices = set([i for i in range(len(point_biserial_p_values_cross_B)) if point_biserial_p_values_cross_B[i] <= 0.05])
    filter_point_biserial_p_values_cross_indices = sorted(list(filter_point_biserial_p_values_cross_A_indices.intersection(filter_point_biserial_p_values_cross_B_indices)))
    top_valid_point_biserial_cross_features = []
    for i in filter_point_biserial_p_values_cross_indices:
        tpl = [unique_features[i], point_biserial_cross_A[i], point_biserial_cross_B[i], metric_harmonic_mean(point_biserial_cross_A[i], point_biserial_cross_B[i])]
        top_valid_point_biserial_cross_features.append(tpl)
    sorted_top_valid_point_biserial_cross_features = sorted(top_valid_point_biserial_cross_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_cross}/top_valid_cross_point_biserial_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Point-Biserial Correlation_A", "Point-Biserial Correlation_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_point_biserial_cross_features)
    print(f"Cross-task Point-Biserial Correlation completed in {time.time() - t} seconds")
    print(f"Processing Within-task Point-Biserial Correlation")
    t = time.time()
    point_biserial_within_A, point_biserial_p_values_within_A = list(zip(*[pointbiserialr(tdf_A[col], score_A) for col in tdf_A.columns]))
    point_biserial_within_B, point_biserial_p_values_within_B = list(zip(*[pointbiserialr(tdf_B[col], score_B) for col in tdf_B.columns]))
    filter_point_biserial_p_values_within_A_indices = set([i for i in range(len(point_biserial_p_values_within_A)) if point_biserial_p_values_within_A[i] <= 0.05])
    filter_point_biserial_p_values_within_B_indices = set([i for i in range(len(point_biserial_p_values_within_B)) if point_biserial_p_values_within_B[i] <= 0.05])
    filter_point_biserial_p_values_within_indices = sorted(list(filter_point_biserial_p_values_within_A_indices.intersection(filter_point_biserial_p_values_within_B_indices)))
    top_valid_point_biserial_within_features = []
    for i in filter_point_biserial_p_values_within_indices:
        tpl = [unique_features[i], point_biserial_within_A[i], point_biserial_within_B[i], metric_harmonic_mean(point_biserial_within_A[i], point_biserial_within_B[i])]
        top_valid_point_biserial_within_features.append(tpl)
    sorted_top_valid_point_biserial_within_features = sorted(top_valid_point_biserial_within_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_within}/top_valid_within_point_biserial_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Point-Biserial Correlation_A", "Point-Biserial Correlation_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_point_biserial_within_features)
    print(f"Within-task Point-Biserial Correlation completed in {time.time() - t} seconds")
    ###########################################################################################
    # Kolmogorov-Smirnov Test
    ###########################################################################################
    print(f"Processing Kolmogorov-Smirnov Test for threshold {threshold}")
    print(f"Processing Cross-task Kolmogorov-Smirnov Test")
    t = time.time()
    ks_stat_cross_A, ks_stat_p_values_cross_A = list(zip(*[ks_2samp(tdf_A[col][score_B == 0], tdf_A[col][score_B == 1]) for col in tdf_A.columns]))
    ks_stat_cross_B, ks_stat_p_values_cross_B = list(zip(*[ks_2samp(tdf_B[col][score_A == 0], tdf_B[col][score_A == 1]) for col in tdf_B.columns]))
    filter_ks_stat_p_values_cross_A_indices = set([i for i in range(len(ks_stat_p_values_cross_A)) if ks_stat_p_values_cross_A[i] <= 0.05])
    filter_ks_stat_p_values_cross_B_indices = set([i for i in range(len(ks_stat_p_values_cross_B)) if ks_stat_p_values_cross_B[i] <= 0.05])
    filter_ks_stat_p_values_cross_indices = sorted(list(filter_ks_stat_p_values_cross_A_indices.intersection(filter_ks_stat_p_values_cross_B_indices)))
    top_valid_ks_stat_cross_features = []
    for i in filter_ks_stat_p_values_cross_indices:
        tpl = [unique_features[i], ks_stat_cross_A[i], ks_stat_cross_B[i], safe_harmonic_mean(ks_stat_cross_A[i], ks_stat_cross_B[i])]
        top_valid_ks_stat_cross_features.append(tpl)
    sorted_top_valid_ks_stat_cross_features = sorted(top_valid_ks_stat_cross_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_cross}/top_valid_cross_ks_stat_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "KS-Statistic_A", "KS-Statistic_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_ks_stat_cross_features)
    print(f"Cross-task Kolmogorov-Smirnov Test completed in {time.time() - t} seconds")
    print(f"Processing Within-task Kolmogorov-Smirnov Test")
    t = time.time()
    ks_stat_within_A, ks_stat_p_values_within_A = list(zip(*[ks_2samp(tdf_A[col][score_A == 0], tdf_A[col][score_A == 1]) for col in tdf_A.columns]))
    ks_stat_within_B, ks_stat_p_values_within_B = list(zip(*[ks_2samp(tdf_B[col][score_B == 0], tdf_B[col][score_B == 1]) for col in tdf_B.columns]))
    filter_ks_stat_p_values_within_A_indices = set([i for i in range(len(ks_stat_p_values_within_A)) if ks_stat_p_values_within_A[i] <= 0.05])
    filter_ks_stat_p_values_within_B_indices = set([i for i in range(len(ks_stat_p_values_within_B)) if ks_stat_p_values_within_B[i] <= 0.05])
    filter_ks_stat_p_values_within_indices = sorted(list(filter_ks_stat_p_values_within_A_indices.intersection(filter_ks_stat_p_values_within_B_indices)))
    top_valid_ks_stat_within_features = []
    for i in filter_ks_stat_p_values_within_indices:
        tpl = [unique_features[i], ks_stat_within_A[i], ks_stat_within_B[i], safe_harmonic_mean(ks_stat_within_A[i], ks_stat_within_B[i])]
        top_valid_ks_stat_within_features.append(tpl)
    sorted_top_valid_ks_stat_within_features = sorted(top_valid_ks_stat_within_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_within}/top_valid_within_ks_stat_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "KS-Statistic_A", "KS-Statistic_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_ks_stat_within_features)
    print(f"Within-task Kolmogorov-Smirnov Test completed in {time.time() - t} seconds")
    ###########################################################################################
    # Scalable Metadata
    ###########################################################################################
    # print(f"Processing Scalable Metadata for threshold {threshold}")
    # print(f"Processing Cross-task Scalable Metadata")
    # t = time.time()
    # f_statistic_metadata_A, fstat_p_values_metadata_A = f_classif(smdf_A, score_B)
    # f_statistic_metadata_B, fstat_p_values_metadata_B = f_classif(smdf_B, score_A)
    # filter_fstat_p_values_metadata_A_indices = set([i for i in range(len(fstat_p_values_metadata_A)) if fstat_p_values_metadata_A[i] <= 0.05])
    # filter_fstat_p_values_metadata_B_indices = set([i for i in range(len(fstat_p_values_metadata_B)) if fstat_p_values_metadata_B[i] <= 0.05])

    # pearson_metadata_A = smdf_A.corrwith(score_B, method="pearson")
    # pearson_metadata_B = smdf_B.corrwith(score_A, method="pearson")
    # filter_pearson_metadata_A_indices = set([i for i in range(len(pearson_metadata_A)) if abs(pearson_metadata_A[i]) >= 0.2])
    # filter_pearson_metadata_B_indices = set([i for i in range(len(pearson_metadata_B)) if abs(pearson_metadata_B[i]) >= 0.2])

    # spearman_metadata_A = smdf_A.corrwith(score_B, method="spearman")
    # spearman_metadata_B = smdf_B.corrwith(score_A, method="spearman")
    # filter_spearman_metadata_A_indices = set([i for i in range(len(spearman_metadata_A)) if abs(spearman_metadata_A[i]) >= 0.2])
    # filter_spearman_metadata_B_indices = set([i for i in range(len(spearman_metadata_B)) if abs(spearman_metadata_B[i]) >= 0.2])

    # mutual_information_metadata_A = mutual_info_classif(smdf_A, score_B, discrete_features='auto', n_neighbors=3, n_jobs=-1)
    # mutual_information_metadata_B = mutual_info_classif(smdf_B, score_A, discrete_features='auto', n_neighbors=3, n_jobs=-1)

    # point_biserial_metadata_A, point_biserial_p_values_metadata_A = list(zip(*[pointbiserialr(smdf_A[col], score_B) for col in smdf_A.columns]))
    # point_biserial_metadata_B, point_biserial_p_values_metadata_B = list(zip(*[pointbiserialr(smdf_B[col], score_A) for col in smdf_B.columns]))
    # filter_point_biserial_p_values_metadata_A_indices = set([i for i in range(len(point_biserial_p_values_metadata_A)) if point_biserial_p_values_metadata_A[i] <= 0.05])
    # filter_point_biserial_p_values_metadata_B_indices = set([i for i in range(len(point_biserial_p_values_metadata_B)) if point_biserial_p_values_metadata_B[i] <= 0.05])

    # ks_stat_metadata_A, ks_stat_p_values_metadata_A = list(zip(*[ks_2samp(smdf_A[col][score_B == 0], smdf_A[col][score_B == 1]) for col in smdf_A.columns]))
    # ks_stat_metadata_B, ks_stat_p_values_metadata_B = list(zip(*[ks_2samp(smdf_B[col][score_A == 0], smdf_B[col][score_A == 1]) for col in smdf_B.columns]))
    # filter_ks_stat_p_values_metadata_A_indices = set([i for i in range(len(ks_stat_p_values_metadata_A)) if ks_stat_p_values_metadata_A[i] <= 0.05])
    # filter_ks_stat_p_values_metadata_B_indices = set([i for i in range(len(ks_stat_p_values_metadata_B)) if ks_stat_p_values_metadata_B[i] <= 0.05])

    ###########################################################################################
    # Chi-Square Test (Metadata)
    ###########################################################################################
    print(f"Processing Chi-Square Test (Metadata) for threshold {threshold}")
    print(f"Processing Cross-task Chi-Square Test (Metadata)")
    t = time.time()
    chisq_cross_A, chisq_p_values_cross_A, chisq_dof_cross_A, chisq_expected_cross_A = list(zip(*[chi2_contingency(pd.crosstab(mdf_A[col], score_B)) for col in mdf_A.columns]))
    chisq_cross_B, chisq_p_values_cross_B, chisq_dof_cross_B, chisq_expected_cross_B = list(zip(*[chi2_contingency(pd.crosstab(mdf_B[col], score_A)) for col in mdf_B.columns]))
    filter_chisq_p_values_cross_A_indices = set([i for i in range(len(chisq_p_values_cross_A)) if chisq_p_values_cross_A[i] <= 0.05])
    filter_chisq_p_values_cross_B_indices = set([i for i in range(len(chisq_p_values_cross_B)) if chisq_p_values_cross_B[i] <= 0.05])
    filter_chisq_p_values_cross_indices = sorted(list(filter_chisq_p_values_cross_A_indices.intersection(filter_chisq_p_values_cross_B_indices)))
    top_valid_chisq_cross_features = []
    for i in filter_chisq_p_values_cross_indices:
        tpl = [metadata_all_unique_features[i], chisq_cross_A[i], chisq_cross_B[i], safe_harmonic_mean(chisq_cross_A[i], chisq_cross_B[i])]
        top_valid_chisq_cross_features.append(tpl)
    sorted_top_valid_chisq_cross_features = sorted(top_valid_chisq_cross_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_cross}/top_valid_cross_chisq_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Chi-Square_A", "Chi-Square_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_chisq_cross_features)
    print(f"Cross-task Chi-Square Test (Metadata) completed in {time.time() - t} seconds")
    print(f"Processing Within-task Chi-Square Test (Metadata)")
    t = time.time()
    chisq_within_A, chisq_p_values_within_A, chisq_dof_within_A, chisq_expected_within_A = list(zip(*[chi2_contingency(pd.crosstab(mdf_A[col], score_A)) for col in mdf_A.columns]))
    chisq_within_B, chisq_p_values_within_B, chisq_dof_within_B, chisq_expected_within_B = list(zip(*[chi2_contingency(pd.crosstab(mdf_B[col], score_B)) for col in mdf_B.columns]))
    filter_chisq_p_values_within_A_indices = set([i for i in range(len(chisq_p_values_within_A)) if chisq_p_values_within_A[i] <= 0.05])
    filter_chisq_p_values_within_B_indices = set([i for i in range(len(chisq_p_values_within_B)) if chisq_p_values_within_B[i] <= 0.05])
    filter_chisq_p_values_within_indices = sorted(list(filter_chisq_p_values_within_A_indices.intersection(filter_chisq_p_values_within_B_indices)))
    top_valid_chisq_within_features = []
    for i in filter_chisq_p_values_within_indices:
        tpl = [metadata_all_unique_features[i], chisq_within_A[i], chisq_within_B[i], safe_harmonic_mean(chisq_within_A[i], chisq_within_B[i])]
        top_valid_chisq_within_features.append(tpl)
    sorted_top_valid_chisq_within_features = sorted(top_valid_chisq_within_features, key=lambda x: x[3], reverse=True)

    with open(f"{pth_within}/top_valid_within_chisq_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Chi-Square_A", "Chi-Square_B", "Harmonic Mean"])
        writer.writerows(sorted_top_valid_chisq_within_features)
    print(f"Within-task Chi-Square Test (Metadata) completed in {time.time() - t} seconds")
