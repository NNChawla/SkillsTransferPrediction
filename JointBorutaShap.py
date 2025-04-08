import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy.stats import binomtest
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from scipy.stats import binomtest, ks_2samp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import pandas as pd
import numpy as np
from numpy.random import choice
import seaborn as sns
import shap
import os
import re

def harmonic_mean_vector(a, b):
    # Element-wise harmonic mean for two arrays a and b
    # Avoid division by zero by returning the minimum value if any element is <= 0
    result = np.zeros_like(a)
    for i, (x, y) in enumerate(zip(a, b)):
        result[i] = 2 / ((1 / abs(x)) + (1 / abs(y)))
    return result

class JointBorutaShap:
    """
    A joint extension of BorutaShap for feature selection across two datasets.
    
    Instead of relying on a dummy matrix, this class processes two datasets in parallel.
    For each iteration, it creates shadow features for both datasets, trains separate models,
    computes feature importances, and then combines the importance values (e.g. via harmonic mean)
    to decide which features to accept or reject jointly.
    """
    
    def __init__(self, model=None, cv=None, importance_measure='shap', classification=True,
                 percentile=100, pvalue=0.05, n_trials=20, random_state=0):
        self.model = model
        self.cv = cv
        self.importance_measure = importance_measure
        self.classification = classification
        self.percentile = percentile
        self.pvalue = pvalue
        self.n_trials = n_trials
        self.random_state = random_state
        self.check_model()

    def check_model(self):
        check_fit = hasattr(self.model, 'fit')
        check_predict_proba = hasattr(self.model, 'predict')
        try:
            check_feature_importance = hasattr(self.model, 'feature_importances_')
        except:
            check_feature_importance = True

        # if self.model is None:
            # raise ValueError('Model cannot be None')
        # elif check_fit is False and check_predict_proba is False:
            # raise AttributeError('Model must contain both the fit() and predict() methods')
        # elif check_feature_importance is False and self.importance_measure == 'gini':
            # raise AttributeError('Model must contain the feature_importances_ method to use Gini try Shap instead')       

    def check_data(self, X, y):
        if isinstance(X, pd.DataFrame) is False:
            raise AttributeError('Data must be a pandas Dataframe')
        X_missing = X.isnull().any().any()
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            Y_missing = y.isnull().any().any()
        elif isinstance(y, np.ndarray):
            Y_missing = np.isnan(y).any()
        else:
            raise AttributeError('Y must be a pandas Dataframe, Series, or a numpy array')

        models_to_check = ('xgb', 'catboost', 'lgbm', 'lightgbm')
        model_name = str(type(self.model)).lower()
        if X_missing or Y_missing:
            if any([x in model_name for x in models_to_check]):
                print('Warning there are missing values in your data !')
            else:
                raise ValueError('There are missing values in your Data')
            
    def split_train_model(self, X_boruta_A, X_boruta_B):
        if self.cv is not None:
            train_idx, test_idx = list(self.cv.split(X_boruta_A, self.y_A))[0]
            X_A_train, X_A_test = X_boruta_A.iloc[train_idx], X_boruta_A.iloc[test_idx]
            y_A_train, y_A_test = self.y_A.iloc[train_idx], self.y_A.iloc[test_idx]
            X_B_train, X_B_test = X_boruta_B.iloc[train_idx], X_boruta_B.iloc[test_idx]
            y_B_train, y_B_test = self.y_B.iloc[train_idx], self.y_B.iloc[test_idx]
            sample_weight_A_train, sample_weight_A_test = self.sample_weight_A[train_idx], self.sample_weight_A[test_idx]
            sample_weight_B_train, sample_weight_B_test = self.sample_weight_B[train_idx], self.sample_weight_B[test_idx]
            self._train_models(X_A_train, y_A_train, sample_weight_A_train, X_B_train, y_B_train, sample_weight_B_train)
        else:
            self._train_models(X_boruta_A, self.y_A, self.sample_weight_A, X_boruta_B, self.y_B, self.sample_weight_B)

    def _train_models(self, X_boruta_A, y_A, sample_weight_A, X_boruta_B, y_B, sample_weight_B):       
        self.model_A.fit(X_boruta_A, y_A, sample_weight=sample_weight_A)
        self.model_B.fit(X_boruta_B, y_B, sample_weight=sample_weight_B)
        
    def fit(self, X_A, y_A, X_B, y_B, sample_weight_A, sample_weight_B, use_sampling=False):
        np.random.seed(self.random_state)
        
        self.X_A = X_A.copy()
        self.X_B = X_B.copy()
        self.y_A = y_A.copy()
        self.y_B = y_B.copy()
        self.sample_weight_A = sample_weight_A
        self.sample_weight_B = sample_weight_B

        self.check_data(self.X_A, self.y_A)
        self.check_data(self.X_B, self.y_B)
        self.use_sampling = use_sampling

        self.n_features = self.X_A.shape[1]
        self.accepted_columns = []
        self.rejected_columns = []
        self.features_to_remove = []

        self.hits = np.zeros(self.n_features, dtype=int)
        self.unique_features = np.array([i.replace("_A", "") for i in self.X_A.columns])
        self.unique_order = dict(zip(self.unique_features.tolist(), np.arange(self.X_A.shape[1])))
        self.create_importance_history()
        
        if self.use_sampling:
            self.preds_A = self.isolation_forest(self.X_A, self.sample_weight_A)
            self.preds_B = self.isolation_forest(self.X_B, self.sample_weight_B)
        
        # Clone the base model for each dataset
        self.model_A = clone(self.model) if self.model is not None else None
        self.model_B = clone(self.model) if self.model is not None else None
        
        # Main iterative loop
        for trial in tqdm(range(1, self.n_trials + 1)):
            self._remove_rejected_features()
            self.columns = np.array([i.replace("_A", "") for i in self.X_A.columns])
            X_boruta_A, X_shadow_A = self._create_shadow_features(self.X_A)
            X_boruta_B, X_shadow_B = self._create_shadow_features(self.X_B)

            if self.X_A.shape[1] == 0 or self.X_B.shape[1] == 0:
                break
            
            self.split_train_model(X_boruta_A, X_boruta_B)
            imp_A, imp_B, shadow_imp_A, shadow_imp_B = self._compute_feature_importance(X_boruta_A, X_boruta_B, X_shadow_A, X_shadow_B)
            joint_imp = harmonic_mean_vector(imp_A, imp_B)
            joint_shadow = harmonic_mean_vector(shadow_imp_A, shadow_imp_B)

            # print("Data")
            # print(shadow_imp_A)
            # print(shadow_imp_B)
            # print(joint_imp)
            # print(joint_shadow)
            # print(self.columns)
            # print(self.unique_order)
            # print(self.n_features)

            self.update_importance_history(joint_shadow, joint_imp)

            hits = self.calculate_hits(joint_shadow, joint_imp)
            self.hits += hits
            self.history_hits = np.vstack((self.history_hits, self.hits))
            self._test_features(trial)
            
        self.store_feature_importance()
        self._calculate_rejected_accepted_tentative(verbose=True)

    def transform(self, X_A, X_B, include_tentative=False):
        if include_tentative:
            X_A_trans = X_A[[f"{i}_A" for i in self.accepted] + [f"{i}_A" for i in self.tentative]].copy() if self.accepted else pd.DataFrame(index=X_A.index)
            X_B_trans = X_B[[f"{i}_B" for i in self.accepted] + [f"{i}_B" for i in self.tentative]].copy() if self.accepted else pd.DataFrame(index=X_B.index)
        else:
            X_A_trans = X_A[[f"{i}_A" for i in self.accepted]].copy() if self.accepted else pd.DataFrame(index=X_A.index)
            X_B_trans = X_B[[f"{i}_B" for i in self.accepted]].copy() if self.accepted else pd.DataFrame(index=X_B.index)
        return X_A_trans, X_B_trans
    
    def _calculate_rejected_accepted_tentative(self, verbose):
        self.rejected  = list(set(self.flatten_list(self.rejected_columns))-set(self.flatten_list(self.accepted_columns)))
        self.accepted  = list(set(self.flatten_list(self.accepted_columns)))
        self.tentative = list(set(self.unique_features) - set(self.rejected + self.accepted))
        if verbose:
            print(str(len(self.accepted))  + ' attributes confirmed important: ' + str(self.accepted))
            print(str(len(self.rejected))  + ' attributes confirmed unimportant: ' + str(self.rejected))
            print(str(len(self.tentative)) + ' tentative attributes remains: ' + str(self.tentative))

    def _remove_rejected_features(self):
        """
        Remove features that have already been rejected from the datasets.
        """
        if len(self.features_to_remove) > 0:
            self.X_A = self.X_A.drop(columns=[f"{i}_A" for i in self.features_to_remove], errors='ignore')
            self.X_B = self.X_B.drop(columns=[f"{i}_B" for i in self.features_to_remove], errors='ignore')            

    def _create_shadow_features(self, X):
        """
        Create shadow features by shuffling each column in X.
        
        Returns
        -------
        X_boruta : pandas.DataFrame
            DataFrame with original and shadow features concatenated.
        """
        X_shadow = X.apply(np.random.permutation)
        X_shadow.columns = ['shadow_' + col for col in X.columns]
        X_boruta = pd.concat([X, X_shadow], axis=1)
        return X_boruta, X_shadow
    
    @staticmethod
    def flatten_list(array):
        return [item for sublist in array for item in sublist]
    
    def calculate_hits(self, shadow_imp, x_imp):
        shadow_threshold = np.percentile(shadow_imp, self.percentile)
        padded_hits = np.zeros(self.n_features, dtype=int)
        hits = x_imp > shadow_threshold
        for (index, col) in enumerate(self.columns):
            map_index = self.unique_order[col]
            padded_hits[map_index] += hits[index]
        
        return padded_hits

    @staticmethod
    def calculate_Zscore(array):
        mean_value = np.mean(array)
        std_value  = np.std(array)
        return [(element - mean_value)/std_value for element in array]

    def _compute_feature_importance(self, X_boruta_A, X_boruta_B, X_shadow_A, X_shadow_B, normalize=True):
        if self.importance_measure == 'shap':
            self.explain(X_boruta_A, X_boruta_B)
            vals_A = self.shap_values_A
            vals_B = self.shap_values_B

            if normalize:
                vals_A = self.calculate_Zscore(vals_A)
                vals_B = self.calculate_Zscore(vals_B)

            X_feature_import_A = vals_A[:len(self.X_A.columns)]
            Shadow_feature_import_A = vals_A[len(self.X_A.columns):]

            X_feature_import_B = vals_B[:len(self.X_B.columns)]
            Shadow_feature_import_B = vals_B[len(self.X_B.columns):]

        elif self.importance_measure == 'perm':
            perm_importances_A =  permutation_importance(self.model_A, X_boruta_A, self.y_A, scoring=make_scorer(matthews_corrcoef))
            perm_importances_B =  permutation_importance(self.model_B, X_boruta_B, self.y_B, scoring=make_scorer(matthews_corrcoef))
            perm_importances_A = perm_importances_A.importances_mean
            perm_importances_B = perm_importances_B.importances_mean
        
            if normalize:
                perm_importances_A = self.calculate_Zscore(perm_importances_A)
                perm_importances_B = self.calculate_Zscore(perm_importances_B)

            X_feature_import_A = perm_importances_A[:len(self.X_A.columns)]
            Shadow_feature_import_A = perm_importances_A[len(self.X_A.columns):]
            X_feature_import_B = perm_importances_B[:len(self.X_B.columns)]
            Shadow_feature_import_B = perm_importances_B[len(self.X_B.columns):]

        elif self.importance_measure == 'gini':

            feature_importances_A =  np.abs(self.model_A.feature_importances_)
            feature_importances_B =  np.abs(self.model_B.feature_importances_)

            if normalize:
                feature_importances_A = self.calculate_Zscore(feature_importances_A)
                feature_importances_B = self.calculate_Zscore(feature_importances_B)

            X_feature_import_A = feature_importances_A[:len(self.X_A.columns)]
            Shadow_feature_import_A = feature_importances_A[len(self.X_A.columns):]

            X_feature_import_B = feature_importances_B[:len(self.X_B.columns)]
            Shadow_feature_import_B = feature_importances_B[len(self.X_B.columns):]

        else:
            raise ValueError('No Importance_measure was specified select one of (shap, perm, gini)')

        return X_feature_import_A, X_feature_import_B, Shadow_feature_import_A, Shadow_feature_import_B
    
    @staticmethod
    def binomial_H0_test(array, n, p, alternative):
        """
        Perform a test that the probability of success is p.
        This is an exact, two-sided test of the null hypothesis
        that the probability of success in a Bernoulli experiment is p
        """
        return [binomtest(x, n=n, p=p, alternative=alternative).pvalue for x in array]


    @staticmethod
    def symetric_difference_between_two_arrays(array_one, array_two):
        set_one = set(array_one)
        set_two = set(array_two)
        return np.array(list(set_one.symmetric_difference(set_two)))


    @staticmethod
    def find_index_of_true_in_array(array):
        length = len(array)
        return list(filter(lambda x: array[x], range(length)))

    @staticmethod
    def bonferoni_corrections(pvals, alpha=0.05, n_tests=None):
        """
        used to counteract the problem of multiple comparisons.
        """
        pvals = np.array(pvals)

        if n_tests is None:
            n_tests = len(pvals)
        else:
            pass

        alphacBon = alpha / float(n_tests)
        reject = pvals <= alphacBon
        pvals_corrected = pvals * float(n_tests)
        return reject, pvals_corrected

    def _test_features(self, iteration):

        acceptance_p_values = self.binomial_H0_test(self.hits,
                                                    n=iteration,
                                                    p=0.5,
                                                    alternative='greater')

        regect_p_values = self.binomial_H0_test(self.hits,
                                                n=iteration,
                                                p=0.5,
                                                alternative='less')

        # [1] as function returns a tuple
        modified_acceptance_p_values = self.bonferoni_corrections(acceptance_p_values,
                                                                  alpha=0.05,
                                                                  n_tests=len(self.columns))[1]

        modified_regect_p_values = self.bonferoni_corrections(regect_p_values,
                                                              alpha=0.05,
                                                              n_tests=len(self.columns))[1]

        # Take the inverse as we want true to keep featrues
        rejected_columns = np.array(modified_regect_p_values) < self.pvalue
        accepted_columns = np.array(modified_acceptance_p_values) < self.pvalue

        rejected_indices = self.find_index_of_true_in_array(rejected_columns)
        accepted_indices = self.find_index_of_true_in_array(accepted_columns)

        rejected_features = self.unique_features[rejected_indices]
        accepted_features = self.unique_features[accepted_indices]
        self.features_to_remove = rejected_features

        self.rejected_columns.append(rejected_features)
        self.accepted_columns.append(accepted_features)

    def create_importance_history(self):
        self.history_shadow = np.zeros(self.n_features)
        self.history_x = np.zeros(self.n_features)
        self.history_hits = np.zeros(self.n_features)

    def update_importance_history(self, shadow_imp, x_imp):
        padded_history_shadow  = np.full((self.n_features), np.NaN)
        padded_history_x = np.full((self.n_features), np.NaN)
        for (index, col) in enumerate(self.columns):
            map_index = self.unique_order[col]
            padded_history_shadow[map_index] = shadow_imp[index]
            padded_history_x[map_index] = x_imp[index]

        self.history_shadow = np.vstack((self.history_shadow, padded_history_shadow))
        self.history_x = np.vstack((self.history_x, padded_history_x))

    def store_feature_importance(self):
        self.history_x = pd.DataFrame(data=self.history_x, columns=self.unique_features)
        self.history_x['Max_Shadow']    =  [max(i) for i in self.history_shadow]
        self.history_x['Min_Shadow']    =  [min(i) for i in self.history_shadow]
        self.history_x['Mean_Shadow']   =  [np.nanmean(i) for i in self.history_shadow]
        self.history_x['Median_Shadow'] =  [np.nanmedian(i) for i in self.history_shadow]

    @staticmethod
    def isolation_forest(X, sample_weight):
        '''
        fits isloation forest to the dataset and gives an anomally score to every sample
        '''
        clf = IsolationForest().fit(X, sample_weight = sample_weight)
        preds = clf.score_samples(X)
        return preds
    
    def get_5_percent_splits(self, length):
        '''
        splits dataframe into 5% intervals
        '''
        five_percent = round(5  / 100 * length)
        return np.arange(five_percent,length,five_percent)
    
    def find_sample(self, X, X_boruta, preds):
        loop = True
        iteration = 0
        size = self.get_5_percent_splits(X.shape[0])
        element = 1
        while loop:
            sample_indices = choice(np.arange(preds.size),  size=size[element], replace=False)
            sample = np.take(preds, sample_indices)
            if ks_2samp(preds, sample).pvalue > 0.95:
                break
            
            iteration+=1

            if iteration == 20:
                element  += 1
                iteration = 0


        return X_boruta.iloc[sample_indices]
    
    def explain(self, X_boruta_A, X_boruta_B):
        explainer_A = shap.TreeExplainer(self.model_A, feature_perturbation = "tree_path_dependent", approximate = True)
        explainer_B = shap.TreeExplainer(self.model_B, feature_perturbation = "tree_path_dependent", approximate = True)

        if self.use_sampling:
            if self.classification:
                # for some reason shap returns values wraped in a list of length 1
                self.shap_values_A = np.array(explainer_A.shap_values(self.find_sample(self.X_A, X_boruta_A, self.preds_A)))
                self.shap_values_B = np.array(explainer_B.shap_values(self.find_sample(self.X_B, X_boruta_B, self.preds_B)))
                if isinstance(self.shap_values_A, list):
                    class_inds = range(len(self.shap_values_A))
                    shap_imp = np.zeros(self.shap_values_A[0].shape[1])
                    for i, ind in enumerate(class_inds):
                        shap_imp += np.abs(self.shap_values_A[ind]).mean(0)
                    self.shap_values_A /= len(self.shap_values_A)
                elif len(self.shap_values_A.shape) == 3:
                    self.shap_values_A = np.abs(self.shap_values_A).sum(axis=0)
                    self.shap_values_A = self.shap_values_A.mean(0)
                else:
                    self.shap_values_A = np.abs(self.shap_values_A).mean(0)
                if isinstance(self.shap_values_B, list):
                    class_inds = range(len(self.shap_values_B))
                    shap_imp = np.zeros(self.shap_values_B[0].shape[1])
                    for i, ind in enumerate(class_inds):
                        shap_imp += np.abs(self.shap_values_B[ind]).mean(0)
                    self.shap_values_B /= len(self.shap_values_B)
                elif len(self.shap_values_B.shape) == 3:
                    self.shap_values_B = np.abs(self.shap_values_B).sum(axis=0)
                    self.shap_values_B = self.shap_values_B.mean(0)
                else:
                    self.shap_values_B = np.abs(self.shap_values_B).mean(0)
            else:
                self.shap_values_A = explainer_A.shap_values(self.find_sample(self.X_A, X_boruta_A, self.preds_A))
                self.shap_values_B = explainer_B.shap_values(self.find_sample(self.X_B, X_boruta_B, self.preds_B))
                self.shap_values_A = np.abs(self.shap_values_A).mean(0)
                self.shap_values_B = np.abs(self.shap_values_B).mean(0)

        else:
            if self.classification:
                # for some reason shap returns values wraped in a list of length 1
                self.shap_values_A = np.array(explainer_A.shap_values(X_boruta_A))
                self.shap_values_B = np.array(explainer_B.shap_values(X_boruta_B))
                if isinstance(self.shap_values_A, list):
                    class_inds = range(len(self.shap_values_A))
                    shap_imp = np.zeros(self.shap_values_A[0].shape[1])
                    for i, ind in enumerate(class_inds):
                        shap_imp += np.abs(self.shap_values_A[ind]).mean(0)
                    self.shap_values_A /= len(self.shap_values_A)
                elif len(self.shap_values_A.shape) == 3:
                    self.shap_values_A = np.abs(self.shap_values_A).sum(axis=0)
                    self.shap_values_A = self.shap_values_A.mean(0)
                else:
                    self.shap_values_A = np.abs(self.shap_values_A).mean(0)
                if isinstance(self.shap_values_B, list):
                    class_inds = range(len(self.shap_values_B))
                    shap_imp = np.zeros(self.shap_values_B[0].shape[1])
                    for i, ind in enumerate(class_inds):
                        shap_imp += np.abs(self.shap_values_B[ind]).mean(0)
                    self.shap_values_B /= len(self.shap_values_B)
                elif len(self.shap_values_B.shape) == 3:
                    self.shap_values_B = np.abs(self.shap_values_B).sum(axis=0)
                    self.shap_values_B = self.shap_values_B.mean(0)
                else:
                    self.shap_values_B = np.abs(self.shap_values_B).mean(0)
            else:
                self.shap_values_A = explainer_A.shap_values(X_boruta_A)
                self.shap_values_B = explainer_B.shap_values(X_boruta_B)
                self.shap_values_A = np.abs(self.shap_values_A).mean(0)
                self.shap_values_B = np.abs(self.shap_values_B).mean(0)

    @staticmethod
    def create_list(array, color):
        colors = [color for x in range(len(array))]
        return colors
    
    @staticmethod
    def filter_data(data, column, value):
        data = data.copy()
        return data.loc[(data[column] == value) | (data[column] == 'Shadow')]
    
    @staticmethod
    def check_if_which_features_is_correct(my_string):
        my_string = str(my_string).lower()
        if my_string in ['tentative','rejected','accepted','all']:
            pass
        else:
            raise ValueError(my_string + " is not a valid value did you mean to type 'all', 'tentative', 'accepted' or 'rejected' ?")

    def create_mapping_of_features_to_attribute(self, maps = []):

        rejected = list(self.rejected)
        tentative = list(self.tentative)
        accepted = list(self.accepted)
        shadow = ['Max_Shadow','Median_Shadow','Min_Shadow','Mean_Shadow']

        tentative_map = self.create_list(tentative, maps[0])
        rejected_map  = self.create_list(rejected, maps[1])
        accepted_map  = self.create_list(accepted, maps[2])
        shadow_map = self.create_list(shadow, maps[3])

        values = tentative_map + rejected_map + accepted_map + shadow_map
        keys = tentative + rejected + accepted + shadow

        return dict(zip(keys, values))
    
    def box_plot(self, data, X_rotation, X_size, y_scale, figsize):

        if y_scale=='log':
            minimum = data['value'].min()
            if minimum <= 0:
                data['value'] += abs(minimum) + 0.01

        order = data.groupby(by=["Methods"])["value"].mean().sort_values(ascending=False).index
        my_palette = self.create_mapping_of_features_to_attribute(maps= ['yellow','red','green','blue'])

        # Use a color palette
        plt.figure(figsize=figsize)
        ax = sns.boxplot(x=data["Methods"], y=data["value"],
                        order=order, palette=my_palette)

        if y_scale == 'log':ax.set(yscale="log")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=X_rotation, size=X_size)
        ax.set_title('Feature Importance')
        ax.set_ylabel('Z-Score')
        ax.set_xlabel('Features')
    
    def plot(self, X_rotation=90, X_size=8, figsize=(12,8),
            y_scale='log', which_features='all', display=True):

        """
        creates a boxplot of the feature importances

        Parameters
        ----------
        X_rotation: int
            Controls the orientation angle of the tick labels on the X-axis

        X_size: int
            Controls the font size of the tick labels

        y_scale: string
            Log transform of the y axis scale as hard to see the plot as it is normally dominated by two or three
            features.

        which_features: string
            Despite efforts if the number of columns is large the plot becomes cluttered so this parameter allows you to
            select subsets of the features like the accepted, rejected or tentative features default is all.

        Display: Boolean
        controls if the output is displayed or not, set to false when running test scripts

        """
        # data from wide to long
        data = self.history_x.iloc[1:]
        data['index'] = data.index
        data = pd.melt(data, id_vars='index', var_name='Methods')

        decision_mapper = self.create_mapping_of_features_to_attribute(maps=['Tentative','Rejected','Accepted', 'Shadow'])
        data['Decision'] = data['Methods'].map(decision_mapper)
        data.drop(['index'], axis=1, inplace=True)

        options = { 'accepted' : self.filter_data(data,'Decision', 'Accepted'),
                    'tentative': self.filter_data(data,'Decision', 'Tentative'),
                    'rejected' : self.filter_data(data,'Decision', 'Rejected'),
                    'all' : data
                    }

        self.check_if_which_features_is_correct(which_features)
        data = options[which_features.lower()]

        self.box_plot(data=data,
                      X_rotation=X_rotation,
                      X_size=X_size,
                      y_scale=y_scale,
                      figsize=figsize)
        if display:
            plt.show()
        else:
            plt.close()