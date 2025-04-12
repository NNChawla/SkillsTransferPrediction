from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from BorutaShap import BorutaShap
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import hmean
from sklearn.base import clone
import pandas as pd
import numpy as np

def harmonic_mean(x, y):
    if x <= 0 or y <= 0:
        return min(x, y)
    return 2 / ((1 / x) + (1 / y))

def disjoint_scorer(y_true, y_pred):
    y_pred_A, y_pred_B, y_true_A, y_true_B = y_pred
    mcc_A = matthews_corrcoef(y_true_A, y_pred_A)
    mcc_B = matthews_corrcoef(y_true_B, y_pred_B)
    return mcc_A, mcc_B

def joint_scorer(y_true, y_pred):
    mcc_A, mcc_B = disjoint_scorer(y_true, y_pred)
    return harmonic_mean(mcc_A, mcc_B)

def evaluate_predictions(y_true, y_pred, score_fn):
    y_pred_A, y_pred_B, y_true_A, y_true_B = y_pred
    accuracy_A, accuracy_B = accuracy_score(y_true_A, y_pred_A), accuracy_score(y_true_B, y_pred_B)
    accuracy = harmonic_mean(accuracy_A, accuracy_B)
    balanced_accuracy_A, balanced_accuracy_B = balanced_accuracy_score(y_true_A, y_pred_A), balanced_accuracy_score(y_true_B, y_pred_B)
    balanced_accuracy = harmonic_mean(balanced_accuracy_A, balanced_accuracy_B)
    f1_A, f1_B = f1_score(y_true_A, y_pred_A, average='weighted', zero_division=0), f1_score(y_true_B, y_pred_B, average='weighted', zero_division=0)
    f1 = harmonic_mean(f1_A, f1_B)
    precision_A, precision_B = precision_score(y_true_A, y_pred_A, average='weighted', zero_division=0), precision_score(y_true_B, y_pred_B, average='weighted', zero_division=0)
    precision = harmonic_mean(precision_A, precision_B)
    recall_A, recall_B = recall_score(y_true_A, y_pred_A, average='weighted', zero_division=0), recall_score(y_true_B, y_pred_B, average='weighted', zero_division=0)
    recall = harmonic_mean(recall_A, recall_B)
    sensitivity_A, sensitivity_B = recall_score(y_true_A, y_pred_A, average='binary', pos_label=1, zero_division=0), recall_score(y_true_B, y_pred_B, average='binary', pos_label=1, zero_division=0)
    sensitivity = harmonic_mean(sensitivity_A, sensitivity_B)
    specificity_A, specificity_B = recall_score(y_true_A, y_pred_A, average='binary', pos_label=0, zero_division=0), recall_score(y_true_B, y_pred_B, average='binary', pos_label=0, zero_division=0)
    specificity = harmonic_mean(specificity_A, specificity_B)
    joint_mcc = score_fn(y_true, y_pred)
    cm_A = confusion_matrix(y_true_A, y_pred_A, labels=[0, 1])
    cm_B = confusion_matrix(y_true_B, y_pred_B, labels=[0, 1])
    normalized_matrices = []
    for cm in [cm_A, cm_B]:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        normalized_matrices.append(cm_norm)
    cm = hmean(normalized_matrices, axis=0)

    result = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'joint_mcc': joint_mcc,
        'confusion_matrix': cm,
        'y_true_A': y_true_A,
        'y_pred_A': y_pred_A,
        'y_true_B': y_true_B,
        'y_pred_B': y_pred_B
    }
    return result

def make_joint_dummy_data(X, y):
    X_dummy = pd.DataFrame(np.arange(X.shape[0] * X.shape[1]).reshape(X.shape[1], X.shape[0]).T, columns = X.columns)
    y_dummy = sorted(y)
    return X_dummy, y_dummy

class JointDummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.k_feature_idx_ = None
        return self
    
    def set_k_feature_idx(self, k_feature_idx):
        self.k_feature_idx_ = k_feature_idx

    def transform(self, X):
        selected_Dummy, _ = make_joint_dummy_data(X, np.arange(X.shape[0]))
        if self.k_feature_idx_ is not None:
            selected_Dummy = selected_Dummy.iloc[:, list(self.k_feature_idx_)]
        selected_Dummy = selected_Dummy.to_numpy(copy=True)
        return selected_Dummy
        
class JointEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, data_A, data_B, score_A, score_B):
        self.estimator = estimator
        self.estimator_B = clone(estimator)
        self.data_A = data_A
        self.data_B = data_B
        self.score_A = score_A
        self.score_B = score_B
        self.nrows = self.data_A.shape[0]

    def set_data(self, data_A, data_B, score_A, score_B):
        self.data_A = data_A
        self.data_B = data_B
        self.score_A = score_A
        self.score_B = score_B
        self.nrows = self.data_A.shape[0]

    def retrieve_data(self, X):
        selected_cols = [int(np.floor(i / self.nrows)) for i in X[0]]
        selected_rows = [i % self.nrows for i in X[:,0]]
        return (self.data_A[selected_rows, :][:, selected_cols],
        self.data_B[selected_rows, :][:, selected_cols],
        self.score_A[selected_rows], self.score_B[selected_rows])

    def fit(self, X, y):
        # print(f"Fitting {X} and {y}")
        X_A, X_B, y_A, y_B = self.retrieve_data(X)
        # print(f"Got data: {X_A}, {X_B}, {y_A}, {y_B}")
        self.estimator.fit(X_A, y_A)
        self.estimator_B.fit(X_B, y_B)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        # print(f"Predicting {X}")
        X_A, X_B, y_true_A, y_true_B = self.retrieve_data(X)
        # print(f"Got data: {X_A}, {X_B}, {y_true_A}, {y_true_B}")
        y_pred_A = self.estimator.predict(X_A)
        y_pred_B = self.estimator_B.predict(X_B)
        # print(f"Predicted: {y_pred_A}, {y_pred_B}")
        return (y_pred_A, y_pred_B, y_true_A, y_true_B)

class JointSFSSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, k_features, forward=True, floating=True, scoring=None, cv=None, fixed_features=None, feature_groups=None, n_jobs=1):
        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.scoring = scoring
        self.cv = cv
        self.fixed_features = fixed_features
        self.feature_groups = feature_groups
        self.n_jobs = n_jobs
        self.k_feature_idx_ = None

    def fit(self, X, y):
        X_dummy, y_dummy = make_joint_dummy_data(X, y)
        self.sfs_ = SFS(estimator=self.estimator,
                        k_features=self.k_features,
                        forward=self.forward,
                        floating=self.floating,
                        scoring=self.scoring,
                        cv=self.cv,
                        n_jobs=self.n_jobs,
                        fixed_features=self.fixed_features,
                        feature_groups=self.feature_groups,
                        verbose=0)
        self.sfs_.fit(X_dummy, y_dummy)
        self.k_feature_idx_ = self.sfs_.k_feature_idx_
        return self

    def transform(self, X):
        selected_Dummy, _ = make_joint_dummy_data(X, np.arange(X.shape[0]))
        selected_Dummy = selected_Dummy.iloc[:, list(self.k_feature_idx_)]
        selected_Dummy = selected_Dummy.to_numpy(copy=True)
        return selected_Dummy

# TODO: Add RFE wrapper
# TODO: Add BorutaShap wrapper