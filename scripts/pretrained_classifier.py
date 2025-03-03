import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
# This wrapper will let us plug in a pre-trained classifier.
class PreTrainedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, name=None):
        self.clf = clf
        self.name = name if name else f"model_{id(clf)}"  # Generate unique name if none provided

    def fit(self, X, y):
        # Do nothing: assume clf is already trained.
        return self

    def predict(self, X):
        # No need to check if fitted since we're wrapping a pre-trained model
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def get_params(self, deep=True):
        return {"clf": self.clf, "name": self.name}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __iter__(self):
        """Make the classifier iterable to work with VotingClassifier"""
        yield self.name, self

    def _get_support_mask(self):
        # Required for feature selection compatibility
        if hasattr(self.clf, '_get_support_mask'):
            return self.clf._get_support_mask()
        return None