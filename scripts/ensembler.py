import numpy as np
from mlxtend.classifier import EnsembleVoteClassifier

# Stacking makes sense with full dataset, not with subsets
class Ensembler:
    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def hard_voting_pretrained_ensemble(self):
        voting_hard = EnsembleVoteClassifier(
            clfs=self.models,
            voting='hard',
            weights=[1] * len(self.models),
            fit_base_estimators=False
        )
        return voting_hard

    def soft_voting_pretrained_ensemble(self):
        voting_soft = EnsembleVoteClassifier(
            clfs=self.models,
            voting='soft',
            weights=[1] * len(self.models),
            fit_base_estimators=False
        )
        return voting_soft

    def hard_voting_ensemble(self):
        voting_hard = EnsembleVoteClassifier(
            clfs=self.models,
            voting='hard',
            weights=[1] * len(self.models)
        )
        return voting_hard

    def soft_voting_ensemble(self):
        voting_soft = EnsembleVoteClassifier(
            clfs=self.models,
            voting='soft',
            weights=[1] * len(self.models)
        )
        return voting_soft