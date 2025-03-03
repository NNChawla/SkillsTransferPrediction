from .pretrained_classifier import PreTrainedClassifier
from .trainer_tester import TrainerTester
from .ensembler import Ensembler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
import yaml

class Experiments:
    def __init__(self):
        with open('feature_config.yaml', 'r') as f:
            feature_config = yaml.safe_load(f)
        
        self.trainer_tester = TrainerTester()
        self.metadata_features = feature_config['metadata_fields']
        self.global_features = feature_config['global_features']

    def stepwise_voting_ensemble(self, type='hard', train_set='A'):
        ensembler = Ensembler()
        
        for i in range(13):
            svm = SVC(probability=True, kernel='rbf', random_state=42)
            trained_svm = self.trainer_tester.train_model(svm, train_set, self.metadata_features, self.global_features, cross_task=True, step=i)
            # wrapped_svm = PreTrainedClassifier(trained_svm, name=f'step_{i}_svm')
            ensembler.add_model(trained_svm)

        feature_names = self.trainer_tester.get_feature_names(self.metadata_features, self.global_features)
        voting_classifier = ensembler.soft_voting_pretrained_ensemble() if type == 'soft' else ensembler.hard_voting_pretrained_ensemble()        
        results = self.trainer_tester.cross_validate_model(voting_classifier, train_set, self.metadata_features, self.global_features, cross_task=True)
        self.trainer_tester.plot_model_results(results, train_set, f'stepwise_{type}_voting', feature_names)
        
        print(f"mcc: {results['feature_selection_measure']} mcc_std: {results['feature_selection_measure_std']}")

    def voting_ensemble(self, type='hard', train_set='A'):
        ensembler = Ensembler()

        ensembler.add_model(SVC(probability=True, kernel='rbf', random_state=42))
        ensembler.add_model(RandomForestClassifier(random_state=42))
        ensembler.add_model(LogisticRegression(random_state=42))
        ensembler.add_model(LGBMClassifier(random_state=42))
        ensembler.add_model(KNeighborsClassifier())

        feature_names = self.trainer_tester.get_feature_names(self.metadata_features, self.global_features)
        voting_classifier = ensembler.soft_voting_ensemble() if type == 'soft' else ensembler.hard_voting_ensemble()
        results = self.trainer_tester.cross_validate_model(voting_classifier, train_set, self.metadata_features, self.global_features, cross_task=True)
        self.trainer_tester.plot_model_results(results, train_set, f'{type}_voting', feature_names)
        
        print(f"mcc: {results['feature_selection_measure']} mcc_std: {results['feature_selection_measure_std']}")

    def object_voting_ensemble(self, type='hard', train_set='A'):
        ensembler = Ensembler()

        feature_names = self.trainer_tester.get_feature_names(self.metadata_features, self.global_features)

        objects = ["Head", "LeftHand", "RightHand"]

        for object in objects:
            object_global_features = {k: v for k, v in self.global_features.items() if object in k}

            svm = SVC(probability=True, kernel='rbf', random_state=42)
            trained_svm = self.trainer_tester.train_model(svm, train_set, self.metadata_features, object_global_features, cross_task=True)
            # wrapped_svm = PreTrainedClassifier(trained_svm, name=f'{object}_svm')
            ensembler.add_model(trained_svm)

        voting_classifier = ensembler.soft_voting_pretrained_ensemble() if type == 'soft' else ensembler.hard_voting_pretrained_ensemble()        
        results = self.trainer_tester.cross_validate_model(voting_classifier, train_set, self.metadata_features, self.global_features, cross_task=True)
        self.trainer_tester.plot_model_results(results, train_set, f'object_{type}_voting', feature_names)
        
        print(f"mcc: {results['feature_selection_measure']} mcc_std: {results['feature_selection_measure_std']}")
    
    def single_model(self, model_config, train_set='A'):
        feature_names = self.trainer_tester.get_feature_names(self.metadata_features, self.global_features)
        model = self.trainer_tester.get_model(model_config, 'classification')
        results = self.trainer_tester.cross_validate_model(model, train_set, self.metadata_features, self.global_features, cross_task=True)
        self.trainer_tester.plot_model_results(results, train_set, f"single_{model_config['type']}", feature_names)
        
        print(f"mcc: {results['feature_selection_measure']} mcc_std: {results['feature_selection_measure_std']}")