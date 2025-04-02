from scripts.experiments import Experiments

if __name__ == "__main__":
    experiments = Experiments()

    print("Stepwise Hard Voting Ensemble")
    experiments.stepwise_voting_ensemble(type='hard', train_set='A')
    experiments.stepwise_voting_ensemble(type='hard', train_set='B')

    print("Stepwise Soft Voting Ensemble")
    experiments.stepwise_voting_ensemble(type='soft', train_set='A')
    experiments.stepwise_voting_ensemble(type='soft', train_set='B')

    print("Hard Voting Ensemble")
    experiments.voting_ensemble(type='hard', train_set='A')
    experiments.voting_ensemble(type='hard', train_set='B')

    print("Soft Voting Ensemble")
    experiments.voting_ensemble(type='soft', train_set='A')
    experiments.voting_ensemble(type='soft', train_set='B')

    print("Single Model")
    experiments.single_model(model_config={'type': 'svm'}, train_set='A')
    experiments.single_model(model_config={'type': 'svm'}, train_set='B')

    print("Object Voting Ensemble")
    experiments.object_voting_ensemble(type='hard', train_set='A')
    experiments.object_voting_ensemble(type='hard', train_set='B')

    print("Object Soft Voting Ensemble")
    experiments.object_voting_ensemble(type='soft', train_set='A')
    experiments.object_voting_ensemble(type='soft', train_set='B')
