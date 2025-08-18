from NeuCoReClassAD import *

if __name__=="__main__":
    hyperparameters = {
        "dataset": "Epilepsy",
        "normality": 'sawing',
        "reverse": True,
        "n_transforms":12,
        "measure":"cosine",
        "temperature":0.1,
        "max_epochs": 1,
        "batch_size":32,
        "verbose":True,
        "seed": 123,
        }
    run_experiment(hyperparameters)