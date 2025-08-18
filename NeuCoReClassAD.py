from models import *
from datasets import *
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_experiment(hyperparams):

    dataset = hyperparams.get('dataset') if hyperparams.get('dataset') is not None else 5
    reverse = hyperparams.get('reverse') if hyperparams.get('reverse') is not None else 0
    normality = hyperparams.get('normality') if hyperparams.get('normality') is not None else 0
    n_transforms = hyperparams.get('n_transforms') if hyperparams.get('n_transforms') is not None else 12
    measure = hyperparams.get('measure') if hyperparams.get('measure') is not None else 'cosine'
    temperature = hyperparams.get('temperature') if hyperparams.get('temperature') is not None else 0.1
    max_epochs = hyperparams.get('max_epochs') if hyperparams.get('max_epochs') is not None else 1
    batch_size = hyperparams.get('batch_size') if hyperparams.get('batch_size') is not None else 32
    verbose = hyperparams.get('verbose') if hyperparams.get('verbose') is not None else False
    seed = hyperparams.get('seed')

    if not (measure == 'cosine' or measure == 'euclidean'):
        raise Exception("The similarity measure for the contrastive loss must be 'cosine' or 'euclidean' (based on negative euclidean distance)")
    
    dataset, train_data, test_data, test_labels, _  = get_dataset(dataset, normality, reverse)
    
    if seed is not None:
        set_seed(seed)

    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    input_dims = train_data[0].shape[1]
    model = Experiment(input_dims = input_dims, train_data = train_dataset, val_data = val_dataset, test_data = test_data, n_transforms = n_transforms, temperature = temperature, batch_size = batch_size, measure = measure).to(device)

    start = time.time()
    model._net.train()

    if seed is not None:
        set_seed(seed)
    model.train_model(max_epochs=max_epochs, verbose=verbose)
    end = time.time()
    training_time = end - start

    model.eval()
    start = time.time()
    scores = model.compute_scores()
    auroc, aupr = metrics(scores, test_labels)
    end = time.time()
    inference_time = end - start

    print("training time: ", training_time)
    print("inference time: ", inference_time)
    print("AUROC: ", auroc)
    print("AUPR: ", aupr)
    

