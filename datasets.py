# Dataset.py
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import aeon
from aeon.datasets import load_classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfDataset(Dataset):
    def __init__(self,feature):
        self.feature = feature
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        
        return item
    
def get_dataset(dataset, normal, reverse=False):

    train_data, train_labels, metadata = load_classification(dataset, split="train", load_equal_length=True, load_no_missing=True, return_metadata=True)
    test_data, test_labels = load_classification(dataset, split="test", load_equal_length=True, load_no_missing=True)


    real_labels = test_labels.copy()

    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    
        
    if normal not in real_labels:
        raise Exception("You need to choose an existing class value for parameter 'normal'. Possible values: ", np.unique(real_labels))

    normal = le.transform(np.array(normal).reshape(1,))[0]

    if not reverse:
        train_data = train_data[train_labels==normal]
        test_labels[test_labels!=normal]=-1
        test_labels[test_labels==normal]=1
        test_labels[test_labels==-1]=0
    else:
        train_data = train_data[train_labels!=normal]
        test_labels[test_labels==normal]=-1
        test_labels[test_labels!=-1]=1
        test_labels[test_labels==-1]=0

    print("Problem name: ", metadata['problemname'])
    print("Existing Classes: ", metadata['class_values'])
    print("Normal Class(es): ", np.unique(real_labels[np.where(test_labels==1)]))
    print("Anomalous Class(es): ", np.unique(real_labels[np.where(test_labels==0)]))

    scalers = [StandardScaler() for _ in range(train_data.shape[2])]

    for i in range(train_data.shape[2]):
        scalers[i].fit(train_data[:, :, i])
        train_data[:, :, i] = scalers[i].transform(train_data[:, :, i])
        test_data[:, :, i] = scalers[i].transform(test_data[:, :, i])

    return dataset, train_data.astype(np.float32) , test_data.astype(np.float32) , test_labels.astype(np.float32).astype(int), real_labels