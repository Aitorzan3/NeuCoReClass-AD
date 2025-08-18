# Utils.py
import torch
from torch import nn
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch.utils.data import random_split
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def contrastive_loss(x, pos_indices, temperature, n_transforms, measure):
    """
    Computes contrastive loss for a batch using softmax-like normalization.

    Args:
    - x: Tensor of embeddings of shape [batch_size, embedding_dim].
    - pos_indices: Tensor of indices where positive pairs are located, of shape [num_pos_pairs, 2].
    - temperature: Temperature scaling factor.

    Returns:
    - loss: Contrastive loss scalar.
    """

    x = F.normalize(x, p=2, dim=-1).to(device)

    # Calculate pairwise cosine similarities between all pairs

    if (measure=="euclidean"):
        sim = -torch.cdist(x, x, p=2).to(device)
    elif (measure=="cosine"):
        sim = torch.mm(x, x.t()).to(device)  # Pairwise similarities [batch_size, batch_size]
    # Apply temperature scaling
    sim = sim / temperature

    # Exponentiate the scaled cosine similarities (softmax-like behavior)
    exp_sim = torch.exp(sim)

    # Create mask for positive pairs
    pos_mask = torch.eye(exp_sim.shape[0]).to(device)
    
    pos_mask[pos_indices[:, 0], pos_indices[:, 1]] = 1
    pos_mask[pos_indices[:, 1], pos_indices[:, 0]] = 1

    # Numerator: Positive pair similarities (same transformation pairs)  
    numerator = exp_sim * pos_mask
    numerator = numerator.sum(dim=1, keepdim=True)

    # Denominator: Sum over all similarities (both positive and negative)
    denominator = exp_sim.sum(dim=1, keepdim=True)

    # Calculate the contrastive loss: -log(numerator / denominator)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    loss = torch.log(denominator + epsilon) - torch.log(numerator + epsilon)
    scale = 1 / n_transforms
    loss = loss.view(n_transforms, -1).sum(dim=0) * scale
    return loss.mean()


def get_positives(batch_size, n_transforms):
    num_augments = batch_size * n_transforms
    indices = torch.arange(num_augments).view(n_transforms, batch_size)  # Create indices grouped by transformation
    
   # Create all the possible pairs
    pairs = []
    for group in indices: 
        group_pairs = torch.combinations(group, r=2)
        pairs.append(group_pairs)
   
    # Concat all the possible pairs
    pairs = torch.cat(pairs, dim=0)
    reversed_pairs = pairs.flip(dims=[1])
    all_pairs = torch.cat((pairs, reversed_pairs), dim=0)
    
    return all_pairs

def load_positives(loader, n_transforms):
    positives = []
    for batch in loader:
        positives.append(get_positives(batch.shape[0], n_transforms))
    return positives

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def metrics(anomaly_scores, true_labels):
    true_labels = 1 - true_labels

    # Compute precision, recall, and thresholds
    precision, recall, _ = precision_recall_curve(true_labels, anomaly_scores)

    # Compute AUC-PR using the trapezoidal rule
    auc_pr = auc(recall, precision)
    
    # Compute AUC-ROC
    auc_roc = roc_auc_score(true_labels, anomaly_scores)
    
    return auc_roc, auc_pr

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.Generator().manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
