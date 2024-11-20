import os, re
import pandas as pd 
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

def run_inference(model, test_data, device, directed_graph):
    """
    Perform inference on test data using a trained model.

    Parameters:
    - model (torch.nn.Module): The trained graph-based model.
    - test_data (Data): The test dataset containing node features, edge indices, and labels.
    - device (torch.device): The device (CPU/GPU) to run the inference on.
    - directed_graph (bool): Indicates whether the graph is directed or undirected.

    Returns:
    - targets (torch.Tensor): The ground-truth labels for the test dataset.
    - preds (torch.Tensor): The predicted labels from the model.
    """
    # Set the model to evaluation mode to disable training-specific operations like dropout.
    model.eval()

    # Adjust the ground-truth labels based on whether the graph is directed or undirected.
    # For undirected graphs, labels may be sampled or reduced to avoid duplicates.
    if not directed_graph:
        targets = test_data.y[::2, ]  # Use every second label for undirected graphs.
    else:
        targets = test_data.y  # Use all labels for directed graphs.

    with torch.no_grad():
        # prediction.
        outputs = model(test_data.x.to(device), test_data.edge_index.to(device))
        # Get predicted class indices by taking the maximum value along the class dimension.
        _, preds = torch.max(outputs.cpu(), 1)

    return targets, preds


def compute_metrics(targets, preds):
    """
    Compute classification metrics and a confusion matrix for model predictions.

    Parameters:
    - targets (array-like): Ground-truth labels.
    - preds (array-like): Predicted labels.

    Returns:
    - report (str): Classification report containing precision, recall, F1-score, and support.
    - cm (array): Confusion matrix showing true positive, false positive, etc., for each class.
    """
    # Calculate the confusion matrix to summarize classification performance.
    cm = confusion_matrix(targets, preds)
    # Generate a detailed classification report with precision, recall, F1-score, and support for each class.
    report = classification_report(targets, preds, target_names=["No-link", "Link-exist"])

    return report, cm
