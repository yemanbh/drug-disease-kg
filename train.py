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

class EdgeClassifier(torch.nn.Module):
    """
    A graph-based edge classification model using GCN layers and an MLP for edge prediction.

    Parameters:
    - in_channels (int): Number of input features for each node.
    - hidden_channels (int): Number of hidden units in GCN layers.
    - num_classes (int): Number of edge classes to predict.
    - directed_graph (bool, optional): Whether the graph is directed or undirected. Default is True.
    - drop_rate (float, optional): Dropout rate for regularization. Default is 0.2.
    """
    def __init__(self, in_channels, hidden_channels, num_classes, directed_graph=True, drop_rate=0.2):
        super(EdgeClassifier, self).__init__()
        
        # GCN layers to update node features
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.act1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.act2 = nn.ReLU()
        
        # MLP for edge classification using concatenated node embeddings
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, 64),  # First layer of MLP
            nn.Dropout(drop_rate),              # Regularization
            nn.Linear(64, 32),                  # Second layer of MLP
            nn.Linear(32, num_classes)          # Output layer
        )
        
        # Flag to handle directed vs undirected graphs
        self.directed_graph = directed_graph

    def forward(self, x, edge_index):
        """
        Forward pass for edge classification.

        Parameters:
        - x (torch.Tensor): Node feature matrix.
        - edge_index (torch.Tensor): Edge list (source and target node indices).

        Returns:
        - out (torch.Tensor): Predicted edge class logits.
        """
        # Update node features using GCN layers
        x = self.act1(self.conv1(x, edge_index))
        x = self.act2(self.conv2(x, edge_index))
        
        # Handle undirected graphs by filtering specific edges
        if not self.directed_graph:
            # Extract source and target nodes for undirected edges
            source, target = edge_index[:, ::2]
            edge_features = torch.cat([x[source], x[target]], dim=1)
        else:
            # Use all edges for directed graphs
            source, target = edge_index
            edge_features = torch.cat([x[source], x[target]], dim=1)
        
        # Pass edge features through MLP for classification
        out = self.edge_mlp(edge_features)
        return out

def train_model(model, train_data, val_data, optim, criteria, output_dir, epochs=10, device='cpu', directed_graph=True, early_stop=20):
    """
    Train a graph-based edge classifier and validate it periodically.

    Parameters:
    - model (torch.nn.Module): The edge classifier model.
    - train_data (Data): Training graph data (features, edges, and labels).
    - val_data (Data): Validation graph data (features, edges, and labels).
    - optim (torch.optim.Optimizer): Optimizer for training.
    - criteria (torch.nn.Module): Loss function for training.
    - output_dir (str): Directory to save the best model during training.
    - epochs (int, optional): Number of training epochs. Default is 10.
    - device (str, optional): Device for computation ('cpu' or 'cuda'). Default is 'cpu'.
    - directed_graph (bool, optional): Whether the graph is directed. Default is True.
    - early_stop (int, optional): Number of epochs to wait for validation improvement before stopping. Default is 20.

    Returns:
    - train_loss_hist (list): History of training losses.
    - val_loss_hist (list): History of validation losses.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for early stopping and tracking losses
    best_val_loss = float("inf")
    patience = 0
    val_loss_hist = []
    train_loss_hist = []

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training/validation", total=epochs):
        # Train the model
        model.train()
        optim.zero_grad()
        out = model(train_data.x.to(device), train_data.edge_index.to(device))
        
        # Adjust labels for directed or undirected graphs
        y = train_data.y[::2] if not directed_graph else train_data.y
        train_loss = criteria(out, y.to(device))  # Compute training loss
        train_loss.backward()  # Backpropagation
        optim.step()  # Update model weights
        train_loss_hist.append(train_loss.item())

        # Validate the model
        model.eval()
        with torch.no_grad():
            out = model(val_data.x.to(device), val_data.edge_index.to(device))
            y = val_data.y[::2] if not directed_graph else val_data.y
            val_loss = criteria(out, y.to(device))  # Compute validation loss
            val_loss_hist.append(val_loss.item())

        # Print training and validation loss every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs}: train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

        # Check for validation improvement and save the model if improvement occurs
        if (best_val_loss - val_loss.item()) > 1e-3:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            patience = 0  # Reset patience counter
        else:
            patience += 1
            if patience > early_stop:
                # Stop training early if no improvement for `early_stop` epochs
                break

    return train_loss_hist, val_loss_hist

    