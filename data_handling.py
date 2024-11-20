import os, re
import pandas as pd 
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
import torch

def get_embd_as_list_float(x):
    """
    Converts a string representation of a list (e.g., "[1, 2, 3]") with potential
    irregular formatting (e.g., whitespace, newlines) into a list of floats.

    Parameters:
    - x (str): A string representation of a list of numerical values.

    Returns:
    - x_list_float (list of float): The parsed list of floats.
    """
    # Remove unnecessary spaces or newlines and replace them with commas
    x = re.sub(r"[\s\n]+", ",", x)
    
    # Remove square brackets and strip trailing commas
    x = re.sub(r"[\[\]]", "", x).strip(",")
    
    # Split the string into individual numbers and convert them to floats
    x_list_float = [float(num) for num in x.split(',')]
    
    return x_list_float



def plot_nodes_degree_distribution(data, comment=None):
    """
    Plots the degree distribution of source and target nodes from the provided graph data.

    Parameters:
    - data (pd.DataFrame): DataFrame with columns 'source' and 'target' representing graph edges.
    - comment (str, optional): Additional comment to add as a prefix to the plot titles.

    Output:
    - Two bar plots:
      1. Degree distribution of source nodes.
      2. Degree distribution of target nodes.
    """
    # Reset the index to ensure proper plot alignment
    data.reset_index(drop=True, inplace=True)
    
    # Plot the degree distribution of source nodes
    ax = data["source"].value_counts().plot(kind='bar')
    ax.set_xticks([])  # Hide x-axis tick marks
    ax.set_ylabel("Degree")
    ax.set_xlabel("Source node")
    ax.set_title(f"{comment + ':' if comment else ''} Distribution of degree of source nodes")
    plt.show()
    
    # Plot the degree distribution of target nodes
    ax1 = data["target"].value_counts().plot(kind='bar')
    ax1.set_xticks([])  # Hide x-axis tick marks
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Target node")
    ax1.set_title(f"{comment + ':' if comment else ''} Distribution of degree of target nodes")
    plt.show()

    
def stratified_train_test_split(df, group_col, target_col, test_size=0.4, random_state=1234):
    """
    Splits a DataFrame into stratified training and testing datasets while ensuring no overlap
    in the specified groups between the two sets.

    Parameters:
    - df (pd.DataFrame): Input data containing groups and target labels.
    - group_col (str): Column representing groups (e.g., 'target' or 'source').
    - target_col (str): Target label column for stratification.
    - test_size (float): Proportion of the data to be included in the test set (default is 0.4).
    - random_state (int): Random seed for reproducibility (default is 1234).

    Returns:
    - train_df (pd.DataFrame): Stratified training dataset.
    - test_df (pd.DataFrame): Stratified testing dataset.

    Raises:
    - AssertionError: If there is any overlap of target labels between the training and testing datasets.
    """
    # Group the data by the specified group column
    grouped = df.groupby(group_col)
    
    # Create a list of unique group identifiers
    groups = list(grouped.groups.keys())
    
    # Map the target labels to each group by taking the mode (most frequent target)
    group_targets = grouped[target_col].apply(lambda x: x.mode()[0])
    
    # Perform stratified splitting based on the mode of target labels
    train_groups, test_groups = train_test_split(
        groups, test_size=test_size, stratify=group_targets, random_state=random_state
    )
    
    # Filter the original DataFrame to create train and test sets
    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    # Ensure that there is no overlap of target labels between the train and test sets
    assert len(set(train_df.target).intersection(test_df.target)) == 0, "There is overlap between train and test sets."
    
    return train_df, test_df


def get_train_val_test_graph(gt):
    """
    Splits the input data into training, validation, and test sets using stratified splits.

    Parameters:
    - gt (pd.DataFrame): Input graph data with nodes and target labels.

    Returns:
    - train (pd.DataFrame): Stratified training dataset.
    - val (pd.DataFrame): Stratified validation dataset.
    - test (pd.DataFrame): Stratified testing dataset.
    """
    # Perform the first stratified split into training and test sets
    train, test = stratified_train_test_split(gt, "target", "y", test_size=0.2)
    
    # Perform a second stratified split to create a validation set from the training set
    train, val = stratified_train_test_split(train, "target", "y", test_size=0.2)

    # Print the sizes of the resulting datasets
    print("Training set size:", len(train))
    print("Test set size:", len(test))
    print("Validation set size:", len(val))

    return train, val, test


def create_graph_data(embedding_csv, edge_csv, directed_graph=True):
    """
    Creates graph data from the node embeddings and edges, including features, edge indices, and labels.

    Parameters:
    - embedding_csv (pd.DataFrame): DataFrame containing node embeddings.
    - edge_csv (pd.DataFrame): DataFrame containing the edges with source, target, and labels.
    - directed_graph (bool): Whether the graph is directed or undirected (default is True for directed).

    Returns:
    - data (torch_geometric.data.Data): Graph data including node features, edge indices, and edge labels.
    """
    # Extract the list of nodes from the edge DataFrame
    nodes_list = edge_csv.target.unique().tolist() + edge_csv.source.unique().tolist()
    
    # Select the embeddings for nodes that are part of the edges
    embedding_csv_selected = embedding_csv.loc[embedding_csv['id'].isin(nodes_list), :].reset_index(drop=True)
    
    # Create a tensor of node features from the selected embeddings
    node_features = torch.tensor(embedding_csv_selected['topological_embedding'].tolist(), dtype=torch.float32)
    
    # Initialize edge indices and labels for the graph
    edge_csv.reset_index(drop=True, inplace=True)
    n_edges = len(edge_csv)
    
    if directed_graph:
        edge_index = torch.zeros(2, n_edges, dtype=torch.int32)
        edge_label = torch.zeros(n_edges, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 2 * n_edges, dtype=torch.int32)
        edge_label = torch.zeros(2 * n_edges, dtype=torch.long)
        
    # Map edges and labels to the graph data
    idx = 0
    for _, row in edge_csv.iterrows():
        node1 = embedding_csv_selected.loc[embedding_csv_selected['id'] == row['source'], :].index[0]
        node2 = embedding_csv_selected.loc[embedding_csv_selected['id'] == row['target'], :].index[0]

        if directed_graph:
            edge_index[0, idx], edge_index[1, idx] = node1, node2
            edge_label[idx] = row['y']
            idx += 1
        else:
            edge_index[0, idx], edge_index[1, idx] = node1, node2
            edge_index[0, idx+1], edge_index[1, idx+1] = node2, node1
            edge_label[idx], edge_label[idx+1] = row['y'], row['y']
            idx += 2
            
    # Create the graph data object
    data = Data(x=node_features, edge_index=edge_index, y=edge_label)
    
    # Print the dimensions of the created graph data
    print(f"Node features/embedding dimension:{data.x.shape}")
    print(f"Edge index dimension:{data.edge_index.shape}")
    print(f"Edge labels dimension:{data.y.shape}")
    
    return data


def get_graph_data(embd_csv, train, val, test, directed_graph):
    """
    Creates graph data for the training, validation, and test datasets by calling the
    `create_graph_data` function for each dataset (train, val, test). The function processes
    the node embeddings and edges for each dataset and returns the corresponding graph data objects.

    Parameters:
    - embd_csv (pd.DataFrame): DataFrame containing node embeddings.
    - train (pd.DataFrame): DataFrame containing training edges and labels.
    - val (pd.DataFrame): DataFrame containing validation edges and labels.
    - test (pd.DataFrame): DataFrame containing test edges and labels.
    - directed_graph (bool): Whether the graph is directed or undirected (default is True).

    Returns:
    - train_graph_data (torch_geometric.data.Data): Graph data for the training set.
    - val_graph_data (torch_geometric.data.Data): Graph data for the validation set.
    - test_graph_data (torch_geometric.data.Data): Graph data for the test set.
    """
    # Create graph data for the training dataset
    print("Create TRAINING graph data...")
    train_graph_data = create_graph_data(embd_csv, train, directed_graph=directed_graph)
    
    # Create graph data for the validation dataset
    print("Create VALIDATION graph data..")
    val_graph_data = create_graph_data(embd_csv, val, directed_graph=directed_graph)
    
    # Create graph data for the test dataset
    print("Create TEST graph data...")
    test_graph_data = create_graph_data(embd_csv, test, directed_graph=directed_graph)

    # Return the graph data for all three datasets
    return train_graph_data, val_graph_data, test_graph_data










