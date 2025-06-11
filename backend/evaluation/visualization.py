"""Functions for building and storing embeddings and storing data such as nearest neighbors as pt files in the enbeddings directory.
Image neighbors are stored as a pt file to prevent having to compute at startup, TODO should other neighbor type such as shark neighbors 
also be stored as pt files, with different comprison methods this could eventually get out of hand."""

import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import sys
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NeighborhoodComponentsAnalysis

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from backend.constants import EMBEDDINGS_DIR

import repo_utils
import backend.utils

def load_embeddings_labels(model_name, centroid=False, train_test=False):
    """
    Load the embeddings and dataset stored as PT files.
    """
    device = repo_utils.get_torch_device()
    
    if not centroid:
        print("Loading image embeddings.")
        embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_embeddings.pt"), map_location=device, weights_only=False).cpu().detach().numpy()
        dataset = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"), map_location=device, weights_only=False)
        ids = dataset.get_ids()
        labels = dataset.get_labels().cpu().detach().numpy()

        # Split to just the train or test embeddings.
        if train_test:
            focus_embeddings = []
            focus_labels = []
            train_ids = dataset.get_train_ids()
            test_ids = dataset.get_test_ids()
            for index, id in enumerate(ids):
                if id in test_ids: # Select train or test here.
                    focus_embeddings.append(embeddings[index])
                    focus_labels.append(labels[index])
            embeddings = np.array(focus_embeddings)
            labels = np.array(focus_labels)

    else:
        # TODO train/test loading for centroids.
        print("Loading centroid embeddings.")
        embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_centroid_embeddings.pt"), map_location=device, weights_only=False).cpu().detach().numpy()
        labels = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_centroid_labels.pt"), map_location=device, weights_only=False).cpu().detach().numpy()
    
    return embeddings, labels

def filter_embeddings(embeddings, labels, min_count):
    """
    Filter the embeddings and labels to only those where there are min_count embeddings per label.
    """
    # Count label occurrences. Filter labels that occur more than x times.
    label_counts = pd.Series(labels).value_counts()
    frequent_labels = label_counts[label_counts >= min_count].index.tolist()

    # Filter embeddings and labels
    filtered_indices = np.isin(labels, frequent_labels)
    filtered_embeddings = embeddings[filtered_indices]
    filtered_labels = labels[filtered_indices]

    print("Number of embeddings that meet the count per lable threshold:", len(filtered_embeddings))
    print("Number of unique labels represented", len(set(filtered_labels)))

    return filtered_embeddings, filtered_labels

def visualize_embeddings_nca(embeddings, labels):
    """
    Dimensionality reduction technique ideal for KNN.
    """
    
    nca = NeighborhoodComponentsAnalysis(n_components=3, random_state=1)
    reduced_embeddings = nca.fit_transform(embeddings, labels)

    print(len(labels))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    z = reduced_embeddings[:, 2]

    scatter = ax.scatter(x, y, z, c=labels, cmap=cm.viridis) # Added color based on labels
    ax.set_xlabel('NC1')
    ax.set_ylabel('NC2')
    ax.set_zlabel('NC3')
    plt.colorbar(scatter)
    plt.title('3D NCA of Embeddings')
    plt.show()

def visualize_embeddings_pca(embeddings, labels):
    """
    A linear dimensionality reduction technique that identifies the principal components (directions of maximum variance) in the data and projects the data onto them.
    """
    pca = PCA(n_components=3, random_state=0)
    reduced_embeddings = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    z = reduced_embeddings[:, 2]

    scatter = ax.scatter(x, y, z, c=labels, cmap=cm.viridis) # Added color based on labels
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter)
    plt.title('3D PCA of Embeddings')
    plt.show()
    
def visualize_embeddings_tsne(embeddings, labels):
    """
    A non-linear dimensionality reduction technique particularly effective at preserving local structure in the data, making it suitable for visualizing clusters.
    """
    tsne = TSNE(n_components=3, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    z = reduced_embeddings[:, 2]

    scatter = ax.scatter(x, y, z, c=labels, cmap=cm.viridis) # Added color based on labels
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Embeddings')
    plt.show()

def main():
    embeddings, labels = load_embeddings_labels(model_name="lora", centroid=False, train_test=False)
    embeddings, labels = filter_embeddings(embeddings, labels, 10)
    visualize_embeddings_nca(embeddings, labels)
    # visualize_embeddings_pca(embeddings, labels)
    visualize_embeddings_tsne(embeddings, labels)
    pass

if __name__ == "__main__":
    main()
