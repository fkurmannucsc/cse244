import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader # Import DataLoader

from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

from backend.constants import MINI_BATCH_SIZE

def eval_collate_fn(batch):
    """
    Collate function for evaluation. Handles (image, mask_or_None, label) tuples.
    Masks are collected into a list and are not stacked if None is present.
    """
    # batch: list of (image_tensor, mask_tensor_or_None, label_tensor)
    images = torch.stack([item[0] for item in batch], dim=0)
    
    # Collect masks into a list. This part will be ignored by the model call in hits_at_k.
    masks = [item[1] for item in batch] 
    
    labels = torch.stack([item[2] for item in batch], dim=0)
    return images, masks, labels


def hits_at_k(model: torch.nn.Module,
              train_dataset: Dataset,
              valid_dataset: Dataset,
              device: torch.device) -> (float, dict[str, float]):

    # Compute embeddings for the training and validation datasets in batches.
    train_embeddings = []
    train_loader = DataLoader(train_dataset, batch_size=MINI_BATCH_SIZE, shuffle=False, collate_fn=eval_collate_fn) # Use custom collate_fn
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Ensure no gradients are computed during evaluation
        for images_batch, _, _ in train_loader: # Unpack 3 items, use images_batch
            train_embeddings.append(model(images_batch.to(device)).cpu().numpy())

    # Correctly get all labels from the dataset
    # Assumes train_dataset is an instance of SharkfinDataset or similar
    # and get_labels() returns a torch tensor on CPU.
    train_labels = train_dataset.get_labels().cpu().numpy()

    train_label_counts = np.unique(train_labels, return_counts=True)
    train_label_counts_dict = dict(zip(train_label_counts[0], train_label_counts[1]))

    # dim_reduced_train_embedding = PCA(n_components=10).fit_transform(train_embedding)
    # dim_reduced_valid_embedding = PCA(n_components=10).fit_transform(valid_embedding)
    train_embeddings = np.concatenate(train_embeddings, axis=0) # Concatenate after the loop

    # Determine the number of neighbors for KNN
    num_train_embeddings = len(train_embeddings)
    n_neighbors_val = min(200, num_train_embeddings)
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors_val, metric='euclidean').fit(train_embeddings)

    # Compute hits@k with nearest neighbors.
    average_train_hit_at_1 = 0
    average_train_hit_at_5 = 0
    average_train_hit_at_10 = 0
    average_train_hit_at_25 = 0
    average_train_hit_at_50 = 0
    average_train_hit_at_100 = 0

    average_valid_hit_at_1 = 0
    average_valid_hit_at_5 = 0
    average_valid_hit_at_10 = 0
    average_valid_hit_at_25 = 0
    average_valid_hit_at_50 = 0
    average_valid_hit_at_100 = 0

    # Compute training nearest neighbors
    _, train_neighbor_indices = nearest_neighbors.kneighbors(train_embeddings)

    # Training hits@k evaluation.
    num_training_samples = 0
    for index, neighbors in enumerate(train_neighbor_indices):
        if train_label_counts_dict[train_labels[index]] == 1:
            continue

        unique_train_neighbor_label_indices = np.sort(np.unique(train_labels[neighbors[1:]], return_index=True)[1])
        train_neighbor_labels = train_labels[neighbors[1:]][unique_train_neighbor_label_indices]

        train_hit_at_1, train_hit_at_5, train_hit_at_10, train_hit_at_25, train_hit_at_50, train_hit_at_100 = __hits_at_k__(train_labels[index], train_neighbor_labels)

        average_train_hit_at_1 += train_hit_at_1
        average_train_hit_at_5 += train_hit_at_5
        average_train_hit_at_10 += train_hit_at_10
        average_train_hit_at_25 += train_hit_at_25
        average_train_hit_at_50 += train_hit_at_50
        average_train_hit_at_100 += train_hit_at_100

        num_training_samples += 1

    average_train_hit_at_1 /= num_training_samples
    average_train_hit_at_5 /= num_training_samples
    average_train_hit_at_10 /= num_training_samples
    average_train_hit_at_25 /= num_training_samples
    average_train_hit_at_50 /= num_training_samples
    average_train_hit_at_100 /= num_training_samples

    # --- Validation hits@k evaluation (conditional) ---
    process_validation = valid_dataset is not None and len(valid_dataset) > 0

    if process_validation:
        valid_loader = DataLoader(valid_dataset, batch_size=MINI_BATCH_SIZE, shuffle=False, collate_fn=eval_collate_fn) # Use custom collate_fn
        valid_embeddings = []
        model.eval() # Ensure model is in eval mode
        with torch.no_grad(): # Ensure no gradients are computed
            for images_batch, _, _ in valid_loader: # Unpack 3 items, use images_batch
                valid_embeddings.append(model(images_batch.to(device)).cpu().numpy())
        
        if not valid_embeddings: # Should not happen if len(valid_dataset) > 0, but as a safeguard
            print("Warning: Validation dataset provided but no embeddings were generated.")
            process_validation = False
        else: # Only concatenate if embeddings were generated
            valid_embeddings = np.concatenate(valid_embeddings, axis=0)
            # Correctly get all labels from the dataset
            # Assumes valid_dataset is an instance of SharkfinDataset or similar
            # and get_labels() returns a torch tensor on CPU.
            valid_labels = valid_dataset.get_labels().cpu().numpy()
            _, valid_neighbor_indices = nearest_neighbors.kneighbors(valid_embeddings)

            for index, neighbors in enumerate(valid_neighbor_indices):
                # Neighbors are indices into the train_embeddings/train_labels
                unique_valid_neighbor_label_indices = np.sort(np.unique(train_labels[neighbors], return_index=True)[1])
                valid_neighbor_labels = train_labels[neighbors][unique_valid_neighbor_label_indices]

                # The query label is from the validation set
                valid_hit_at_1, valid_hit_at_5, valid_hit_at_10, valid_hit_at_25, valid_hit_at_50, valid_hit_at_100 = __hits_at_k__(valid_labels[index], valid_neighbor_labels)

                average_valid_hit_at_1 += valid_hit_at_1
                average_valid_hit_at_5 += valid_hit_at_5
                average_valid_hit_at_10 += valid_hit_at_10
                average_valid_hit_at_25 += valid_hit_at_25
                average_valid_hit_at_50 += valid_hit_at_50
                average_valid_hit_at_100 += valid_hit_at_100

            if len(valid_neighbor_indices) > 0:
                average_valid_hit_at_1 /= len(valid_neighbor_indices)
                average_valid_hit_at_5 /= len(valid_neighbor_indices)
                average_valid_hit_at_10 /= len(valid_neighbor_indices)
                average_valid_hit_at_25 /= len(valid_neighbor_indices)
                average_valid_hit_at_50 /= len(valid_neighbor_indices)
                average_valid_hit_at_100 /= len(valid_neighbor_indices)

    # Ensure model is returned to training mode if needed by the trainer
    # model.train() # The Trainer class handles setting train/eval mode

    # Return the validation hit@10 and the full dictionary of results
    return average_valid_hit_at_10, {
        "average_train_hit_at_1": average_train_hit_at_1 if num_training_samples > 0 else 0.0,
        "average_train_hit_at_5": average_train_hit_at_5 if num_training_samples > 0 else 0.0,
        "average_train_hit_at_10": average_train_hit_at_10 if num_training_samples > 0 else 0.0,
        "average_train_hit_at_25": average_train_hit_at_25 if num_training_samples > 0 else 0.0,
        "average_train_hit_at_50": average_train_hit_at_50 if num_training_samples > 0 else 0.0,
        "average_train_hit_at_100": average_train_hit_at_100 if num_training_samples > 0 else 0.0,
        "average_valid_hit_at_1": average_valid_hit_at_1 if process_validation and len(valid_neighbor_indices) > 0 else 0.0,
        "average_valid_hit_at_5": average_valid_hit_at_5 if process_validation and len(valid_neighbor_indices) > 0 else 0.0,
        "average_valid_hit_at_10": average_valid_hit_at_10 if process_validation and len(valid_neighbor_indices) > 0 else 0.0,
        "average_valid_hit_at_25": average_valid_hit_at_25 if process_validation and len(valid_neighbor_indices) > 0 else 0.0,
        "average_valid_hit_at_50": average_valid_hit_at_50 if process_validation and len(valid_neighbor_indices) > 0 else 0.0,
        "average_valid_hit_at_100": average_valid_hit_at_100 if process_validation and len(valid_neighbor_indices) > 0 else 0.0,
    }

def __hits_at_k__(label: np.array, neighbor_labels: np.array) -> (int, int, int, int):
    """
    Helper function that computes and returns the hits at k given the actual label and 
    a list of nearest neighbor labels.
    """
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    hit_at_25 = 0
    hit_at_50 = 0
    hit_at_100 = 0


    for i in range(0, min(100, len(neighbor_labels))):
        if label == neighbor_labels[i]:
            if i < 1:
                hit_at_1 = 1
                hit_at_5 = 1
                hit_at_10 = 1
                hit_at_25 = 1
                hit_at_50 = 1
            if i < 5:
                hit_at_5 = 1
                hit_at_10 = 1
                hit_at_25 = 1
                hit_at_50 = 1
            if i < 10:
                hit_at_10 = 1
                hit_at_25 = 1
                hit_at_50 = 1
            if i < 25:
                hit_at_25 = 1
                hit_at_50 = 1
            if i < 50:
                hit_at_50 = 1
            if i < 100:
                hit_at_100 = 1

            break

    return hit_at_1, hit_at_5, hit_at_10, hit_at_25, hit_at_50, hit_at_100

def main():
    pass

if __name__ == "__main__":
    main()
