import torch

from typing import Callable

def create_triplet_loss() -> Callable:
    def batch_hard_triplet_loss(embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute pairwise distances between embeddings.
        pairwise_distances = torch.cdist(embedding, embedding, p=2.0)

        # Find the hardest positive sample for each anchor.
        hardest_positives = torch.zeros(pairwise_distances.size()[0], dtype=torch.int64, device=pairwise_distances.device)
        for i in range(pairwise_distances.size()[0]):
            max_positive_distance = float("-inf")
            for j in range(pairwise_distances.size()[1]):
                if labels[i] == labels[j]:
                    if pairwise_distances[i, j] > max_positive_distance:
                        max_positive_distance = pairwise_distances[i, j]
                        hardest_positives[i] = j

        # Find the hardest negative sample for each anchor.
        hardest_negatives = torch.zeros(pairwise_distances.size()[0], dtype=torch.int64, device=pairwise_distances.device)
        for i in range(pairwise_distances.size()[0]):
            min_negative_distance = float("inf")
            for j in range(pairwise_distances.size()[1]):
                if labels[i] != labels[j]:
                    if pairwise_distances[i, j] < min_negative_distance:
                        min_negative_distance = pairwise_distances[i, j]
                        hardest_negatives[i] = j

        # Compute the anchor-positive and anchor-negative distances.
        anchor_positive_distances = pairwise_distances[torch.arange(pairwise_distances.size()[0]), hardest_positives]
        anchor_negative_distances = pairwise_distances[torch.arange(pairwise_distances.size()[0]), hardest_negatives]

        # Compute the triplet loss.
        return torch.sum(torch.nn.functional.softplus(anchor_positive_distances - anchor_negative_distances))

    return batch_hard_triplet_loss
