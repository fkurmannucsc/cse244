import os
import sys
import torch

from torch.utils.data.sampler import Sampler
from typing import Iterator

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

from backend.sharkfin_dataset import SharkfinDataset

"""
A torch sampler that that first permutes the indices 
and then inserts indices of images with the same label in a contiguous block.
"""
class PositiveMatchSampler(Sampler):
    def __init__(self, dataset: SharkfinDataset, num_positive_samples: int = 3):
        super(PositiveMatchSampler, self).__init__(dataset)

        self.dataset = dataset

        self.num_positive_samples = num_positive_samples

        self.indices = list(range(len(dataset)))

        # Create a map from labels to lists of indices.
        self.label_to_indices = {}
        for index in range(len(dataset)):
            _, _, label = dataset[index] # Unpack image, mask, and label; use only label
            if label.item() not in self.label_to_indices:
                self.label_to_indices[label.item()] = []
            self.label_to_indices[label.item()].append(index)

    def __len__(self) -> int:
       return (self.num_positive_samples + 1) * len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        # Shuffle the indices.
        shuffled_indices = torch.randperm(len(self.dataset)).tolist()

        final_indices = []

        # For every index, insert a index with a matching label.
        for i in range(0, len(shuffled_indices)):
            final_indices.append(shuffled_indices[i])

            # Get the label of the current image.
            _, _, label = self.dataset[shuffled_indices[i]] # Unpack image, mask, and label; use only label

            # Get the list of indices with the same label.
            label_indices = self.label_to_indices[label.item()]

            # Sample self.num_positive_samples random indices from the list of indices with the same label.
            for _ in range(self.num_positive_samples):
                random_index = torch.randint(0, len(label_indices), (1,)).item()
                final_indices.append(label_indices[random_index])

        return iter(final_indices)
