import numpy as np
import math
import os
import random # Import random for transform application
import sys
import torch
import torchvision

from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

from backend.sharkfin_dataset import SharkfinDataset

class ContrastiveLearningSharkfinDataset(Dataset):
    def __init__(self, dataset: SharkfinDataset, augmentations: list, 
                 min_image_augmentations: int=10, max_image_augmentations: int=10, 
                 rebalance_augmentations: bool=False):
        # print("Augmentations:", augmentations, min_image_augmentations, max_image_augmentations, rebalance_augmentations)
        
        # Build separate transform pipelines for image and mask
        image_transforms_list = []
        mask_transforms_list = []

        for transform_name, transform_constructor, transform_options in augmentations: # Unpack name, constructor, and options
            # transform_name is now the string like "ColorJitter"
            transform_instance = transform_constructor(**transform_options)
            
            # Apply spatial transforms to both image and mask
            if transform_name in ["RandomAffine", "RandomErasing", "RandomHorizontalFlip", 
                                  "RandomPerspective", "RandomResizedCrop", "RandomRotation"]:
                image_transforms_list.append(transform_instance)
                mask_transforms_list.append(transform_instance)
            # Apply color transforms only to the image
            elif transform_name in ["ColorJitter"]:
                 image_transforms_list.append(transform_instance)
            else:
                 print(f"Warning: Unknown augmentation '{transform_name}'. Skipping.")

        self.image_view_generator = ContrastiveLearningViewGenerator(
            torchvision.transforms.Compose(image_transforms_list)
        )
        self.label_transform = ContrastiveLearningViewGenerator(torchvision.transforms.Compose([]))
        self.dataset = dataset
        
        # Augmentation count values
        self.min_image_augmentations = min_image_augmentations
        self.max_image_augmentations = max_image_augmentations

        self.rebalance_augmentations = rebalance_augmentations

        self.labels = self.dataset.get_ids()
        label_counts = np.unique(self.dataset.get_labels().cpu().numpy(), return_counts=True)
        self.label_counts_dict = dict(zip(label_counts[0], label_counts[1]))
        self.max_count = max(self.label_counts_dict.values())

        # Create a separate view generator for masks, using only spatial transforms
        # This ensures masks are augmented identically to images spatially.
        self.mask_view_generator = ContrastiveLearningViewGenerator(
            torchvision.transforms.Compose(mask_transforms_list)
        )
    def __getitem__(self, index):
        if self.rebalance_augmentations:
            # print("Balancing augmentations")
            images_for_shark = self.label_counts_dict[self.dataset[index][1].item()]
            num_augmentations = min(math.floor(self.max_count / images_for_shark) * self.min_image_augmentations, self.max_image_augmentations)
            # print(f"Num augmentations {num_augmentations}")
        else:
            num_augmentations = self.min_image_augmentations

        # Get image, mask, and label from the base dataset
        image, mask, label = self.dataset[index]

        return (self.image_view_generator(image, num_augmentations),
                self.mask_view_generator(mask, num_augmentations), # Apply mask augmentations
                self.label_transform(label, num_augmentations)) # Label transform is identity

    def __len__(self):
        return len(self.dataset)

class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_transform: torchvision.transforms.Compose):
        self.base_transform = base_transform

    def __call__(self, x: torch.Tensor, num_augmentations: int):
        # Transforms can randomly produce nan values, so we need to retry until we get a valid tensor.
        # We also add retries for RuntimeErrors that might occur with incompatible transforms.
        view = None
        max_set_retries = 5  # Max attempts to generate the full set of augmentations
        set_retry_count = 0

        while view is None and set_retry_count < max_set_retries:
            try:
                candidate_views = []
                for _ in range(num_augmentations):
                    # Apply transform only if input is not None
                    if x is not None:
                        candidate_views.append(self.base_transform(x))
                    else:
                        candidate_views.append(None) # Keep None if input was None
                
                # Check for NaNs only if the input was not None
                if any(v is not None and torch.isnan(v).any() for v in candidate_views):
                    # print(f"NaN detected in one or more views, retrying set (attempt {set_retry_count + 1}/{max_set_retries})")
                    set_retry_count += 1
                    continue # Retry generating the whole set of views
                
                view = candidate_views # Successfully generated all views

            except RuntimeError as e:
                # This can happen if a transform is incompatible with the input (e.g., hue jitter on 4-channels)
                # print(f"RuntimeError during augmentation set generation: {e}. Retrying set (attempt {set_retry_count + 1}/{max_set_retries})")
                set_retry_count += 1
        
        # Fallback logic
        if view is None:
            # If all retries failed, fallback to using clones of the original tensor (or None)
            # This handles the case where x is None (for masks when channels=3) or augmentation failed.
            # print(f"Warning: Failed to generate a valid set of {num_augmentations} augmentations after {max_set_retries} attempts. Using original tensor/None for all views.")
            # print(f"Warning: Failed to generate a valid set of {num_augmentations} augmentations after {max_set_retries} attempts. Using original tensor for all views.")
            view = [x.clone() for _ in range(num_augmentations)]

        return view
