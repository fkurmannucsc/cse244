import os
import sys
import numpy as np
import random
import torch

from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from backend.constants import IMAGES_DIR
from backend.constants import SEGMENTED_IMAGES_DIR
from backend.constants import IMAGE_HEIGHT
from backend.constants import IMAGE_WIDTH
from backend.constants import FULL_ANNOTATIONS_PATH
from backend.constants import TRAIN_ANNOTATIONS_PATH
from backend.constants import VALID_ANNOTATIONS_PATH
from repo_utils import load_csv_file
class SharkfinDataset(Dataset):
    def __init__(self,
                 images_path,
                 annotations_path,
                 image_resize=1.0,
                 channels=3,
                 device=torch.device('cpu')): # Device for labels, images will be loaded to CPU first.
        self.dataset = self
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.image_resize = image_resize
        self.channels = channels
        self.device = device

        print(f"Loading sharkfin dataset, {channels} channels.")

        # Lists to contain needed data.
        self.ids = []
        self.filenames = []
        self.labels = []

        # Store train and test ids for reference.
        train_annotations = load_csv_file(TRAIN_ANNOTATIONS_PATH, delimiter='\t')
        test_annotations = load_csv_file(VALID_ANNOTATIONS_PATH, delimiter='\t')
        self.train_ids = []
        self.test_ids = []
        for annotation in train_annotations:
            self.train_ids.append(int(annotation[0]))
        for annotation in test_annotations:
            self.test_ids.append(int(annotation[0]))

        # Load the ids from the selected annotations file, train, test, or full.
        annotations = load_csv_file(annotations_path, delimiter='\t')
        for annotation in annotations:
            if not os.path.exists(os.path.join(self.images_path, annotation[1])):
                print(f"Image {annotation[1]} does not exist.")
                continue
            self.ids.append(int(annotation[0]))
            self.filenames.append(annotation[1])
            self.labels.append(int(annotation[2]))

        print("Dataset size: ", len(self.ids))

        # Check ids and filenames lists are same length
        assert(len(self.ids) == len(self.filenames) == len(self.labels))

        # Store labels as tensor.
        # Keep labels on CPU initially, they will be moved to device by the Trainer.
        self.labels = torch.tensor(self.labels, dtype=torch.long, device=torch.device('cpu'))

        # Images will be loaded on-the-fly in __getitem__

    def load_images(self):
        target_h, target_w = self.image_height(), self.image_width()

        for index, filename in enumerate(self.filenames):
            try:
                # Load image
                image_path = os.path.join(self.images_path, filename)
                img_original = read_image(image_path, mode=ImageReadMode.RGB)
            except Exception as e_img:
                print(f"Error reading image {image_path}: {e_img}. Skipping this sample and using a placeholder.")
                # Create a placeholder (e.g., zeros) for the image data
                self.images[index] = torch.zeros((self.channels, target_h, target_w), dtype=torch.float32, device=self.device)
                continue

            # Define transformations
            # Conditional Resize (based on original image logic)
            current_h, current_w = img_original.shape[1], img_original.shape[2]
            if current_h > target_h or current_w > target_w:
                ratio = min(target_h / current_h, target_w / current_w)
                intermediate_size = (int(current_h * ratio), int(current_w * ratio))
                resize_img_op = v2.Resize(size=intermediate_size, antialias=True)
            else:
                resize_img_op = v2.Identity()

            crop_op = v2.CenterCrop(size=(target_h, target_w))
            to_float_op = v2.ToDtype(torch.float32, scale=True)

            # Apply transforms to image
            img_t = resize_img_op(img_original)
            img_t = crop_op(img_t)
            img_final = to_float_op(img_t)

            if self.channels == 4:
                # Load corresponding mask
                # Assumes mask has the same filename and is in SEGMENTED_IMAGES_DIR
                mask_filename = filename 
                mask_filename = mask_filename.replace('.jpg', '.png')
                mask_filename = mask_filename.replace('.jpeg', '.png')
                mask_path = os.path.join(SEGMENTED_IMAGES_DIR, mask_filename)
                msk_original = None
                if os.path.exists(mask_path):
                    try:
                        msk_original = read_image(mask_path, mode=ImageReadMode.GRAY)
                        if msk_original.ndim == 2: msk_original = msk_original.unsqueeze(0) # Ensure (1, H, W)
                        if msk_original.shape[0] != 1: msk_original = msk_original[0, :, :].unsqueeze(0) # Take first channel if multi-channel gray

                        # Ensure mask has same H, W as image's original dimensions for consistent transforms
                        if img_original.shape[1:] != msk_original.shape[1:]:
                            print(f"Warning: Image {filename} ({img_original.shape[1:]}) and mask ({msk_original.shape[1:]}) "
                                  f"have different original dimensions. Resizing mask to match image's original dimensions.")
                            msk_original = v2.Resize(size=img_original.shape[1:], 
                                                     interpolation=v2.InterpolationMode.NEAREST)(msk_original)
                    except Exception as e_mask:
                        print(f"Error reading or pre-processing mask {mask_path}: {e_mask}. Using a zero mask.")
                        msk_original = None # Fallback to zero mask
                else:
                    print(f"Warning: Mask {mask_path} not found. Using a zero mask.")
                    msk_original = None # Fallback to zero mask

                # Define resize op for mask (if needed, same logic as image)
                if current_h > target_h or current_w > target_w:
                    resize_msk_op = v2.Resize(size=intermediate_size, interpolation=v2.InterpolationMode.NEAREST)
                else:
                    resize_msk_op = v2.Identity()

                # Apply transforms to mask
                if msk_original is not None:
                    msk_t = resize_msk_op(msk_original)
                    msk_t = crop_op(msk_t)
                    msk_final = to_float_op(msk_t)
                else: # Create a zero mask of target size
                    msk_final = torch.zeros((1, target_h, target_w), dtype=torch.float32)

                # Concatenate image (3 channels) and mask (1 channel)
                # Ensure tensors are on CPU for concatenation if not already, then move to target device
                combined_data = torch.cat((img_final.cpu(), msk_final.cpu()), dim=0)
                self.images[index] = combined_data.to(self.device)

            elif self.channels == 3:
                self.images[index] = img_final.to(self.device)

            else:
                raise ValueError(f"Unsupported number of channels: {self.channels}. Must be 3 or 4.")

    # Image property accessors.
    def image_height(self):
        return int(IMAGE_HEIGHT * self.image_resize)

    def image_width(self):
        return int(IMAGE_WIDTH * self.image_resize)

    def get_ids(self):
        return self.ids

    def get_filenames(self):
        return self.filenames

    def get_labels(self):
        return self.labels
    
    def get_train_ids(self):
        return self.train_ids

    def get_test_ids(self):
        return self.test_ids

    # Cross property getters.
    def get_filename_by_id(self, id):
        if id not in self.ids:
            print(f"Image with id {id} not found.")
            return
        return self.filenames[self.ids.index(id)]

    def get_label_by_id(self, id):
        if id not in self.ids:
            print(f"Image with id {id} not found.")
            return
        return self.labels[self.ids.index(id)]

    def get_id_by_filename(self, filename):
        if filename not in self.filenames:
            print(f"Image with filename {filename} not found.")
            return
        return self.ids[self.filenames.index(filename)]

    def get_label_by_filename(self, filename):
        index = self.filenames.index(filename)
        return self.labels.tolist()[index]

    def get_ids_by_label(self, label):
        """
        Gets all ids for a particular label, or shark ID.
        """
        if label not in self.labels.tolist():
            print(f"Label with label {label} not found.")
        ids = []
        for index, item in enumerate(self.labels.tolist()):
            if item == label:
                ids.append(self.ids[index])
        return ids

    def get_filenames_by_label(self, label):
        """
        Gets all filenames for a particular label, or shark ID.
        """
        if label not in self.labels.tolist():
            print(f"Label with label {label} not found.")
        filenames = []
        for index, item in enumerate(self.labels.tolist()):
            if item == label:
                filenames.append(self.filenames[index])
        return filenames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, image_index):
        # Ensure index is an integer
        if not isinstance(image_index, int):
            raise TypeError(f"SharkfinDataset.__getitem__ expects an integer index, but got {type(image_index)}. "
                            "If you are trying to get all items, use a DataLoader or specific methods like get_labels().")

        filename = self.filenames[image_index]
        label = self.labels[image_index]
        target_h, target_w = self.image_height(), self.image_width()

        image_path = "Unknown" # Initialize for logging
        try: # Load image
            # Load image
            image_path = os.path.join(self.images_path, filename)
            img_original = read_image(image_path, mode=ImageReadMode.RGB) # Load to CPU
        except Exception as e_img:
            print(f"Error reading image {image_path}: {e_img}. Returning a placeholder.")
            # Create a placeholder (e.g., zeros) for the image data
            image = torch.zeros((self.channels, target_h, target_w), dtype=torch.float32)
            return image, label

        # Define transformations
        current_h, current_w = img_original.shape[1], img_original.shape[2]
        if current_h > target_h or current_w > target_w:
            ratio = min(target_h / current_h, target_w / current_w)
            intermediate_size = (int(current_h * ratio), int(current_w * ratio))
            resize_img_op = v2.Resize(size=intermediate_size, antialias=True)
        else:
            resize_img_op = v2.Identity()

        crop_op = v2.CenterCrop(size=(target_h, target_w))
        to_float_op = v2.ToDtype(torch.float32, scale=True)

        # Apply transforms to image
        img_t = resize_img_op(img_original)
        img_t = crop_op(img_t)
        img_final = to_float_op(img_t) # Stays on CPU

        if self.channels == 4:
            # Load and process mask
            mask_filename = filename.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(SEGMENTED_IMAGES_DIR, mask_filename)
            msk_original = None
            if os.path.exists(mask_path):
                try:
                    msk_original = read_image(mask_path, mode=ImageReadMode.GRAY) # Load to CPU
                    if msk_original.ndim == 2: msk_original = msk_original.unsqueeze(0)
                    if msk_original.shape[0] != 1: msk_original = msk_original[0, :, :].unsqueeze(0)

                    if img_original.shape[1:] != msk_original.shape[1:]:
                        # print(f"Warning: Image {filename} ({img_original.shape[1:]}) and mask ({msk_original.shape[1:]}) "
                        #       f"have different original dimensions. Resizing mask to match image's original dimensions.")
                        msk_original = v2.Resize(size=img_original.shape[1:],
                                                 interpolation=v2.InterpolationMode.NEAREST)(msk_original)
                except Exception as e_mask:
                    # print(f"Error reading or pre-processing mask {mask_path}: {e_mask}. Using a zero mask.")
                    msk_original = None
            else:
                # print(f"Warning: Mask {mask_path} not found. Using a zero mask.")
                msk_original = None

            if current_h > target_h or current_w > target_w:
                resize_msk_op = v2.Resize(size=intermediate_size, interpolation=v2.InterpolationMode.NEAREST)
            else:
                resize_msk_op = v2.Identity()

            if msk_original is not None:
                msk_t = resize_msk_op(msk_original)
                msk_t = crop_op(msk_t)
                msk_final = to_float_op(msk_t) # Stays on CPU
            else:
                msk_final = torch.zeros((1, target_h, target_w), dtype=torch.float32)

            # Return image, mask, and label separately
            return img_final, msk_final, label
        elif self.channels == 3:
            # Return image, None for mask, and label
            return img_final, None, label
        else:
            raise ValueError(f"Unsupported number of channels: {self.channels}. Must be 3 or 4.")

def main():
    """
    Main
    """
    test_dataset = SharkfinDataset(IMAGES_DIR, FULL_ANNOTATIONS_PATH, device=torch.device("cpu"))
    # print(len(test_dataset.get_train_ids()))
    # print(len(test_dataset.get_test_ids()))
    returned_item = test_dataset.__getitem__(1)
    print(returned_item)


if __name__ == "__main__":
    main()
