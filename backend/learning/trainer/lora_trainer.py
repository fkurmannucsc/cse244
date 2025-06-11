"""
Fine-tune a model using LORA.
"""
import copy
import datetime
import json
import os
import sys

# Set PYTORCH_ENABLE_MPS_FALLBACK to allow unsupported MPS ops to run on CPU
# This is particularly for ops like _cdist_backward used in some loss functions.
# This MUST be done BEFORE importing torch for it to reliably take effect.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch # Now import torch

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("PYTORCH_ENABLE_MPS_FALLBACK set to 1 for MPS device.")
# torch.autograd.set_detect_anomaly(True)

from peft import get_peft_model
from peft import PeftModel
from typing import Optional
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

from backend.config import AUGMENTATION_CONSTRUCTORS
from backend.config import AUGMENTATIONS
from backend.config import CONFIGS
from backend.config import LORA_CONFIG
from backend.config import LORA_CONFIG_CONSTRUCTOR
from backend.constants import TRAIN_ANNOTATIONS_PATH
from backend.constants import VALID_ANNOTATIONS_PATH
from backend.constants import IMAGES_DIR
from backend.constants import TRAINED_MODELS_DIR
from backend.learning.contrastive_learning_sharkfin_dataset import ContrastiveLearningSharkfinDataset
from backend.learning.positive_match_sampler import PositiveMatchSampler
from backend.learning.trainer.trainer import Trainer
from backend.sharkfin_dataset import SharkfinDataset
from repo_utils import get_torch_device
from repo_utils import print_trainable_parameters
from repo_utils import seed_everything

TRAINED_MODELS_DIR = os.path.join(TRAINED_MODELS_DIR, "lora")

def custom_collate_fn(batch):
    # batch is a list of tuples: [(images_aug_list_sample1, masks_aug_list_sample1, labels_aug_list_sample1), ...]
    # Each _aug_list_sampleX is a list of augmented views for that sample.

    images_per_sample = [item[0] for item in batch]
    masks_per_sample = [item[1] for item in batch]
    labels_per_sample = [item[2] for item in batch]

    # The Trainer expects lists of lists, which it will then flatten.
    # The main purpose of this collate_fn is to handle the masks_per_sample
    # if it's a list of [None, None, ...] when all input masks were None.
    # The default_collate fails if it tries to stack a list containing None.

    # We don't need to stack here, just ensure the structure is consistent.
    # The ContrastiveLearningSharkfinDataset returns:
    # (list_of_augmented_images, list_of_augmented_masks, list_of_augmented_labels)
    # So, `batch` will be:
    # [
    #   ( [img1_aug1, img1_aug2], [mask1_aug1, mask1_aug2], [label1_aug1, label1_aug2] ), # sample 1
    #   ( [img2_aug1, img2_aug2], [mask2_aug1, mask2_aug2], [label2_aug1, label2_aug2] )  # sample 2
    # ]
    # images_per_sample will be [ [img1_aug1, img1_aug2], [img2_aug1, img2_aug2] ]
    # masks_per_sample will be [ [mask1_aug1, mask1_aug2], [mask2_aug1, mask2_aug2] ]
    #   or if mask was None for sample 1: [ [None, None], [mask2_aug1, mask2_aug2] ]
    # labels_per_sample will be [ [label1_aug1, label1_aug2], [label2_aug1, label2_aug2] ]

    # The default collate function would try to make these into tensors.
    # Since the elements are already lists (of tensors or Nones), we can just return them.
    # The error occurs when default_collate tries to process the list of lists of masks
    # if one of the inner lists contains only Nones.

    return images_per_sample, masks_per_sample, labels_per_sample

class MaskedSequential(torch.nn.Module):
    """
    A sequential container that handles an optional x_mask argument for the first module (backbone).
    """
    def __init__(self, backbone_module: torch.nn.Module, projection_head_module: torch.nn.Module):
        super().__init__()
        self.backbone_module = backbone_module
        self.projection_head_module = projection_head_module

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass that allows passing x_mask to the backbone.
        Assumes the backbone_module's forward method can accept x_mask.
        """
        features = self.backbone_module(x, x_mask=x_mask)
        output = self.projection_head_module(features)
        return output
class LoRAFineTuning(Trainer):
    def __init__(self, model: torch.nn.Module,
                 backbone: torch.nn.Module, # Changed PeftModel to torch.nn.Module
                 projection_head: torch.nn.Module,
                 device: torch.device,
                 config: dict = None):
        super(LoRAFineTuning, self).__init__(model, device, config)

        self.backbone = backbone
        self.projection_head = projection_head

    def save_model(self, trained_models_dir: str, timestamp: str, prefix: str = "best"):
        os.makedirs(os.path.join(trained_models_dir, timestamp), exist_ok=True)

        if isinstance(self.backbone, PeftModel):
            self.backbone.save_pretrained(os.path.join(trained_models_dir, timestamp, f"{prefix}-backbone-lora"))
        else: # Backbone was frozen (not a PeftModel)
            torch.save(self.backbone.state_dict(), os.path.join(trained_models_dir, timestamp, f"{prefix}-backbone-state.pt"))
            
        torch.save(self.projection_head.state_dict(),
                   os.path.join(trained_models_dir, timestamp, f"{prefix}-projection-head.pt"))


def lora(trained_models_dir: str, timestamp: str, config: dict, freeze_backbone: bool = False):
    """
    Fine-tune a model using LORA.
    """
    seed_everything()
    device = get_torch_device()

    # Get the number of input channels from the config
    num_input_channels = config["model"]["backbone"]["options"].get("in_channels", 3) # Default to 3 if not specified
    print(f"Using {num_input_channels} input channels for datasets and backbone.")

    # Load the dataset.
    contrastive_train_dataset = ContrastiveLearningSharkfinDataset(
        SharkfinDataset(IMAGES_DIR, TRAIN_ANNOTATIONS_PATH, 
                        channels=num_input_channels, device=torch.device("cpu")),
        [(name, AUGMENTATION_CONSTRUCTORS[name], options) for name, options in AUGMENTATIONS], # Pass (name_str, constructor_func, options_dict) tuples
        config["training"]["min_augmentations"], config["training"]["max_augmentations"], config["training"]["rebalance_augmentations"]
    )

    print("Contrastive train dataset size:", len(contrastive_train_dataset))
    
    train_original_dataset = SharkfinDataset(IMAGES_DIR, TRAIN_ANNOTATIONS_PATH, 
                                             channels=num_input_channels, device=torch.device("cpu"))
    valid_original_dataset = SharkfinDataset(IMAGES_DIR, VALID_ANNOTATIONS_PATH, 
                                             channels=num_input_channels, device=torch.device("cpu"))
                                             
    print("Train/test set sizes:", len(train_original_dataset), len(valid_original_dataset))

    train_sampler = PositiveMatchSampler(train_original_dataset, num_positive_samples=config["training"]["num_positive_samples"])

    train_loader = torch.utils.data.DataLoader(contrastive_train_dataset,
                                               batch_size=config["training"]["batch_size"],
                                               shuffle=False,
                                               sampler=train_sampler,
                                               drop_last=False,
                                               collate_fn=custom_collate_fn) # Use custom collate function

    # Build the model.
    # First, create the base backbone instance.
    base_backbone_instance = config["model"]["backbone"]["constructor"](**config["model"]["backbone"]["options"]).to(device)

    # Either the backbone is frozen or trainable.
    if freeze_backbone:
        print("Backbone is FROZEN. Only the projection head will be trained.")
        for param in base_backbone_instance.parameters():
            param.requires_grad = False
        # The 'trainable_backbone' is the base instance itself, with frozen parameters.
        trainable_backbone = base_backbone_instance
    else:
        print("Backbone will be fine-tuned with LoRA.")
        # Wrap the backbone with PEFT for LoRA fine-tuning.
        trainable_backbone = get_peft_model(base_backbone_instance, LORA_CONFIG_CONSTRUCTOR(**LORA_CONFIG)).to(device)
    
    print_trainable_parameters(trainable_backbone)

    # Create the projection head.
    projection_head = config["model"]["projection_head"]["constructor"](
        input_dim=base_backbone_instance.get_output_dim(),
        **config["model"]["projection_head"]["options"]
    ).to(device)

    # Use MaskedSequential instead of torch.nn.Sequential to handle x_mask
    model = MaskedSequential(trainable_backbone, projection_head).to(device)

    # Directory for model.
    os.makedirs(os.path.join(trained_models_dir, timestamp), exist_ok=True)

    # Save the configuration.
    string_config = copy.deepcopy(config)
    def recursive_constructor_to_string(_config):
        for key in _config:
            if key in ["constructor", "evaluator"]:
                _config[key] = _config[key].__name__
            elif isinstance(_config[key], dict):
                recursive_constructor_to_string(_config[key])
    recursive_constructor_to_string(string_config)
    json.dump(string_config, open(os.path.join(trained_models_dir, timestamp, "config.json"), "w"),
              indent=4, sort_keys=True)

    # Initialize learning components.
    lora_trainer = LoRAFineTuning(model, trainable_backbone, projection_head, device, config)

    # Train the model.
    lora_trainer.train(train_loader, config["training"]["epochs"],
                              train_dataset=train_original_dataset, valid_dataset=valid_original_dataset,
                              trained_models_dir=trained_models_dir, timestamp=timestamp)

    # Save the trained model.
    lora_trainer.save_model(trained_models_dir, timestamp, "last")

def main():
    for config_key in CONFIGS:
        config = CONFIGS[config_key]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        freeze_backbone_option = config["training"].get("freeze_backbone", False) # Get option from config

        lora(TRAINED_MODELS_DIR, timestamp, config, freeze_backbone=freeze_backbone_option)

if __name__ == "__main__":
    main()
