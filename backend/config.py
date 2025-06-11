import os
import sys
import torch

from peft import LoraConfig
from torchvision.transforms import v2

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from backend.learning.evaluator.evaluators import hits_at_k
from backend.learning.loss.triplet import create_triplet_loss
from backend.models.mlp import MLP
from backend.models.vit_backbone import ViTBackbone

AUGMENTATION_CONSTRUCTORS = {
    "RandomAffine": lambda **kwargs: v2.RandomAffine(**kwargs),
    "RandomErasing": lambda **kwargs: v2.RandomErasing(**kwargs),
    "RandomHorizontalFlip": lambda **kwargs: v2.RandomHorizontalFlip(**kwargs),
    "RandomPerspective": lambda **kwargs: v2.RandomPerspective(**kwargs),
    "RandomResizedCrop": lambda **kwargs: v2.RandomResizedCrop(**kwargs),
    "RandomRotation": lambda **kwargs: v2.RandomRotation(**kwargs),
    "ColorJitter": lambda **kwargs: v2.ColorJitter(**kwargs),
}

AUGMENTATIONS = [
        ("ColorJitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}),
        ("RandomHorizontalFlip", {"p": 0.5}),
        ("RandomPerspective", {"distortion_scale": 0.33, "p": 0.75}),
        ("RandomRotation", {"degrees": 30}),
        ("RandomErasing", {"p": 0.5, "scale": (0.02, 0.25)})
]

EXPERIMENTAL_AUGMENTATIONS = [
    [
        ("ColorJitter", {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05}),
        ("RandomHorizontalFlip", {"p": 0.5}),
        ("RandomPerspective", {"distortion_scale": 0.33, "p": 1.0}),
        ("RandomRotation", {"degrees": 30}),
        ("RandomErasing", {"p": 0.75, "scale": (0.02, 0.25)})
    ]
]

BACKBONE_CONFIG = {
    "ViTModel": {
        "pretrained_model_name_or_path": "google/vit-large-patch16-384",
        "output_dim": 1024
    },
}

BACKBONE_CONSTRUCTORS = {
    "ViTModel": ViTBackbone,
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["query", "key", "value", "dense"]
}

LORA_CONFIG_CONSTRUCTOR = LoraConfig

EPOCHS = 100 # Usually 100, less for testing.

CONFIGS = {
    "vit_config_1": {
        "training": {
            "batch_size": 10, # Originally 10
            "gradient_accumulation_steps": 4, # Set to > 1 to enable accumulation
            "epochs": EPOCHS, 
            "evaluator": hits_at_k,
            "loss": {
                "name": "triplet",
                "constructor": create_triplet_loss,
                "options": {
                },
            },
            "freeze_backbone": False,
            "min_augmentations": 1, # Testing two augmentations, originally 1
            "max_augmentations": 1,
            "rebalance_augmentations": False,
            "optimizer": {
                "name": "AdamW",
                "constructor": torch.optim.AdamW,
                "learning_rate": 3.25e-5,  # lr = 0.0005 * batchsize/256, for 10: 1.25e-5
                "weight_decay": 0.5
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "constructor": torch.optim.lr_scheduler.CosineAnnealingLR,
                "options": {
                    "T_max": EPOCHS,  # Should be equal to the number of epochs.
                }
            },
            "num_positive_samples": 2
        },
        "model": {
            "backbone": {
                "name": "ViTModel",
                "constructor": ViTBackbone,
                "options": {
                    "pretrained_model_name_or_path": "google/vit-large-patch16-384",
                    "output_dim": 1024,
                    "in_channels": 4 # Specify 3 or 4 input channels for mask or no mask.
                }
            },
            "projection_head": {
                "name": "MLP",
                "constructor": MLP,
                "options": {
                    "hidden_dim": 256,
                    "output_dim": 1024,
                    "num_layers": 1,
                    "dropout_rate": 0.3
                }
            }
        }
    },
}

# Temporarily adding this back because shark matcher depends on it
PROJECTION_HEAD_CONFIG = {
    "MLP": {
        "hidden_dim": 4096,
        "output_dim": 2048,
        "num_layers": 2,
        "dropout_rate": 0.2
    },
    "MLP_1": {
        "hidden_dim": 4096,
        "output_dim": 2048,
        "num_layers": 2,
        "dropout_rate": 0.5
    },
    "MLP_2": {
        "hidden_dim": 4096,
        "output_dim": 2048,
        "num_layers": 2,
        "dropout_rate": 0.8
    },
    "MLP_3": {
        "hidden_dim": 1024,
        "output_dim": 1024,
        "num_layers": 3,
        "dropout_rate": 0.0
    }
}
PROJECTION_HEAD_CONSTRUCTORS = {
    "MLP": MLP,
    "MLP_1": MLP,
    "MLP_2": MLP,
    "MLP_3": MLP,
}
