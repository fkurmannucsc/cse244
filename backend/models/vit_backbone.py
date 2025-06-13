import torch

from transformers import AutoImageProcessor
from transformers import ViTModel

class ViTBackbone(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        pretrained_model_name = kwargs["pretrained_model_name_or_path"]
        self.num_channels = kwargs.get("in_channels", 3)  # Get in_channels from kwargs, default to 3

        # Load the pre-trained ViT model
        self.backbone = ViTModel.from_pretrained(
            pretrained_model_name,
            add_pooling_layer=False  # Kept from original, to get sequence of hidden states
        )

        # Set and validate output_dim. It's expected to match the backbone's natural hidden size.
        self.output_dim = kwargs["output_dim"]
        if self.output_dim != self.backbone.config.hidden_size:
            raise ValueError(
                f"output_dim in kwargs ({self.output_dim}) does not match "
                f"the backbone's hidden_size ({self.backbone.config.hidden_size}). "
                "Ensure these are consistent in your configuration."
            )

        if self.num_channels == 4:
            # Configure image processor for 4 channels (disable 3-channel specific normalization)
            self.image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name,
                do_rescale=False,    # Consistent with original behavior
                do_normalize=False   # Disable default 3-channel normalization
            )
            
            # Update model's config and patch embedding layer for 4 channels
            self.backbone.config.num_channels = 4
            self.backbone.embeddings.patch_embeddings.num_channels = 4

            # Modify the patch embedding layer to accept 4 input channels
            original_projection: torch.nn.Conv2d = self.backbone.embeddings.patch_embeddings.projection
            
            new_projection = torch.nn.Conv2d(
                in_channels=4,
                out_channels=original_projection.out_channels,
                kernel_size=original_projection.kernel_size,
                stride=original_projection.stride,
                padding=original_projection.padding,
                dilation=original_projection.dilation,
                groups=original_projection.groups,
                bias=(original_projection.bias is not None)
            )

            with torch.no_grad():
                # Copy weights for RGB channels
                new_projection.weight.data[:, :3, :, :] = original_projection.weight.data.clone()
                # Initialize weights for the 4th channel (e.g., mask)
                torch.nn.init.normal_(new_projection.weight.data[:, 3, :, :], mean=0.0, std=0.01)
                
                if original_projection.bias is not None:
                    new_projection.bias.data = original_projection.bias.data.clone()
            
            self.backbone.embeddings.patch_embeddings.projection = new_projection

        elif self.num_channels == 3:
            # Configure image processor for 3 channels (use default normalization)
            self.image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name,
                do_rescale=False  # Consistent with original behavior
                # do_normalize=True is typically the default for ViT processors for 3 channels
            )
            # ViTModel is already configured for 3 channels by default.
        else:
            raise ValueError(f"Unsupported number of channels: {self.num_channels}. Must be 3 or 4.")

    def forward(self, x):
        # The image_processor handles resizing, normalization (if applicable), and tensor conversion.
        # Explicitly provide input_data_format for robustness.
        pixel_values = self.image_processor(
            x,
            return_tensors="pt",
            input_data_format="channels_first"  # Assumes x is [B, C, H, W] or list of such
        )["pixel_values"]
        pixel_values = pixel_values.to(self.backbone.device) # Ensure tensor is on the same device as the model

        # Pass the processed tensor to the backbone
        # ViTModel expects 'pixel_values' as the argument name for the input tensor
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Extract the CLS token embedding (first token of the sequence)
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        cls_token_embedding = outputs.last_hidden_state[:, 0, :] # Shape: (batch_size, hidden_size)
        return torch.flatten(cls_token_embedding, start_dim=1, end_dim=-1)

    def get_output_dim(self):
        # Returns the output dimension specified in kwargs, validated against backbone's hidden_size.
        return self.output_dim

    def save_pretrained(self, path):
        # This saves the modified backbone, including the new input projection layer.
        self.backbone.save_pretrained(path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegates gradient_checkpointing_enable to the underlying Hugging Face backbone."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            # print(f"Gradient checkpointing enabled on underlying ViTModel with kwargs: {gradient_checkpointing_kwargs}")
        else:
            print("Warning: Underlying backbone in ViTBackbone does not support gradient_checkpointing_enable.")