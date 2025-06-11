import math
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoImageProcessor
from transformers import ViTConfig, ViTModel
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTAttention, ViTLayer, ViTEncoder, ViTSelfOutput, ViTIntermediate, ViTOutput


class CustomViTSelfAttention(ViTSelfAttention): # Changed inheritance from ViTAttention
    def __init__(self, config):
        super().__init__(config)
        # self.dropout_prob = config.dropout

    # Modified forward to accept patch_level_attention_mask
    def forward(self, hidden_states, head_mask=None, output_attentions=False, patch_level_attention_mask=None):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if patch_level_attention_mask is not None:
            # Debug: Check mask shape and device before applying
            print(f"[DEBUG] CustomViTSelfAttention: patch_level_attention_mask shape: {patch_level_attention_mask.shape}, device: {patch_level_attention_mask.device}")
            print(f"[DEBUG] CustomViTSelfAttention: attention_scores shape: {attention_scores.shape}, device: {attention_scores.device}")

            # patch_level_attention_mask is [batch_size, sequence_length (num_patches + 1)]
            # It's 1 for foreground/CLS, 0 for background.
            # We want to add a large negative number to background scores.
            # The mask should apply to the 'key' dimension (columns of QK^T)
            # So, we expand it to [batch_size, 1, 1, sequence_length]
            attention_mask_value = -torch.finfo(attention_scores.dtype).max
            # Ensure mask is on the same device as attention_scores
            additive_mask = patch_level_attention_mask.to(attention_scores.device).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, SeqLen]
            # Convert 0s (background) to large_negative, 1s (foreground) to 0
            # Debug: Check additive_mask shape and values
            print(f"[DEBUG] CustomViTSelfAttention: additive_mask shape: {additive_mask.shape}")
            additive_mask = (1.0 - additive_mask) * attention_mask_value
            attention_scores = attention_scores + additive_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # attention_probs = self.dropout_prob(attention_probs) # Switched to dropout_prob

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class CustomViTAttention(ViTAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CustomViTSelfAttention(config)
        self.output = ViTSelfOutput(config) # Standard output layer

    def forward(self, hidden_states, head_mask=None, output_attentions=False, patch_level_attention_mask=None):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions, patch_level_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states) # self_outputs[0] is context_layer
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class CustomViTLayer(ViTLayer):
    def __init__(self, config):
        super().__init__(config) # Initializes intermediate, output, layernorms
        self.attention = CustomViTAttention(config) # Replace with custom attention

    def forward(self, hidden_states, head_mask=None, output_attentions=False, patch_level_attention_mask=None):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is before self-attention
            head_mask,
            output_attentions=output_attentions,
            patch_level_attention_mask=patch_level_attention_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + outputs
        return outputs

class CustomViTEncoder(ViTEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CustomViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True, patch_level_attention_mask=None):
        # This forward method is mostly a copy from ViTEncoder, just passing patch_level_attention_mask
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, patch_level_attention_mask)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        # Use the base class's return type for compatibility if needed, or a custom one
        # For simplicity, returning a tuple similar to non-dict mode
        # Or, if ViTModel expects BaseModelOutput, construct that.
        # Let's return what ViTModel's forward pass would typically get from its encoder.
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class CustomViTModel(ViTModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.encoder = CustomViTEncoder(config)

    def forward(
        self,
        pixel_values=None,
        bool_masked_pos=None, # Not used by standard ViT
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding: bool = False, # Keep this arg
        return_dict=None,
        patch_level_attention_mask=None, # New argument
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_level_attention_mask=patch_level_attention_mask, # Pass the mask here
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # Use the base class's return type for compatibility
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class ViTBackbone(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        pretrained_model_name = kwargs["pretrained_model_name_or_path"]
        self.num_channels = 3

        # Get config from a standard ViTModel to initialize CustomViTModel
        vit_config = ViTConfig.from_pretrained(pretrained_model_name)

        print("ViT Config: ", vit_config)

        # Instantiate CustomViTModel directly.
        # CustomViTModel's __init__ already sets up CustomViTEncoder.
        # add_pooling_layer=False is to get the sequence of hidden states, not just pooled output.
        self.backbone = CustomViTModel(vit_config, add_pooling_layer=False)

        # Load pretrained weights from a standard ViTModel into our CustomViTModel.
        # The custom classes (CustomViTEncoder, CustomViTLayer, CustomViTAttention, CustomViTSelfAttention)
        # are designed to be name-compatible for weight loading from their Hugging Face counterparts.
        print(f"Loading pretrained weights from {pretrained_model_name} into CustomViTModel.")
        standard_model_for_weights = ViTModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)
        missing_keys, unexpected_keys = self.backbone.load_state_dict(standard_model_for_weights.state_dict(), strict=False)

        if missing_keys:
            print(f"[WARN] ViTBackbone init: Missing keys when loading state_dict into CustomViTModel: {missing_keys}")
        if unexpected_keys:
            print(f"[WARN] ViTBackbone init: Unexpected keys when loading state_dict into CustomViTModel: {unexpected_keys}")

        # Set and validate output_dim. It's expected to match the backbone's natural hidden size.
        self.output_dim = kwargs["output_dim"]
        if self.output_dim != self.backbone.config.hidden_size:
            raise ValueError(
                f"output_dim ({self.output_dim}) in kwargs does not match "
                f"the backbone's hidden_size ({self.backbone.config.hidden_size}). "
                "Ensure these are consistent in your configuration."
            )

        # Standard image processor for 3-channel input
        self.image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name,
            do_rescale=False # Consistent with original behavior
        )

    def forward(self, x, x_mask=None):
        # x: image tensor [B, 3, H_orig, W_orig]
        # x_mask: binary foreground mask [B, 1, H_orig, W_orig] (1 for FG, 0 for BG)
        #         Assumed to have the same H_orig, W_orig as x.

        # Debug: Check input shapes and devices
        # print(f"[DEBUG] ViTBackbone.forward: x shape: {x.shape}, device: {x.device}")
        if x_mask is not None: print(f"[DEBUG] ViTBackbone.forward: x_mask shape: {x_mask.shape}, device: {x_mask.device}")

        # The image_processor handles resizing, normalization (if applicable), and tensor conversion.
        pixel_values = self.image_processor(
            x,
            return_tensors="pt",
            input_data_format="channels_first"  # Assumes x is [B, C, H, W] or list of such
        )["pixel_values"]
        pixel_values = pixel_values.to(self.backbone.device) # Ensure tensor is on the same device as the model

        patch_level_attention_mask_for_model = None
        if x_mask is not None:
            # Resize x_mask to match the ViT's processed image dimensions (H_vit, W_vit)
            # H_vit, W_vit are dimensions of pixel_values after image_processor (e.g., 384x384)
            h_vit, w_vit = pixel_values.shape[-2:]
            # Debug: Check target resize dimensions
            print(f"[DEBUG] ViTBackbone.forward: Target ViT size: ({h_vit}, {w_vit})")
            # Ensure x_mask is on the same device as x for interpolation
            x_mask = x_mask.to(pixel_values.device) # Ensure mask is on the same device as pixel_values
            
            # Ensure mask has a channel dimension [B, 1, H, W] before interpolation
            if x_mask.ndim == 3: # If shape is [B, H, W]
                 x_mask = x_mask.unsqueeze(1) # Add channel dimension -> [B, 1, H, W]

            print(f"[DEBUG ViTBackbone] x_mask.float() shape before interpolate: {x_mask.float().shape}") # Add this print
            resized_x_mask = F.interpolate(x_mask.float(), size=(h_vit, w_vit), mode='nearest') # Should now be [B, 1, H_vit, W_vit]
            # Debug: Check resized mask shape
            print(f"[DEBUG] ViTBackbone.forward: resized_x_mask shape: {resized_x_mask.shape}")

            patch_size = self.backbone.config.patch_size
            # Downsample the resized mask to patch resolution using max pooling
            # This ensures if any part of a patch is foreground, the patch is considered foreground.
            patch_mask_pooled = F.max_pool2d(resized_x_mask, kernel_size=patch_size, stride=patch_size) # [B, 1, num_patches_h, num_patches_w]
            patch_sequence_mask = patch_mask_pooled.flatten(2).squeeze(1) # [B, num_patches_total]
            # Debug: Check pooled and flattened mask shape
            print(f"[DEBUG] ViTBackbone.forward: patch_sequence_mask shape: {patch_sequence_mask.shape}")

            # Create the final mask including the CLS token
            # Add a mask for the CLS token (always allow attention to/from CLS token)
            cls_token_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=patch_sequence_mask.dtype)
            patch_level_attention_mask_for_model = torch.cat((cls_token_mask, patch_sequence_mask), dim=1) # [B, num_patches_total + 1]
            # This mask (1 for FG/CLS, 0 for BG) will be used by CustomViTSelfAttention

        # Pass the processed tensor to the backbone
        # ViTModel expects 'pixel_values' as the argument name for the input tensor
        # Debug: Check final mask shape before passing to model
        # if patch_level_attention_mask_for_model is not None: print(f"[DEBUG] ViTBackbone.forward: patch_level_attention_mask_for_model shape: {patch_level_attention_mask_for_model.shape}")
        outputs = self.backbone(
            pixel_values=pixel_values,
            patch_level_attention_mask=patch_level_attention_mask_for_model, # Pass to our custom model
            interpolate_pos_encoding=True # Often needed if input size differs from pretraining
        )
        
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