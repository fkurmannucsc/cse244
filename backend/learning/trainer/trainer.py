"""
Code for training an encoder model using contrastive learning.
"""
import numpy as np
import os
import sys
import torch
# torch.autograd.set_detect_anomaly(True)
import tqdm
import torchvision.utils
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

from backend.constants import COMPUTE_PERIOD
from backend.constants import COMPUTE_PERIOD_MIN
from repo_utils import write_csv_file

class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, device: torch.device, config: dict = None):
        optimizer = config["training"]["optimizer"]["constructor"](
            filter(lambda p: p.requires_grad, model.parameters()), # Pass only trainable parameters
            lr=config["training"]["optimizer"]["learning_rate"],
            weight_decay=config["training"]["optimizer"]["weight_decay"]
        )
        scheduler = config["training"]["scheduler"]["constructor"](
            optimizer, **config["training"]["scheduler"]["options"]
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = config["training"]["loss"]["constructor"](**config["training"]["loss"]["options"])
        self.evaluate = config["training"]["evaluator"]
        self.device = device
        self.config = config # Store the full config

        # Mixed Precision Training (AMP) setup, TODO needs debugging.
        self.use_amp = False # self.device.type == 'cuda'
        # Ensure GradScaler is initialized correctly for the device type
        # self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)

    def train(self, train_loader: DataLoader, epochs: int, train_dataset: Dataset = None, valid_dataset: Dataset = None,
              compute_period_min=COMPUTE_PERIOD_MIN, compute_period=COMPUTE_PERIOD,
              trained_models_dir: str = None, timestamp: str = None):
        
        # Optional gradient accumulation.
        accumulation_steps = self.config["training"].get("gradient_accumulation_steps", 1)
        if accumulation_steps > 1:
            print(f"Using gradient accumulation with {accumulation_steps} steps.")

        training_history = []
        best_rep_metric = 0
        evaluation_dict = {}

        for epoch_counter in range(epochs):
            self.model.train() # Set model to training mode
            self.optimizer.zero_grad() # Zero gradients at the beginning of each epoch

            epoch_total_loss_sum_samples = 0.0 # Sum of individual sample losses for the epoch
            epoch_total_samples_processed = 0  # Total samples processed in the epoch

            with tqdm.tqdm(train_loader) as tq:
                tq.set_description("Epoch {}".format(epoch_counter))
                for step, (images, masks, labels) in enumerate(tq): # Unpack batch into images, masks, labels
                    # Handle batches from ContrastiveLearningSharkfinDataset where 'images' and 'labels'
                    # are lists of lists (list of augmentations per sample in batch).
                    if isinstance(images, list) and images and isinstance(images[0], list):
                        # Flatten the list of lists for images and masks
                        images_flat = [tensor for sublist in images for tensor in sublist]
                        images = torch.stack(images_flat, dim=0)
                        
                        # Flatten masks similarly
                        masks_flat = [tensor for sublist in masks for tensor in sublist if tensor is not None] # Filter out None masks
                        masks = torch.cat(masks_flat, dim=0) if masks_flat else None # Concatenate only if masks exist

                        # Flatten labels
                        labels_flat = [tensor for sublist in labels for tensor in sublist]
                        labels = torch.stack(labels_flat, dim=0) # Stack 0-D tensors to form a 1-D tensor
                    elif isinstance(images, list): # Fallback for simpler list of tensors (less likely with current setup)
                        images = torch.stack(images, dim=0) # This line might be hit if dataset __getitem__ changes
                        labels = torch.stack(labels, dim=0) # Stack 0-D tensors if labels is a flat list

                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    if masks is not None:
                        masks = masks.to(self.device)
                        # Ensure masks tensor has a channel dimension if it's [B, H, W]
                        if masks.ndim == 3: # Check if it's [B, H, W]
                            masks = masks.unsqueeze(1) # Reshape to [B, 1, H, W]


                    # # --- Temporary Visualization Code ---
                    # # This will display the RGB channels of the current batch of images.
                    # # Close the plot window to continue training.
                    # # Remember to remove or comment out these lines after debugging.
                    # print("[DEBUG] Displaying current batch of images (RGB channels)... Close plot to continue.")
                    # with torch.no_grad(): # Ensure no gradients are computed for visualization
                    #     images_for_display = images[:, :3, :, :].cpu() # Select RGB, move to CPU
                    #     grid = torchvision.utils.make_grid(images_for_display, nrow=int(images_for_display.size(0)**0.5))
                    #     plt.figure(figsize=(12, 12))
                    #     plt.imshow(grid.permute(1, 2, 0))
                    #     plt.title("Current Batch (RGB channels) - Before Model Forward Pass")
                    #     plt.axis('off')
                    #     plt.show() # This will block execution until the window is closed

                    # Mixed precision TODO needs debugging.
                    # Forward pass (autocast disabled for now)
                    # with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    
                    # --- NaN/Inf check for embeddings (moved earlier for per-batch check) ---
                    # These checks are helpful if NaN issues reappear.
                    # if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                    #     print(f"!!! ALERT: NaN or Inf detected in embeddings at epoch {epoch_counter}, step {step}.")
                    # --- NaN/Inf check for loss ---
                    # if torch.isnan(unscaled_loss).any() or torch.isinf(unscaled_loss).any():
                    #     print(f"!!! ALERT: NaN or Inf detected in unscaled_loss at epoch {epoch_counter}, step {step}.")

                    # Forward pass.
                    embedding = self.model(images, x_mask=masks)
                    unscaled_loss = self.loss(embedding, labels)
                    
                    # Accumulate total loss and samples for accurate epoch average
                    epoch_total_loss_sum_samples += unscaled_loss.item() * images.size(0) # loss.item() is avg loss for batch
                    epoch_total_samples_processed += images.size(0)

                    # Scale loss for gradient accumulation
                    if accumulation_steps > 1:
                        scaled_loss = unscaled_loss / accumulation_steps
                    else:
                        scaled_loss = unscaled_loss

                    # Backward pass on scaled loss.
                    scaled_loss.backward()

                    # Perform optimizer step and zero gradients only after `accumulation_steps`
                    # or on the last step of the epoch.
                    if ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(train_loader)):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad() # Zero gradients for the next accumulation cycle

                    # Update progress bar with average loss per sample so far in this epoch
                    if epoch_total_samples_processed > 0:
                        tq.set_postfix(loss=(epoch_total_loss_sum_samples / epoch_total_samples_processed))
                    else:
                        tq.set_postfix(loss=0.0)

            if (epoch_counter >= compute_period_min) and (epoch_counter % compute_period == 0):
                # Evaluation is done after the model parameters have been updated for the epoch.
                # self.model.eval() # Ensure model is in eval mode for evaluation
                rep_metric, evaluation_dict = self.evaluate(self.model, train_dataset, valid_dataset, self.device)
                # self.model.train() # Switch back to train mode if evaluate doesn't do it

                print(f"Epoch {epoch_counter}: {evaluation_dict}")

                best_rep_metric = max(best_rep_metric, rep_metric)

                if rep_metric == best_rep_metric:
                    self.save_model(trained_models_dir, timestamp)

            # Calculate average loss per sample for the entire epoch for history
            final_epoch_avg_loss_per_sample = 0.0
            if epoch_total_samples_processed > 0:
                final_epoch_avg_loss_per_sample = epoch_total_loss_sum_samples / epoch_total_samples_processed

            training_history.append(
                [
                    epoch_counter,
                    final_epoch_avg_loss_per_sample,
                    *(evaluation_dict.values() if evaluation_dict else [float('nan')] * (len(training_history[0])-2 if training_history else 0) ) # Handle case where eval_dict might be empty on first epoch if not computed
                ]
            )

            os.makedirs(os.path.join(trained_models_dir, timestamp), exist_ok=True)
            write_csv_file(os.path.join(trained_models_dir, timestamp, "training_history.csv"), training_history)

            self.scheduler.step()

    @abstractmethod
    def save_model(self, trained_models_dir: str, timestamp: str):
        pass
