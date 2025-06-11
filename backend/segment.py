import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import sys
import requests
import tqdm

# Set PYTORCH_ENABLE_MPS_FALLBACK to allow unsupported MPS ops to run on CPU
# This is particularly for ops like _cdist_backward used in some loss functions.
# This MUST be done BEFORE importing torch for it to reliably take effect.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch # Now import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image, ImageEnhance
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from backend.constants import EMBEDDINGS_DIR
from backend.constants import IMAGES_DIR
from backend.constants import SEGMENTED_IMAGES_DIR
from backend.constants import POPPED_IMAGES_DIR
from backend.constants import FRONTEND_IMAGES_DIR
from backend.sharkfin_dataset import SharkfinDataset
from backend.constants import FULL_ANNOTATIONS_PATH

import repo_utils

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  plt.show()
  del mask
  gc.collect()

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def parse_annotations(source_dir, annotations_path, verbose=False):
    """
    Reads an annotations file and returns a list of image filenames from source_dir
    that are listed in the annotations file. If annotations_path is None or invalid,
    returns all image filenames from source_dir.
    """
    all_source_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if annotations_path:
        if not os.path.exists(annotations_path):
            print(f"ERROR: Annotations file not found at '{annotations_path}'. Processing all images in source_dir.")
            image_files_to_process = all_source_files
        else:
            if verbose: print(f"Loading filenames from annotations file: {annotations_path}")
            try:
                annotations_data = repo_utils.load_csv_file(annotations_path, delimiter='\t')
                # Assuming filename is in the second column (index 1)
                allowed_filenames = {row[1] for row in annotations_data if len(row) > 1}
                image_files_to_process = [f for f in all_source_files if f in allowed_filenames]
                if verbose: print(f"Found {len(allowed_filenames)} filenames in annotations. Will process {len(image_files_to_process)} matching images from {source_dir}.")
            except Exception as e:
                print(f"Error reading or parsing annotations file {annotations_path}: {e}. Processing all images in source_dir.")
                image_files_to_process = all_source_files
    else:
        image_files_to_process = all_source_files
    
    return image_files_to_process

def generate_sam_masks(source_dir=IMAGES_DIR, output_dir=SEGMENTED_IMAGES_DIR, annotations_path=None, use_center_prompt=False, verbose=False):
    """
    Generates SAM2 segmentation masks for images in the source directory
    and saves them to the output directory.
    Selects the largest mask as the primary subject.
    If annotations_path is provided, only processes images listed in that file.
    If use_center_prompt is True, a point prompt at the image center is used with SAM2ImagePredictor.
    """
    device = repo_utils.get_torch_device()

    # --- SAM2 Model Setup ---
    # IMPORTANT: Ensure these paths are correct for your SAM2 installation/checkpoints.
    # These might be relative to your project root or an environment variable.
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt" 
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # if not os.path.exists(sam2_checkpoint):
    #     print(f"ERROR: SAM2 checkpoint not found at '{os.path.abspath(sam2_checkpoint)}'. Please check the path.")
    #     return
    # if not os.path.exists(model_cfg):
    #     print(f"ERROR: SAM2 model config not found at '{os.path.abspath(model_cfg)}'. Please check the path.")
    #     return

    try:
        print(f"Loading SAM2 model from checkpoint: {sam2_checkpoint} and config: {model_cfg}")
        # Build the base SAM2 model
        sam_model_instance = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        
        if use_center_prompt:
            print("Using SAM2ImagePredictor with center prompts.")
            active_mask_producer = SAM2ImagePredictor(sam_model_instance)
        else:
            print("Using SAM2AutomaticMaskGenerator.")
            active_mask_producer = SAM2AutomaticMaskGenerator(sam_model_instance)
        print("SAM2 model loaded successfully.")
    except Exception as e:
        print(f"Error initializing SAM2 model: {e}. Cannot proceed with mask generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    image_files = parse_annotations(source_dir, annotations_path, verbose)
    
    print(f"Found {len(image_files)} images in {source_dir} to process.")

    for filename in tqdm.tqdm(image_files, desc="Generating SAM2 Masks"):
        image_path = os.path.join(source_dir, filename)
        
        # Save mask with .png extension to ensure it's handled as a grayscale image
        base_filename, _ = os.path.splitext(filename)
        output_mask_path = os.path.join(output_dir, base_filename + ".png")

        if os.path.exists(output_mask_path) and not verbose:
            continue # Skip if mask already exists and not in verbose/force mode

        # try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # SAM expects RGB

        if use_center_prompt:
            # Use SAM2ImagePredictor with a center point prompt
            active_mask_producer.set_image(image_rgb) # image_rgb is HWC uint8 NumPy
            h, w, _ = image_rgb.shape
            # Point prompt: center of the image (x,y)
            input_points = torch.tensor([[[w / 2, h / 2]]], device=device, dtype=torch.float32)
            # Labels for the points (1 indicates foreground)
            input_labels = torch.tensor([[1]], device=device, dtype=torch.int64)

            # multimask_output=True typically returns 3 masks
            # masks_torch: [B, N_masks, H, W], scores_torch: [B, N_masks]
            masks_torch, scores_torch, _ = active_mask_producer.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            print(masks_torch.shape, scores_torch.shape)

            # Convert to tensor if NumPy array is returned
            if isinstance(masks_torch, np.ndarray):
                masks_torch = torch.from_numpy(masks_torch).to(device)
            if isinstance(scores_torch, np.ndarray):
                scores_torch = torch.from_numpy(scores_torch).to(device)

            # Robust check for valid tensor outputs
            valid_masks = isinstance(masks_torch, torch.Tensor) and masks_torch.numel() > 0
            valid_scores = isinstance(scores_torch, torch.Tensor) and scores_torch.numel() > 0

            if not (valid_masks and valid_scores):
                if verbose: print(f"No masks generated by predictor for {filename} with center prompt. Saving a black mask.")
                h, w, _ = image_rgb.shape # Ensure h, w are defined if masks/scores are invalid
                final_mask_np = np.zeros((h, w), dtype=np.uint8)
            else:
                # scores_torch is (N_masks,), masks_torch is (N_masks, H, W)
                best_idx = torch.argmax(scores_torch).item()
                final_mask_torch = masks_torch[best_idx, :, :] # Select best mask (H, W tensor)
                final_mask_np = final_mask_torch.cpu().numpy().astype(np.uint8) * 255
        else: # Use SAM2AutomaticMaskGenerator (original behavior)
            masks_data = active_mask_producer.generate(image_rgb)
            if not masks_data:
                if verbose: print(f"No masks generated by automatic generator for {filename}. Saving a black mask.")
                h, w, _ = image_rgb.shape
                final_mask_np = np.zeros((h, w), dtype=np.uint8)
            else:
                best_mask_ann = sorted(masks_data, key=(lambda x: x['area']), reverse=True)[0]
                final_mask_np = best_mask_ann['segmentation'].astype(np.uint8) * 255 # Boolean to 0/255
        
        mask_pil = Image.fromarray(final_mask_np, mode='L') # 'L' for 8-bit grayscale
        mask_pil.save(output_mask_path)
        
        if verbose:
            print(f"Saved mask for {filename} to {output_mask_path}")

        # except Exception as e:
            # print(f"Error processing {filename}: {e}")
        # finally:
            # Attempt to free memory. Specific methods depend on SAM2's internals.
            # If SAM2 has a specific method to clear cached image data, call it here.
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache() # PyTorch 1.12+ for MPS
    
    print(f"Finished generating SAM2 masks. Output is in {output_dir}")

def image_pop(source_dir=IMAGES_DIR, output_dir=None, annotations_path=None, contrast_factor=1.5, verbose=False):
    """
    Processes images from the source directory, applies a contrast enhancement,
    and saves them to the output directory.
    If annotations_path is provided, only processes images listed in that file.
    """
    if output_dir is None:
        # Create a default output directory if none is provided
        output_dir = os.path.join(os.path.dirname(source_dir), "images_popped")
        print(f"Output directory not specified, using default: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    image_files_to_process = parse_annotations(source_dir, annotations_path, verbose)

    print(f"Found {len(image_files_to_process)} images to process.")

    for filename in tqdm.tqdm(image_files_to_process, desc="Applying Contrast Enhancement"):
        image_path = os.path.join(source_dir, filename)
        output_image_path = os.path.join(output_dir, filename) # Keep original filename and extension

        if os.path.exists(output_image_path) and not verbose:
            continue # Skip if image already exists and not in verbose/force mode

        try:
            img = Image.open(image_path).convert('RGB') # Ensure RGB
            
            enhancer = ImageEnhance.Contrast(img)
            img_contrasted = enhancer.enhance(contrast_factor)
            
            img_contrasted.save(output_image_path)

            if verbose:
                print(f"Saved contrast-enhanced image {filename} to {output_image_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
    print(f"Finished applying contrast enhancement. Output is in {output_dir}")

def main():
    """
    Main.
    """

    # Call the new function to generate SAM masks
    # Example: Process all images
    # generate_sam_masks(verbose=False, use_center_prompt=False) 
    # Example: Process only images listed in an annotations file
    # generate_sam_masks(annotations_path=FULL_ANNOTATIONS_PATH, verbose=False, use_center_prompt=False)
    # Example: Process with center prompt
    # generate_sam_masks(source_dir=POPPED_IMAGES_DIR, annotations_path=FULL_ANNOTATIONS_PATH, verbose=True, use_center_prompt=True)

    # Example: Apply contrast enhancement to images
    # image_pop(source_dir=IMAGES_DIR, annotations_path=FULL_ANNOTATIONS_PATH, contrast_factor=1.5, verbose=True)
    # image_pop(source_dir=IMAGES_DIR, annotations_path=FULL_ANNOTATIONS_PATH, contrast_factor=2.5, verbose=True)



    # SAM Experiments and image visualization.
    # device = repo_utils.get_torch_device()

    # image = cv2.imread('backend/AN15093004.jpeg')
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(image)

    # # raw_image = Image.open('backend/AN15093004.jpeg').convert("RGB")
    # # plt.figure(figsize=(20,20))
    # # plt.imshow(raw_image)
    # # plt.show() 

    # # SAM2
    # sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    # mask_generator = SAM2AutomaticMaskGenerator(sam2)
    # masks = mask_generator.generate(image)

    # # Show the output.
    # # print(len(masks))
    # # print(masks[0].keys())

    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show() 





    











    # ARCHIVE CODE BELOW --------------------------------------------------------------
    # SAM1
    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)

    # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(image)

    # TODO SAM2 HF DEBUG THIS
    # model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)

    # generator = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large").to(device)
    # with torch.inference_mode(), torch.autocast(device, dtype=torch.float32):
    #     masks = generator.generate(image)

    # SAM1 HF
    # generator = pipeline("mask-generation", model="facebook/sam-vit-huge", torch_dtype=torch.float32, device=device)
    # img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    # plt.imshow(raw_image)
    # outputs = generator(raw_image, points_per_batch=64)
    # masks = outputs["masks"]
    # show_masks_on_image(raw_image, masks)



	

if __name__ == "__main__":
	main()


"""
LOAD SAM2 from huggingface

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
    
"""