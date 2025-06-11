import argparse
import cv2
import numpy as np
import os
import sys
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from backend.constants import IMAGES_DIR
from backend.sharkfin_dataset import SharkfinDataset

import backend.retrieval.embeddings
import backend.utils
import repo_utils

class SimilarityToConceptTarget:
    """
    Aadapt pixel attribution methods for embedding networks (as apposed for just vanilla classification networks).
    There is a reference embedding, the concept, then for query embeddings, the question to answer will be:
    What in the image is similar or different than the concept embedding?
    """
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')

    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


class ResnetFeatureExtractor(torch.nn.Module):
    """
    A model wrapper that gets a resnet model and returns the features before the fully connected layer.
    """
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                
    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]
    
def reshape_transform(tensor, height=14, width=14):
    """
    Helper function to reshape activations and gradients to 2D images.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0),height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    
    device = repo_utils.get_torch_device()
    
    # Load the SharkfinDataset.
    print("Loading dataset.")
    dataset = SharkfinDataset(IMAGES_DIR, device=torch.device("cpu"))

    # TODO model name hard coded to LORA
    # Load the pretrained models.
    print("Loading model.")
    # projection_head, model_backbone = backend.utils.load_model("lora")
    # model = pro
    # target_layers = [model.blocks[-1].norm1]

    # Example with vit model:
    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True).to(torch.device(device)).eval()
    # target_layers = [model.blocks[-1].norm1]

    # Example with resent model:
    # resnet = resnet50(pretrained=True).to(device).eval()
    # model = ResnetFeatureExtractor(resnet)
    # target_layers = [resnet.layer4[-1]]
    
    # Load the images.
    # TODO temporarily hard code the image.
    image_path = os.path.join(IMAGES_DIR, "AN13101402.jpeg")
    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    concept_img = cv2.imread(image_path, 1)[:, :, ::-1]
    concept_img = cv2.resize(concept_img, (224, 224)) # Backbone
    # concept_img = cv2.resize(concept_img, (512, 512)) # Resnet
    # concept_img = cv2.resize(concept_img, (1024, 1024))
    concept_img = np.float32(concept_img) / 255
    concept_tensor = preprocess_image(concept_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]).to(device)
    # fin_img, fin_img_float, fin_tensor = get_image_from_url("https://www.wallpapersin4k.org/wp-content/uploads/2017/04/Foreign-Cars-Wallpapers-4.jpg")
    concept_features = model(concept_tensor)[0, :]

    image_path = os.path.join(IMAGES_DIR, "AN14012903.jpeg")
    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    fin_img = cv2.imread(image_path, 1)[:, :, ::-1]
    fin_img = cv2.resize(fin_img, (224, 224)) # Backbone
    # fin_img = cv2.resize(concept_img, (512, 512)) # Resnet
    # fin_img = cv2.resize(concept_img, (1024, 1024))
    fin_img = np.float32(fin_img) / 255
    fin_tensor = preprocess_image(fin_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(device)

    fin_targets = [SimilarityToConceptTarget(concept_features)]

    # Where is the car in the image
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) # reshape_transform=reshape_transform for ViT
    grayscale_cam = cam(input_tensor=fin_tensor, targets=fin_targets)[0, :]

    fin_cam_image = show_cam_on_image(fin_img, grayscale_cam, use_rgb=True)
    img = Image.fromarray(fin_cam_image)
    img.show()

    print("DONE!")


    # cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)


    # # Method selection.
    # if args.method not in methods:
    #     raise Exception(f"Method {args.method} not implemented")

    # if args.method == "ablationcam":
    #     cam = methods[args.method](model=model,
    #                                target_layers=target_layers,
    #                                reshape_transform=reshape_transform,
    #                                ablation_layer=AblationLayerVit())
    # else:
    #     cam = methods[args.method](model=model,
    #                                target_layers=target_layers,
    #                                reshape_transform=reshape_transform)

    # # TODO temporarily hard code the image.
    # image_path = os.path.join(IMAGES_DIR, "AN13101402.jpeg")
    # # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    # rgb_img = cv2.resize(rgb_img, (224, 224))
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
    #                                 std=[0.5, 0.5, 0.5]).to(args.device)

    # # If None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested category.
    # targets = None

    # # AblationCAM and ScoreCAM have batched implementations.
    # # You can override the internal batch size for faster computation.
    # cam.batch_size = 32

    # grayscale_cam = cam(input_tensor=input_tensor,
    #                     targets=targets,
    #                     eigen_smooth=args.eigen_smooth,
    #                     aug_smooth=args.aug_smooth)

    # # Here grayscale_cam has only one image in the batch
    # grayscale_cam = grayscale_cam[0, :]

    