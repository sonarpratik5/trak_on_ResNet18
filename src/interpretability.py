import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAMVisualizer:
    def __init__(self, model, target_layer):
        """
        model: The trained PyTorch model
        target_layer: The specific layer to visualize (usually the last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: (1, C, H, W) normalized tensor
        target_class: int or None. If None, uses the highest predicted class.
        """
        # Define target (class to explain)
        if target_class is None:
            targets = None # Auto-selects highest scoring class
        else:
            targets = [ClassifierOutputTarget(target_class)]

        # Generate raw grayscale CAM (values 0 to 1)
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]

    def overlay_cam(self, rgb_img, grayscale_cam, threshold=0.5):
        """
        rgb_img: (H, W, 3) numpy array, values 0-1
        grayscale_cam: (H, W) numpy array
        threshold: float (0-1). Hide heatmap values below this to reduce noise.
        """
        # Apply threshold to remove weak activations
        mask = grayscale_cam > threshold
        cleaned_cam = grayscale_cam * mask
        
        # Create heatmap overlay
        visualization = show_cam_on_image(rgb_img, cleaned_cam, use_rgb=True)
        return visualization

def get_last_conv_layer(net):
    """
    Helper to find the target layer for ResNet-style models.
    For ResNet, it's usually the last block of the last layer.
    """
    return net.layer4[-1]