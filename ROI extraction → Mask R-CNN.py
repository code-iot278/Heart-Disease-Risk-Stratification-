import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
# ----------------------------
# Backbone (ResNet50 + FPN)
# ----------------------------
backbone = resnet_fpn_backbone('resnet50', pretrained=True)

# ----------------------------
# Anchor Generator for RPN
# ----------------------------
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# RoIAlign setup
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2
)

mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=14,
    sampling_ratio=2
)

# ----------------------------
# Build Mask R-CNN model
# ----------------------------
model = MaskRCNN(
    backbone,
    num_classes=2,   # background + 1 class (change as needed)
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    mask_roi_pool=mask_roi_pooler
)
# ----------------------------
main_input_folder = "/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/denoised_images"

# Output folder
output_folder = "/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/segmented_results"
os.makedirs(output_folder, exist_ok=True)

# Find all PNG/JPG images recursively in subfolders
image_paths = glob.glob(os.path.join(main_input_folder, "**/*.png"), recursive=True) + \
              glob.glob(os.path.join(main_input_folder, "**/*.jpg"), recursive=True) + \
              glob.glob(os.path.join(main_input_folder, "**/*.jpeg"), recursive=True)

print(f"Found {len(image_paths)} images in main folder and subfolders.")

# ----------------------------
# Loop through images
# ----------------------------
for idx, image_path in enumerate(image_paths):
    print(f"Processing {idx+1}/{len(image_paths)}: {image_path}")

    # Load safely
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:  # fallback to matplotlib
        image = mpimg.imread(image_path)
        if len(image.shape) == 3:
            image = cv2.cvtColor((image * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)

    # ----------------------------
    # Thresholding dark regions
    # ----------------------------
    _, mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)  # dark regions = white

    # ----------------------------
    # Segmentation overlay
    # ----------------------------
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (0, 255, 0)  # mark dark portion in green

    # ----------------------------
    # Save output image
    # ----------------------------
    # Preserve relative subfolder structure
    rel_path = os.path.relpath(image_path, main_input_folder)
    save_path = os.path.join(output_folder, rel_path)

    # Create subfolder if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save segmented overlay
    cv2.imwrite(save_path, overlay)

    # ----------------------------
    # Display results (optional)
    # ----------------------------
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(image, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Segmented")
    plt.axis("off")

    plt.show()

print("✅ All segmented images saved & displayed!")