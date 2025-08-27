import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Non-Local Means Denoising Function
# ----------------------------
def denoise_image(img):
    if len(img.shape) == 2:
        return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

# ----------------------------
# Process Main Folder
# ----------------------------
def process_main_folder(input_main_folder, output_main_folder, show_examples=True):
    for root, dirs, files in os.walk(input_main_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                input_path = os.path.join(root, file)

                # Preserve folder structure
                relative_path = os.path.relpath(root, input_main_folder)
                output_folder = os.path.join(output_main_folder, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, file)

                # Read image
                img = cv2.imread(input_path)
                if img is None:
                    print(f"⚠️ Skipping unreadable file: {input_path}")
                    continue

                # Apply Non-Local Means Denoising
                denoised = denoise_image(img)

                # Save output
                cv2.imwrite(output_path, denoised)
                print(f"✅ Saved: {output_path}")

                # Always show side-by-side comparison
                if show_examples:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax[0].set_title("Input Image")
                    ax[0].axis("off")

                    ax[1].imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
                    ax[1].set_title("Denoised Image")
                    ax[1].axis("off")

                    plt.show()

# ----------------------------
# Example Usage
# ----------------------------
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/archive (2)"
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/denoised_images"

process_main_folder(input_main_folder, output_main_folder)
