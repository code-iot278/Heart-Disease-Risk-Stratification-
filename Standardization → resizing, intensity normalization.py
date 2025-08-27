import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
# ----------------------------
main_input_folder = "/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/segmented_results"

# Output folder
output_folder = "/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/Resize+Normalize1"
os.makedirs(output_folder, exist_ok=True)

# Find all PNG/JPG images recursively in subfolders
image_paths = glob.glob(os.path.join(main_input_folder, "**/*.png"), recursive=True) + \
              glob.glob(os.path.join(main_input_folder, "**/*.jpg"), recursive=True) + \
              glob.glob(os.path.join(main_input_folder, "**/*.jpeg"), recursive=True)

print(f"Found {len(image_paths)} images in main folder and subfolders.")

def preprocess_image(img, target_size=(224, 224)):
    """Resize + Normalize (0–1 + z-score)."""
    # Resize
    img_resized = cv2.resize(img, target_size)

    # Convert to float
    img_resized = img_resized.astype(np.float32) / 255.0

    # Z-score normalization
    mean, std = img_resized.mean(), img_resized.std()
    img_standardized = (img_resized - mean) / (std + 1e-8)

    return img_resized, img_standardized

# Toggle display
show_examples = True

# Walk through all subfolders
for root, dirs, files in os.walk(main_input_folder):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):   # read all image formats
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img is None:
                continue  # skip unreadable files

            # Preprocess
            resized_img, standardized_img = preprocess_image(img)

            # Save standardized image (scaled back to 0–255 for saving)
            out_img = ((standardized_img - standardized_img.min()) /
                       (standardized_img.max() - standardized_img.min()) * 255).astype(np.uint8)

            # Keep same subfolder structure in output
            rel_path = os.path.relpath(root, main_input_folder)
            save_dir = os.path.join(output_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            cv2.imwrite(os.path.join(save_dir, filename), out_img)

            # Show side-by-side example
            if show_examples:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax[0].set_title("Input Image")
                ax[0].axis("off")

                ax[1].imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
                ax[1].set_title("Standardized Output")
                ax[1].axis("off")

                plt.show()
