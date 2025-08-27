import os
import numpy as np
import csv
from PIL import Image
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize

# -------- SETTINGS --------
MAIN_FOLDER = "/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/denoised_images"   # main input folder
IMG_SIZE = (4, 4)         # must be power-of-2 in total pixels (4x4 = 16 pixels → 4 qubits)
OUTPUT_FILE = "/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/qubit_dataset.csv"
# --------------------------

def load_and_preprocess_image(img_path, size=(4,4)):
    """Load image, resize, grayscale, normalize to [0,1]."""
    img = Image.open(img_path).convert("L")  # grayscale
    img = img.resize(size)
    arr = np.array(img).astype(np.float64)   # use float64 for precision
    arr = arr / 255.0  # normalize pixel values
    return arr.flatten()

def statevector_from_image(img_vector):
    """Normalize image vector for amplitude encoding (safe for Qiskit)."""
    norm = np.linalg.norm(img_vector)
    if norm == 0:
        raise ValueError("Image vector is zero after preprocessing")
    state = img_vector / norm
    # Extra renormalization to avoid floating point error
    state = state / np.linalg.norm(state)
    # Round to avoid tiny floating-point mismatch
    state = np.round(state, 12)
    return state

def image_to_qubits(img_vector):
    """Return quantum circuit + statevector from image."""
    state = statevector_from_image(img_vector)
    num_qubits = int(np.log2(len(state)))
    qc = QuantumCircuit(num_qubits)
    init_gate = Initialize(state)
    qc.append(init_gate, range(num_qubits))
    qc.barrier()
    return qc, state

# -------- MAIN LOOP --------
rows = []

for root, dirs, files in os.walk(MAIN_FOLDER):
    label = os.path.basename(root)  # use folder name as label
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(root, filename)
            img_vector = load_and_preprocess_image(path, IMG_SIZE)
            _, state = image_to_qubits(img_vector)

            # Prepare row: label + flattened statevector amplitudes
            row = [label] + state.tolist()
            rows.append(row)

            print(f"Processed: {path} -> {len(state)} amplitudes ({int(np.log2(len(state)))} qubits)")

# -------- SAVE TO CSV --------
with open(OUTPUT_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    # Write header
    header = ["label"] + [f"amp_{i}" for i in range(len(rows[0]) - 1)]
    writer.writerow(header)
    # Write data rows
    writer.writerows(rows)

print(f"\n✅ Saved dataset to {OUTPUT_FILE}")
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/•	cardiac imaging (echo)/qubit_dataset.csv')
df