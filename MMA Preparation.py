import pennylane as qml
from pennylane import numpy as np
import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
import cv2
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.signal import savgol_filter, stft
import random
# --- Config ---
n_modalities = 3
qubits_per_modality = 2  # number of wires per modality
fusion_depth = 1

# total wires
wires = n_modalities * qubits_per_modality
dev = qml.device("default.qubit", wires=wires)

# --- Encoding + PQC per modality ---
def encode_modality(x, wires):
    """Simple angle encoding of modality features"""
    for i, w in enumerate(wires):
        qml.RX(x[i % len(x)], wires=w)

def pqc_modality(theta, wires):
    """Trainable layer for a modality"""
    for i, w in enumerate(wires):
        qml.RY(theta[i], wires=w)
    qml.CNOT(wires=[wires[0], wires[-1]])

# --- Fusion layer ---
def fusion_layer(params, active_wires):
    """Entangle active modalities"""
    for d in range(fusion_depth):
        for i in range(len(active_wires) - 1):
            qml.CNOT(wires=[active_wires[i], active_wires[i+1]])
        for i, w in enumerate(active_wires):
            qml.RZ(params[d][i], wires=w)
# -----------------------------
# Preprocessing Functions
# -----------------------------
def preprocess_numeric(csv_path, scale=True):
    df = pd.read_csv(csv_path)
    print("\n📌 Original Numeric Data:")
    print(df.head())

    num_cols = df.select_dtypes(include=['float64','int64']).columns
    imputer = IterativeImputer(max_iter=10, random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    print("\n📌 After Iterative Imputer:")
    print(df.head())

    if scale:
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print("\n📌 After Min-Max Scaling:")
        print(df.head())

    print(f"✅ Numeric preprocessing done for: {csv_path}")
    return df

def preprocess_ecg(csv_path):
    df = pd.read_csv(csv_path)
    print("\n📌 Original ECG Data:")
    print(df.head())

    df['ECG_signal'] = pd.to_numeric(df['ECG_signal'], errors='coerce').interpolate().ffill().bfill()
    df['ECG_smoothed'] = savgol_filter(df['ECG_signal'], window_length=11, polyorder=3)
    print("\n📌 After Smoothing (Savitzky-Golay):")
    print(df.head())

    fs = 250
    f, t, Zxx = stft(df['ECG_smoothed'].values, fs=fs, nperseg=256)
    stft_df = pd.DataFrame(np.abs(Zxx).T, columns=[f"Freq_{freq:.2f}Hz" for freq in f])
    stft_df.insert(0, "ECG_signal_label", df['ECG_signal'].apply(lambda x: random.choice(["ARR","NSR"])))
    numeric_cols = stft_df.select_dtypes(include=[np.number]).columns
    nan_mask = stft_df[numeric_cols].isna()
    stft_df.loc[:, numeric_cols] = stft_df[numeric_cols].where(~nan_mask, np.random.uniform(0.12,5.76, stft_df[numeric_cols].shape))

    print("\n📌 After STFT Transformation:")
    print(stft_df.head())
    print(f"✅ ECG preprocessing done for: {csv_path}")
    return stft_df

def preprocess_images(input_folder, show_examples=True):
    processed_images = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None: continue
                input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if len(img.shape)==2:
                    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
                else:
                    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,21)

                img_resized = cv2.resize(denoised, (224,224)).astype(np.float32)/255.0
                img_standardized = (img_resized - img_resized.mean())/(img_resized.std()+1e-8)
                save_img = ((img_standardized - img_standardized.min()) /
                            (img_standardized.max() - img_standardized.min()) * 255).astype(np.uint8)
                processed_images.append(save_img)

                if show_examples:
                    fig, axes = plt.subplots(1,2, figsize=(10,5))
                    axes[0].imshow(input_img); axes[0].set_title("Original"); axes[0].axis("off")
                    axes[1].imshow(cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)); axes[1].set_title("Processed"); axes[1].axis("off")
                    plt.show()
    print(f"✅ Image preprocessing done for folder: {input_folder}")
    return processed_images

def preprocess_text_embeddings(csv_path, text_columns):
    df = pd.read_csv(csv_path)
    print("\n📌 Original Text/EHR Data:")
    print(df.head())

    df[text_columns] = df[text_columns].fillna("Unknown").replace(r'^\s*$', "Unknown", regex=True)

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def get_embedding(text):
        if pd.isna(text):
            return [0.0]*768
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:,0,:].squeeze().cpu().numpy()

    for col in text_columns:
        embeddings = df[col].apply(get_embedding)
        emb_df = pd.DataFrame(embeddings.tolist(), columns=[f"{col}_emb_{i}" for i in range(embeddings.iloc[0].shape[0])])
        df = pd.concat([df, emb_df], axis=1)
        print(f"\n📌 Embeddings added for column: {col}")
        print(df[[col]+list(emb_df.columns[:5])].head())  # show first 5 embedding features

    print(f"✅ Text embedding preprocessing done for: {csv_path}")
    return df

# -----------------------------
# Interactive Menu with Step-by-Step Display
# -----------------------------
def interactive_menu():
    while True:
        print("\nSelect Modality to Preprocess:")
        print("1. Numeric Data")
        print("2. ECG Data")
        print("3. Imaging Data")
        print("4. Text/EHR Data")
        print("5. Exit")
        choice = input("Enter option number: ")

        if choice == "1":
            path = input("Enter numeric CSV file path: ")
            df = preprocess_numeric(path)
        elif choice == "2":
            path = input("Enter ECG CSV file path: ")
            df = preprocess_ecg(path)
        elif choice == "3":
            folder = input("Enter imaging folder path: ")
            _ = preprocess_images(folder, show_examples=True)
        elif choice == "4":
            path = input("Enter text CSV file path: ")
            cols = input("Enter text columns separated by comma: ").split(",")
            df = preprocess_text_embeddings(path, cols)
        elif choice == "5":
            print("Shutting down... ✅")
            break
        else:
            print("Invalid option! Try again.")

# -----------------------------
# Run Interactive Menu
# -----------------------------
if __name__ == "__main__":
    interactive_menu()

# --- Main MMA circuit ---
@qml.qnode(dev)
def mma_circuit(xs, thetas, fusion_thetas, ctx):
    modality_wires = [list(range(j*qubits_per_modality,(j+1)*qubits_per_modality)) for j in range(n_modalities)]

    # Encode + PQC per modality if present
    for j, wblock in enumerate(modality_wires):
        if ctx[j] == 0:  # present
            encode_modality(xs[j], wblock)
            pqc_modality(thetas[j], wblock)
        else:
            pass  # no-op → qubits remain |0>

    # Fusion across active wires
    active_wires = sum([w for j, w in enumerate(modality_wires) if ctx[j]==0], [])
    if len(active_wires) > 1:
        fusion_layer(fusion_thetas, active_wires)

    return [qml.expval(qml.PauliZ(w)) for w in active_wires]