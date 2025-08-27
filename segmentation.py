import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# ----------------------------
# Load ECG CSV
# ----------------------------
input_csv = '/content/drive/MyDrive/Colab Notebooks/archive (1)/ECG_Original_and_Smoothed.csv'
df = pd.read_csv(input_csv)

# Ensure ECG_signal column exists
if 'ECG_signal' not in df.columns:
    raise ValueError("Column 'ECG_signal' not found in CSV!")

# ----------------------------
# Step 1: Keep original ECG_signal intact
# ----------------------------
df['ECG_Original'] = df['ECG_signal']

# Convert numeric (without altering original)
ecg_numeric = pd.to_numeric(df['ECG_signal'], errors='coerce')
ecg_numeric = ecg_numeric.interpolate(method='linear').ffill().bfill()
# ----------------------------

window_length = 11
polyorder = 3

# ----------------------------
# Step 3: Save CSV
# ----------------------------
output_csv = '/content/drive/MyDrive/Colab Notebooks/archive (1)/ECG_Segmentation_Rpeaks.csv'
df.to_csv(output_csv, index=False)
print(f"Segmentation → Sliding Window + R-Peak detection saved at: {output_csv}")
df