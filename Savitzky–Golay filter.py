import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# ----------------------------
# Load ECG CSV
# ----------------------------
input_csv = '/content/drive/MyDrive/Colab Notebooks/archive (1)/ECGCvdata.csv'
df = pd.read_csv(input_csv)

# Ensure ECG_signal column exists
if 'ECG_signal' not in df.columns:
    raise ValueError("Column 'ECG_signal' not found in CSV!")

# ----------------------------
# Step 1: Keep original ECG_signal intact
# ----------------------------
# Copy original for display / saving
df['ECG_Original'] = df['ECG_signal']

# Convert numeric for smoothing (without altering original)
ecg_numeric = pd.to_numeric(df['ECG_signal'], errors='coerce')
ecg_numeric = ecg_numeric.interpolate(method='linear').ffill().bfill()

# ----------------------------
# Step 2: Apply Savitzky–Golay Filter to smoothed column
# ----------------------------
window_length = 11  # Must be odd
polyorder = 3       # Cubic polynomial
# ----------------------------
# Step 3: Save CSV
# ----------------------------
output_csv = '/content/drive/MyDrive/Colab Notebooks/archive (1)/ECG_Original_and_Smoothed.csv'
df.to_csv(output_csv, index=False)
print(f"Original and smoothed ECG saved at: {output_csv}")
df