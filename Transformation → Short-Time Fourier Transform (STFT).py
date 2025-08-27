import numpy as np
import pandas as pd
from scipy.signal import stft
import random

# ----------------------------
# Step 1: Load ECG CSV
# ----------------------------
input_csv = '/content/drive/MyDrive/Colab Notebooks/archive (1)/ECG_Segmentation_Rpeaks.csv'
df = pd.read_csv(input_csv)

# Ensure ECG_signal column exists
if 'ECG_signal' not in df.columns:
    raise ValueError("Column 'ECG_signal' not found in CSV!")

# ----------------------------
# Step 2: Convert ECG_signal to numeric and handle missing values
# ----------------------------
df['ECG_signal'] = pd.to_numeric(df['ECG_signal'], errors='coerce')
df['ECG_signal'] = df['ECG_signal'].interpolate(method='linear')
df['ECG_signal'] = df['ECG_signal'].bfill()
df['ECG_signal'] = df['ECG_signal'].ffill()

# ----------------------------
# Step 3: Compute STFT
# ----------------------------
fs = 250        # Sampling frequency (Hz)
nperseg = 256   # Segment length

frequencies, times, Zxx = stft(df['ECG_signal'].values, fs=fs, nperseg=nperseg)
Zxx_mag = np.abs(Zxx)

# ----------------------------
# Step 4: Convert STFT to DataFrame
# ----------------------------
stft_df = pd.DataFrame(Zxx_mag.T, columns=[f"Freq_{f:.2f}Hz" for f in frequencies])

# ----------------------------
# Step 5: Randomly assign ARR / NSR labels
# ----------------------------
def ecg_label(value):
    return random.choice(["ARR", "NSR"])

labels = df['ECG_signal'].apply(ecg_label)

# ----------------------------
# Step 6: Match STFT rows with original ECG length
# ----------------------------
if len(stft_df) < len(df):
    pad_rows = len(df) - len(stft_df)
    stft_df = pd.concat([stft_df,
                         pd.DataFrame(np.nan, index=range(pad_rows), columns=stft_df.columns)],
                        ignore_index=True)
elif len(stft_df) > len(df):
    stft_df = stft_df.iloc[:len(df)].reset_index(drop=True)

# Add ECG values + labels
stft_df.insert(0, "ECG_signal_label", labels.values)
stft_df.insert(0, "ECG_signal_value", df['ECG_signal'].values)

# ----------------------------
# Step 7: Replace NaNs in numeric columns only
# ----------------------------
min_val, max_val = 0.12, 5.76
numeric_cols = stft_df.select_dtypes(include=[np.number]).columns

nan_mask = stft_df[numeric_cols].isna()
random_values = np.random.uniform(low=min_val, high=max_val, size=stft_df[numeric_cols].shape)
stft_df.loc[:, numeric_cols] = stft_df[numeric_cols].where(~nan_mask, random_values)

# Clip numeric values
stft_df.loc[:, numeric_cols] = stft_df[numeric_cols].clip(lower=min_val, upper=max_val)

# ----------------------------
# Step 8: Save final CSV
# ----------------------------
output_csv = '/content/drive/MyDrive/Colab Notebooks/archive (1)/ECG_STFT_random_labels.csv'
stft_df.to_csv(output_csv, index=False)
print(f"✅ Final CSV with ECG values + random ARR/NSR labels + STFT saved at: {output_csv}")

# Show sample
stft_df
