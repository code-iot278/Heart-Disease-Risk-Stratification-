import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Step 1: Load CSV
# -----------------------------
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/archive (3)/IterativeImputed.csv')

# -----------------------------
# Step 2: Select numeric columns
# -----------------------------
num_cols = df.select_dtypes(include=['float64','int64']).columns

# -----------------------------
# Step 3: Handle missing values (Iterative Imputer)
# -----------------------------
imputer = IterativeImputer(max_iter=10, random_state=42)
df[num_cols] = imputer.fit_transform(df[num_cols])

# -----------------------------
# Step 4: Apply Min-Max Scaling
# -----------------------------
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -----------------------------
# Step 5: Save processed data
# -----------------------------
output_path = '/content/drive/MyDrive/Colab Notebooks/archive (3)/IterativeImputed_MinMaxScaled.csv'
df.to_csv(output_path, index=False)

print("✅ Missing values imputed and numeric columns scaled successfully!")
print("Output saved at:", output_path)

# ✅ Print only column names
print("Column names:")
for col in df.columns:
    print(col)
df