import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/archive (3)/heart.csv')

# Select numeric columns only
num_cols = df.select_dtypes(include=['float64','int64']).columns
# Apply MissForest imputation
imputer = IterativeImputer(max_iter=10, random_state=42)
df[num_cols] = imputer.fit_transform(df[num_cols])

df.to_csv('/content/drive/MyDrive/Colab Notebooks/archive (3)/IterativeImputed.csv', index=False)
print("Numeric columns imputed successfully!")

# ✅ Print only column names
print("Column names:")
for col in df.columns:
    print(col)
df
