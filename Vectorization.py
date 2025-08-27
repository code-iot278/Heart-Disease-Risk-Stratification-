import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ------------------------------
# Load ClinicalBERT model
# ------------------------------
model_name = "emilyalsentzer/Bio_ClinicalBERT"  # ClinicalBERT base
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set to evaluation mode

# ------------------------------
# Load dataset
# ------------------------------
input_csv = '/content/drive/MyDrive/Colab Notebooks/archive (6)/EHRdata_with_entities.csv'
df = pd.read_csv(input_csv)

# Columns to vectorize (categorical/text columns)
text_columns = ['origin', 'sex', 'cp', 'restecg', 'thal']

# ------------------------------
# Function to get embeddings
# ------------------------------
def get_embedding(text):
    if pd.isna(text):
        return [0.0]*768  # placeholder for missing text
    # Encode text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding as sentence representation
        cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().cpu().numpy()
    return cls_embedding

# ------------------------------
# Apply embedding to each text column
# ------------------------------
for col in text_columns:
    if col in df.columns:
        embeddings = df[col].apply(get_embedding)
        # Split each dimension into separate columns
        emb_df = pd.DataFrame(embeddings.tolist(), columns=[f"{col}_emb_{i}" for i in range(embeddings.iloc[0].shape[0])])
        df = pd.concat([df, emb_df], axis=1)

# ------------------------------
# Save dataframe with embeddings
# ------------------------------
output_csv = '/content/drive/MyDrive/Colab Notebooks/archive (6)/clinical_embeddings.csv'
df.to_csv(output_csv, index=False)
print(f"✅ Clinical embeddings saved to {output_csv}")
df