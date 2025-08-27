import pandas as pd
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Input & output paths
input_csv = '/content/drive/MyDrive/Colab Notebooks/archive (6)/heart_disease_uci.csv'
output_csv = '/content/drive/MyDrive/Colab Notebooks/archive (6)/tokenized_output.csv'

# Load dataset
df = pd.read_csv(input_csv)

# Columns suitable for tokenization (categorical/textual)
text_columns = ['origin', 'sex', 'cp', 'restecg', 'thal']

# Tokenization function
def tokenize_text(sentence):
    if pd.isna(sentence):
        return ""
    doc = nlp(str(sentence))
    tokens = [token.text for token in doc if not token.is_space]
    return " ".join(tokens)

# Apply tokenization only to text columns
for col in text_columns:
    if col in df.columns:
        df[f"{col}_tokenized"] = df[col].apply(tokenize_text)

# Save the updated dataframe
df.to_csv(output_csv, index=False)

print(f"✅ Tokenization complete! Saved tokenized categorical columns to {output_csv}")
df