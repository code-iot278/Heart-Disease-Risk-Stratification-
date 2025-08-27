import pandas as pd
from transformers import pipeline
# ------------------------------
# Input & Output CSV
# ------------------------------
input_csv = "/content/drive/MyDrive/Colab Notebooks/archive (6)/tokenized_output.csv"
output_csv = "/content/drive/MyDrive/Colab Notebooks/archive (6)/EHRdata_with_entities.csv"

# Load dataset
df = pd.read_csv(input_csv)

# ------------------------------
# Replace blank or NaN values in text columns
# ------------------------------
text_columns = df.select_dtypes(include='object').columns
df[text_columns] = df[text_columns].fillna("Unknown")  # Fill NaN with 'Unknown'
df[text_columns] = df[text_columns].replace(r'^\s*$', "Unknown", regex=True)  # Replace empty strings

# ------------------------------
# Load public NER pipeline (BERT-based)
# ------------------------------
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER",
                        aggregation_strategy="simple", device=-1)  # CPU

# ------------------------------
# Function to extract entities from text
# ------------------------------
def extract_entities(text):
    results = ner_pipeline(str(text))
    return "; ".join([f"{r['word']} ({r['entity_group']})" for r in results])

# ------------------------------
# Apply NER to all object/text columns
# ------------------------------
for col in text_columns:
    df[f"{col}_entities"] = df[col].apply(extract_entities)

# ------------------------------
# Save output
# ------------------------------
df.to_csv(output_csv, index=False)
print(f"✅ Entity extraction completed! File saved at: {output_csv}")