import pandas as pd
import re
import nltk
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords

os.makedirs("dataset", exist_ok=True)

# ── 1. Load master dataset ─────────────────────────────────────────────────
print("Loading base_dataset.csv...")
df = pd.read_csv("dataset/base_dataset.csv")
print(f"Loaded {len(df)} rows")

# ── 2. Clean text function ─────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                                # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)              # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()          # remove extra spaces
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# ── 3. Apply cleaning to title and text ───────────────────────────────────
print("Cleaning text... (this takes ~30 seconds)")
df['clean_title'] = df['title'].apply(clean_text)
df['clean_text']  = df['text'].apply(clean_text)

# Combine title + text into one field for better accuracy
df['combined']    = df['clean_title'] + ' ' + df['clean_text']

# Drop rows where combined is empty after cleaning
df = df[df['combined'].str.strip() != '']

# ── 4. Save cleaned dataset ────────────────────────────────────────────────
df.to_csv("dataset/cleaned_dataset.csv", index=False)

print(f"\nDone! Cleaned dataset saved to dataset/cleaned_dataset.csv")
print(f"Total rows : {len(df)}")
print(f"Fake  (0)  : {(df['label'] == 0).sum()}")
print(f"Real  (1)  : {(df['label'] == 1).sum()}")
print("\nSample cleaned text:")
print(df['combined'].iloc[0][:200])
