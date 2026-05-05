from datasets import load_dataset
import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

# ── 1. Load all 3 splits from HuggingFace ──────────────────────────────────
print("Loading dataset...")
ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

train_df      = pd.DataFrame(ds['train'])        # 30,000 rows
validation_df = pd.DataFrame(ds['validation'])   # 6,000 rows
test_df       = pd.DataFrame(ds['test'])         # 8,267 rows

# ── 2. Merge into one clean master CSV ─────────────────────────────────────
df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

# Keep only the columns we need
df = df[['title', 'text', 'subject', 'date', 'label']]

# Drop rows where both title and text are empty
df.dropna(subset=['title', 'text'], inplace=True)

# Remove duplicates
df.drop_duplicates(subset=['title'], inplace=True)

print(f"\nTotal rows after cleaning : {len(df)}")
print(f"Fake news (0)             : {(df['label'] == 0).sum()}")
print(f"Real news (1)             : {(df['label'] == 1).sum()}")
print(f"Subjects                  : {df['subject'].unique()}")

# ── 3. Save splits separately + combined ───────────────────────────────────
train_df.to_csv("dataset/train.csv",      index=False)
validation_df.to_csv("dataset/val.csv",   index=False)
test_df.to_csv("dataset/test.csv",        index=False)
df.to_csv("dataset/base_dataset.csv",     index=False)

print("\nSaved:")
print("  dataset/train.csv")
print("  dataset/val.csv")
print("  dataset/test.csv")
print("  dataset/base_dataset.csv  ← this is your master file")