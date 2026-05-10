from datasets import load_dataset
import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

# ── 1. Load all 3 splits from HuggingFace ──────────────────────────────────
print("Loading primary dataset...")
ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

train_df      = pd.DataFrame(ds['train'])        # 30,000 rows
validation_df = pd.DataFrame(ds['validation'])   # 6,000 rows
test_df       = pd.DataFrame(ds['test'])         # 8,267 rows

# ── 2. Merge into one clean master CSV ─────────────────────────────────────
df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

# ── 2b. Optional: add LIAR dataset (short statements) ──────────────────────
print("Loading LIAR dataset (optional)...")
try:
	liar = load_dataset("liar")

	def map_liar_label(label):
		# 0: pants-fire, 1: false, 2: barely-true, 3: half-true, 4: mostly-true, 5: true
		return 0 if label in [0, 1, 2] else 1

	liar_rows = []
	for split in ["train", "validation", "test"]:
		split_df = pd.DataFrame(liar[split])
		if "statement" not in split_df.columns:
			continue
		split_df = split_df[["statement", "label"]].dropna()
		split_df.rename(columns={"statement": "text"}, inplace=True)
		split_df["title"] = split_df["text"]
		split_df["subject"] = "liar"
		split_df["date"] = ""
		split_df["label"] = split_df["label"].apply(map_liar_label)
		split_df = split_df[["title", "text", "subject", "date", "label"]]
		liar_rows.append(split_df)

	if liar_rows:
		liar_df = pd.concat(liar_rows, ignore_index=True)
		df = pd.concat([df, liar_df], ignore_index=True)
		print(f"Added LIAR rows: {len(liar_df)}")
	else:
		print("LIAR dataset missing expected columns; skipped.")
except Exception as e:
	print(f"LIAR dataset not available; skipped. Reason: {e}")

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
