import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

os.makedirs("model", exist_ok=True)

# ── 1. Load cleaned dataset ────────────────────────────────────────────────
print("Loading cleaned dataset...")
df = pd.read_csv("dataset/cleaned_dataset.csv")
print(f"Total rows: {len(df)}")

X = df['combined']
y = df['label']

# ── 2. Split data ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── 3. TF-IDF Vectorization ────────────────────────────────────────────────
print("\nVectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# ── 4. Train Logistic Regression (fast + strong baseline) ─────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_test_tfidf)

print("\n── Logistic Regression Results ──")
print(f"Accuracy : {accuracy_score(y_test, lr_preds):.4f}")
print(classification_report(y_test, lr_preds, target_names=['Fake', 'Real']))

# ── 5. Train Random Forest ─────────────────────────────────────────────────
print("\nTraining Random Forest (takes 1-2 mins)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_tfidf, y_train)
rf_preds = rf.predict(X_test_tfidf)

print("\n── Random Forest Results ──")
print(f"Accuracy : {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds, target_names=['Fake', 'Real']))

# ── 6. Pick best model and save ───────────────────────────────────────────
lr_acc = accuracy_score(y_test, lr_preds)
rf_acc = accuracy_score(y_test, rf_preds)

if lr_acc >= rf_acc:
    best_model = lr
    best_name  = "Logistic Regression"
else:
    best_model = rf
    best_name  = "Random Forest"

print(f"\nBest model: {best_name} ({max(lr_acc, rf_acc):.4f} accuracy)")

# Save model + vectorizer
pickle.dump(best_model, open("model/model.pkl", "wb"))
pickle.dump(tfidf,      open("model/tfidf.pkl", "wb"))

print("Saved model/model.pkl")
print("Saved model/tfidf.pkl")
print("\nTraining complete! Ready for the live demo.")
