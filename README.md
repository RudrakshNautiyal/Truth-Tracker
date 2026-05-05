# 🔍 Live Fake News Detector

An end-to-end machine learning project that detects fake news in real time using a trained Logistic Regression model and a live news feed powered by NewsAPI. Built with Python, scikit-learn, and Streamlit.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **98.38%** |
| Precision (Fake) | 99% |
| Precision (Real) | 98% |
| Recall (Fake) | 97% |
| Recall (Real) | 99% |
| F1-Score | **0.98** |

---

## 🗂️ Project Structure

```
Live Fake News Detector/
│
├── dataset/
│   ├── base_dataset.csv        ← master dataset (44k+ articles, grows daily)
│   ├── cleaned_dataset.csv     ← preprocessed version
│   ├── train.csv               ← training split (30,000 rows)
│   ├── val.csv                 ← validation split (6,000 rows)
│   └── test.csv                ← test split (8,267 rows)
│
├── model/
│   ├── model.pkl               ← trained Logistic Regression model
│   └── tfidf.pkl               ← fitted TF-IDF vectorizer
│
├── build_dataset.py            ← downloads base dataset from HuggingFace
├── preprocess.py               ← cleans and prepares text data
├── train.py                    ← trains and evaluates the ML model
├── live_fetch.py               ← fetches live articles via NewsAPI
├── update_model.py             ← runs full pipeline: fetch → preprocess → retrain
└── app.py                      ← Streamlit web app
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/live-fake-news-detector.git
cd live-fake-news-detector
```

### 2. Install dependencies

```bash
pip install datasets pandas scikit-learn nltk streamlit requests newsapi-python
```

### 3. Build the dataset

```bash
python build_dataset.py
```

### 4. Preprocess the data

```bash
python preprocess.py
```

### 5. Train the model

```bash
python train.py
```

### 6. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔄 Daily Update Pipeline

To fetch fresh news, retrain the model, and keep the dataset growing — run one command:

```bash
python update_model.py
```

This automatically:
1. Fetches 50 new articles from NewsAPI across 5 topics
2. Appends them to the master dataset
3. Cleans and preprocesses everything
4. Retrains the model on the full updated dataset

You can also schedule this to run automatically every day on Windows:

```powershell
schtasks /create /tn "FakeNewsUpdate" /tr "py D:\Live Fake News Detector\update_model.py" /sc daily /st 08:00
```

---

## 🧠 How It Works

### Dataset
- **Base dataset:** 44,267 labeled articles from [ErfanMoosaviMonazzah/fake-news-detection-dataset-English](https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English) (HuggingFace)
- **Live data:** NewsAPI fetches fresh articles daily, auto-labeled by source trustworthiness
- **Labels:** `0 = Fake`, `1 = Real`

### Preprocessing
- Lowercasing, URL removal, punctuation stripping
- Stopword removal using NLTK
- Title + body text combined into a single feature

### Feature Engineering
- TF-IDF Vectorizer with 50,000 features and bigrams `(1, 2)`
- Captures both individual words and two-word phrases

### Model
- **Logistic Regression** (`max_iter=1000`)
- Chosen over Random Forest for higher accuracy (98.38% vs 98.03%) and faster inference
- Saved with `pickle` for instant loading in the app

### Live Demo
- **Tab 1:** Paste any headline or article text → instant real/fake prediction with confidence score
- **Tab 2:** Enter NewsAPI key + topic → fetches and analyses 10 live headlines in real time

---

## ⚠️ Limitations

- The model detects **writing style and linguistic patterns** of fake news, not factual accuracy
- It cannot fact-check against real-world knowledge (e.g. outdated facts written in a neutral tone may be classified as real)
- Training data is mostly US political news (2016–2017) — performance may vary on other domains
- Live articles are auto-labeled by source, not human-verified

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12+ | Core language |
| pandas | Data handling |
| scikit-learn | TF-IDF + Logistic Regression |
| NLTK | Text preprocessing |
| HuggingFace Datasets | Base dataset download |
| NewsAPI | Live news feed |
| Streamlit | Web app interface |
| pickle | Model serialization |


---

## 👤 Author

Rudraksh Nautiyal
B.Tech — CSE
Jaypee University of Information Technology
