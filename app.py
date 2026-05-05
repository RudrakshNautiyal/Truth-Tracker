import streamlit as st
import pickle
import re
import nltk
import requests
from datetime import datetime

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Load model & vectorizer ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open("model/model.pkl", "rb"))
    tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    return model, tfidf

model, tfidf = load_model()
stop_words = set(stopwords.words('english'))

# ── Text cleaning (same as preprocess.py) ─────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# ── Prediction function ────────────────────────────────────────────────────
def predict(text):
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return pred, proba

# ── Fetch live news from NewsAPI ───────────────────────────────────────────
def fetch_live_news(api_key, query="latest news", count=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": count,
        "apiKey":   api_key
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") == "ok":
            return data.get("articles", [])
        else:
            return []
    except:
        return []

# ── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Live Fake News Detector")
st.caption("ML model trained on 38,276 articles · 98.38% accuracy · Logistic Regression + TF-IDF")

tab1, tab2 = st.tabs(["📝 Check an Article", "📡 Live News Feed"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Manual check
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Paste any news headline or article text")
    user_input = st.text_area("Enter news text here:", height=200,
                              placeholder="e.g. Scientists discover new treatment for cancer...")

    if st.button("Analyse", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            pred, proba = predict(user_input)
            confidence  = round(max(proba) * 100, 2)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                if pred == 1:
                    st.success(f"### REAL NEWS")
                    st.metric("Confidence", f"{confidence}%")
                else:
                    st.error(f"### FAKE NEWS")
                    st.metric("Confidence", f"{confidence}%")

            with col2:
                st.write("**Probability breakdown**")
                st.progress(float(proba[1]), text=f"Real: {round(proba[1]*100,1)}%")
                st.progress(float(proba[0]), text=f"Fake: {round(proba[0]*100,1)}%")

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Live news feed
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Scan live headlines from NewsAPI")

    col1, col2 = st.columns([3, 1])
    with col1:
        api_key = st.text_input("NewsAPI key (free at newsapi.org):",
                                type="password",
                                placeholder="Enter your API key")
    with col2:
        query = st.text_input("Search topic:", value="politics")

    if st.button("Fetch & Analyse Live News", type="primary"):
        if not api_key.strip():
            st.warning("Please enter a NewsAPI key.")
        else:
            with st.spinner("Fetching latest news..."):
                articles = fetch_live_news(api_key, query=query, count=10)

            if not articles:
                st.error("No articles found. Check your API key or try a different topic.")
            else:
                st.success(f"Analysed {len(articles)} live articles")
                st.divider()

                for i, article in enumerate(articles):
                    title   = article.get("title", "") or ""
                    desc    = article.get("description", "") or ""
                    source  = article.get("source", {}).get("name", "Unknown")
                    url     = article.get("url", "#")
                    pubdate = article.get("publishedAt", "")[:10]

                    combined_text   = title + " " + desc
                    pred, proba     = predict(combined_text)
                    confidence      = round(max(proba) * 100, 2)

                    with st.container():
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f"**[{title}]({url})**")
                            st.caption(f"{source} · {pubdate}")
                        with c2:
                            if pred == 1:
                                st.success(f"REAL {confidence}%")
                            else:
                                st.error(f"FAKE {confidence}%")
                        st.divider()