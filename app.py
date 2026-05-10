import streamlit as st
import pickle
import re
import nltk
import requests
from datetime import datetime

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Truth Tracker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #0f0f0f 100%);
    border: 1px solid #ffffff10;
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #e63946 0%, transparent 70%);
    opacity: 0.15;
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, #457b9d 0%, transparent 70%);
    opacity: 0.12;
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-title span { color: #e63946; }
.hero-sub {
    color: #888;
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.3px;
}
.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 1.6rem;
    flex-wrap: wrap;
}
.badge {
    background: #ffffff08;
    border: 1px solid #ffffff15;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #aaa;
    font-weight: 400;
}
.badge strong { color: #fff; font-weight: 500; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* Result cards */
.result-real {
    background: linear-gradient(135deg, #1a3a2a, #1e4a34);
    border: 1px solid #2d7a4a;
    border-left: 4px solid #2ecc71;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
}
.result-fake {
    background: linear-gradient(135deg, #3a1a1a, #4a1e1e);
    border: 1px solid #7a2d2d;
    border-left: 4px solid #e63946;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 0 0 0.3rem 0;
}
.result-real .result-label { color: #2ecc71; }
.result-fake .result-label { color: #e63946; }
.result-conf {
    font-size: 0.85rem;
    color: #aaa;
    margin: 0;
}
.result-conf strong { color: #fff; font-size: 1.1rem; }

/* News cards */
.news-card {
    background: #111118;
    border: 1px solid #ffffff0f;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: #ffffff20; }
.news-card-title {
    font-weight: 500;
    color: #e8e8e8;
    font-size: 0.95rem;
    margin: 0 0 0.5rem 0;
    line-height: 1.4;
}
.news-card-meta {
    font-size: 0.78rem;
    color: #555;
    margin: 0;
}
.pill-real {
    background: #1a3a2a;
    border: 1px solid #2d7a4a;
    color: #2ecc71;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    white-space: nowrap;
}
.pill-fake {
    background: #3a1a1a;
    border: 1px solid #7a2d2d;
    color: #e63946;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    white-space: nowrap;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0a0a0a;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #ffffff0f;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #666;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #1a1a1a !important;
    color: #fff !important;
}

/* Input fields */
.stTextArea textarea, .stTextInput input {
    background: #0d0d0d !important;
    border: 1px solid #ffffff15 !important;
    border-radius: 10px !important;
    color: #e8e8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #e63946 !important;
    box-shadow: 0 0 0 2px #e6394620 !important;
}

/* Button */
.stButton button {
    background: #e63946 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Progress bars */
.stProgress > div > div {
    background: #e63946 !important;
    border-radius: 4px !important;
}

/* Divider */
hr { border-color: #ffffff0a !important; }

/* Metric */
[data-testid="stMetric"] {
    background: #111118;
    border: 1px solid #ffffff0f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & vectorizer ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open(r"D:\Live Fake News Detector\model\model.pkl", "rb"))
    tfidf = pickle.load(open(r"D:\Live Fake News Detector\model\tfidf.pkl", "rb"))
    return model, tfidf

model, tfidf = load_model()
stop_words = set(stopwords.words('english'))

# ── Text cleaning ──────────────────────────────────────────────────────────
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

# ── Prediction ─────────────────────────────────────────────────────────────
def predict(text):
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return pred, proba

# ── NewsAPI ────────────────────────────────────────────────────────────────
def fetch_live_news(api_key, query="latest news", count=10):
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "language": "en",
            "sortBy": "publishedAt", "pageSize": count, "apiKey": api_key
        }, timeout=10)
        data = r.json()
        return data.get("articles", []) if data.get("status") == "ok" else []
    except:
        return []

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">Truth<span>Tracker</span></p>
    <p class="hero-sub">Real-time fake news detection powered by machine learning</p>
    <div class="hero-badges">
        <span class="badge"><strong>98.38%</strong> accuracy</span>
        <span class="badge"><strong>38,276</strong> training articles</span>
        <span class="badge"><strong>Logistic Regression</strong> + TF-IDF</span>
        <span class="badge"><strong>Live</strong> NewsAPI feed</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  Analyse Article  ", "  Live News Feed  "])

# ── Tab 1 ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Paste headline or article text</p>', unsafe_allow_html=True)
    user_input = st.text_area(
        label="text_input",
        label_visibility="collapsed",
        height=180,
        placeholder="e.g.  Scientists discover breakthrough cancer treatment..."
    )

    col_btn, col_space = st.columns([1, 5])
    with col_btn:
        analyse = st.button("Analyse", type="primary", use_container_width=True)

    if analyse:
        if not user_input.strip():
            st.warning("Please enter some text to analyse.")
        else:
            pred, proba = predict(user_input)
            confidence  = round(max(proba) * 100, 2)

            st.markdown("<br>", unsafe_allow_html=True)

            if pred == 1:
                st.markdown(f"""
                <div class="result-real">
                    <p class="result-label">REAL NEWS</p>
                    <p class="result-conf">Confidence: <strong>{confidence}%</strong></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fake">
                    <p class="result-label">FAKE NEWS</p>
                    <p class="result-conf">Confidence: <strong>{confidence}%</strong></p>
                </div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Real probability",  f"{round(proba[1]*100, 1)}%")
            with col2:
                st.metric("Fake probability", f"{round(proba[0]*100, 1)}%")

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(proba[1]), text=f"Real  {round(proba[1]*100,1)}%")
            st.progress(float(proba[0]), text=f"Fake  {round(proba[0]*100,1)}%")

# ── Tab 2 ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-title">Scan live headlines</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        api_key = st.text_input(
            "NewsAPI key",
            type="password",
            placeholder="Paste your free key from newsapi.org"
        )
    with col2:
        query = st.text_input("Topic", value="technology")

    if st.button("Fetch & Analyse", type="primary"):
        if not api_key.strip():
            st.warning("Please enter a NewsAPI key.")
        else:
            with st.spinner("Fetching latest headlines..."):
                articles = fetch_live_news(api_key, query=query, count=10)

            if not articles:
                st.error("No articles found. Check your API key or try a different topic.")
            else:
                real_count = 0
                fake_count = 0

                st.markdown("<br>", unsafe_allow_html=True)

                for article in articles:
                    title  = article.get("title", "")  or ""
                    desc   = article.get("description", "") or ""
                    source = article.get("source", {}).get("name", "Unknown")
                    url    = article.get("url", "#")
                    date   = article.get("publishedAt", "")[:10]

                    pred, proba = predict(title + " " + desc)
                    confidence  = round(max(proba) * 100, 1)

                    if pred == 1:
                        real_count += 1
                        pill = f'<span class="pill-real">REAL · {confidence}%</span>'
                    else:
                        fake_count += 1
                        pill = f'<span class="pill-fake">FAKE · {confidence}%</span>'

                    st.markdown(f"""
                    <div class="news-card">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
                            <p class="news-card-title"><a href="{url}" target="_blank" style="color:#e8e8e8;text-decoration:none;">{title}</a></p>
                            {pill}
                        </div>
                        <p class="news-card-meta">{source} · {date}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total analysed", len(articles))
                c2.metric("Classified real", real_count)
                c3.metric("Classified fake", fake_count)
