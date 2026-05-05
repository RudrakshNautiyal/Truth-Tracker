import requests
import pandas as pd
import os
from datetime import datetime

API_KEY  = "f1226fe39e6646d592e10afdb9cb592f"   # ← paste your key here
CSV_PATH = r"D:\Live Fake News Detector\dataset\base_dataset.csv"

TRUSTED_SOURCES = [
    "reuters.com", "bbc.com", "apnews.com",
    "theguardian.com", "ndtv.com", "thehindu.com"
]

def fetch_and_save(query="latest", count=20):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": count,
        "apiKey":   API_KEY
    }

    r    = requests.get(url, params=params, timeout=10)
    data = r.json()

    if data.get("status") != "ok":
        print(f"API error: {data.get('message')}")
        return

    articles = data.get("articles", [])
    rows     = []

    for a in articles:
        title   = a.get("title")   or ""
        text    = a.get("content") or a.get("description") or ""
        source  = a.get("source",  {}).get("name", "").lower()
        date    = a.get("publishedAt", "")[:10]

        if not title or not text:
            continue

        # Auto-label based on source trustworthiness
        is_trusted = any(s in source for s in TRUSTED_SOURCES)
        label = 1 if is_trusted else 0

        rows.append({
            "title":   title,
            "text":    text,
            "subject": "live",
            "date":    date,
            "label":   label
        })

    if not rows:
        print("No articles fetched.")
        return

    df_new = pd.DataFrame(rows)

    if os.path.exists(CSV_PATH):
        df_old      = pd.read_csv(CSV_PATH)
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset="title")
    else:
        df_combined = df_new

    df_combined.to_csv(CSV_PATH, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(df_new)} articles")
    print(f"Dataset now has {len(df_combined)} total rows")

if __name__ == "__main__":
    topics = ["politics", "health", "technology", "india", "world news"]
    for topic in topics:
        print(f"\nFetching: {topic}")
        fetch_and_save(query=topic, count=10)