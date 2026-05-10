import requests
import pandas as pd
import os
from datetime import datetime, timezone

API_KEY  = "f1226fe39e6646d592e10afdb9cb592f"   # ← paste your key here
CSV_PATH = r"D:\Live Fake News Detector\dataset\base_dataset.csv"

# Lower volume per run to avoid hitting free-tier limits during demos
PAGE_SIZE    = 20
PAGES        = 1
LABEL_POLICY = "trusted_vs_untrusted"  # options: trusted_vs_untrusted, trusted_only

# Safety guard for free-tier limits
MAX_REQUESTS_PER_RUN = 10

# For heavier pulls, expand topics and iterate over date windows
TOPICS = [
    "politics", "health", "technology", "business", "science"
]

# Use small windows to reduce duplicates and comply with NewsAPI free plan limits
DATE_WINDOWS_DAYS = [1]

TRUSTED_SOURCES = [
    "reuters.com", "bbc.com", "apnews.com",
    "theguardian.com", "ndtv.com", "thehindu.com"
]

def fetch_and_save(query="latest", page_size=20, pages=1, from_date=None, to_date=None, req_state=None):
    url = "https://newsapi.org/v2/everything"
    rows = []

    if req_state is None:
        req_state = {"count": 0, "rate_limited": False}

    for page in range(1, pages + 1):
        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": page_size,
            "page":     page,
            "apiKey":   API_KEY
        }

        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        if req_state["count"] >= MAX_REQUESTS_PER_RUN:
            print("Reached MAX_REQUESTS_PER_RUN; stopping early.")
            break

        r    = requests.get(url, params=params, timeout=10)
        req_state["count"] += 1
        data = r.json()

        if data.get("status") != "ok":
            print(f"API error: {data.get('message')}")
            if "too many requests" in str(data.get("message", "")).lower():
                req_state["rate_limited"] = True
            break

        articles = data.get("articles", [])

        for a in articles:
            title   = a.get("title")   or ""
            text    = a.get("content") or a.get("description") or ""
            source  = a.get("source",  {}).get("name", "").lower()
            date    = a.get("publishedAt", "")[:10]

            if not title or not text:
                continue

            # Auto-label based on source trustworthiness
            is_trusted = any(s in source for s in TRUSTED_SOURCES)
            if LABEL_POLICY == "trusted_only" and not is_trusted:
                continue
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
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["title", "date"])
    else:
        df_combined = df_new

    df_combined.to_csv(CSV_PATH, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(df_new)} articles")
    print(f"Dataset now has {len(df_combined)} total rows")

if __name__ == "__main__":
    req_state = {"count": 0, "rate_limited": False}
    for topic in TOPICS:
        if req_state["rate_limited"]:
            print("Rate limit hit; stopping further requests.")
            break
        for window in DATE_WINDOWS_DAYS:
            if req_state["rate_limited"]:
                break
            from_date = (datetime.now(timezone.utc) - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
            to_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            print(f"\nFetching: {topic} | last {window} days")
            fetch_and_save(
                query=topic,
                page_size=PAGE_SIZE,
                pages=PAGES,
                from_date=from_date,
                to_date=to_date,
                req_state=req_state
            )
