from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests

try:
    # Optional dependency
    import feedparser  # type: ignore
except Exception:  # pragma: no cover
    feedparser = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_jsonl(records: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ingest_rss(url: str, out_path: Path, limit: int = 200) -> Path:
    """Fetch an RSS/Atom feed and write posts JSONL.

    Requires `feedparser`.
    """
    if feedparser is None:
        raise RuntimeError(
            "RSS ingestion requires optional dependency `feedparser`. "
            "Install with: pip install -e '.[ingest]'"
        )

    feed = feedparser.parse(url)
    records = []
    for i, e in enumerate(feed.entries[:limit]):
        text = (getattr(e, "title", "") or "").strip()
        if not text:
            continue
        link = (getattr(e, "link", "") or "").strip() or None
        ts = None
        # prefer published if present
        if getattr(e, "published", None):
            ts = str(getattr(e, "published"))
        records.append(
            {
                "id": link or f"rss:{i}",
                "ts": ts or _now_iso(),
                "text": text,
                "source": "rss",
                **({"url": link} if link else {}),
            }
        )

    _write_jsonl(records, out_path)
    return out_path


def ingest_x_recent_search(
    query: str,
    out_path: Path,
    bearer_token: Optional[str] = None,
    max_results: int = 50,
) -> Path:
    """Fetch recent posts from X (Twitter) API v2 recent search.

    Requires `X_BEARER_TOKEN` in env or passed explicitly.
    Note: this connector only works if you have API access.
    """
    token = bearer_token or os.getenv("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing X_BEARER_TOKEN. Put it in a .env file (never commit it) "
            "or export it in your shell."
        )

    url = "https://api.x.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": query,
        "max_results": max(10, min(100, int(max_results))),
        "tweet.fields": "created_at,author_id",  # keep it simple
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"X API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()

    records = []
    for t in data.get("data", []) or []:
        text = (t.get("text") or "").strip()
        if not text:
            continue
        records.append(
            {
                "id": t.get("id"),
                "ts": t.get("created_at") or _now_iso(),
                "text": text,
                "source": "x",
                "author_id": t.get("author_id"),
                "url": f"https://x.com/i/web/status/{t.get('id')}",
            }
        )

    _write_jsonl(records, out_path)
    return out_path


def ingest_telegram_export(
    export_json: Path,
    out_path: Path,
    limit: int = 500,
) -> Path:
    """Ingest a Telegram Desktop JSON export.

    This avoids scraping public pages and avoids needing to store phone credentials.

    How to get an export:
    - Telegram Desktop → Settings → Advanced → Export Telegram data → JSON.
    """
    obj = json.loads(export_json.read_text(encoding="utf-8"))
    messages = obj.get("messages") or []

    records = []
    for m in messages:
        if len(records) >= limit:
            break
        if m.get("type") != "message":
            continue
        text = m.get("text")
        # Telegram exports sometimes store text as list of spans.
        if isinstance(text, list):
            parts = []
            for x in text:
                if isinstance(x, str):
                    parts.append(x)
                elif isinstance(x, dict) and "text" in x:
                    parts.append(str(x["text"]))
            text = "".join(parts)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        ts = m.get("date") or _now_iso()
        records.append(
            {
                "id": m.get("id"),
                "ts": ts,
                "text": text,
                "source": "telegram",
            }
        )

    _write_jsonl(records, out_path)
    return out_path
