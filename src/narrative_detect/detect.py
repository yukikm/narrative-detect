from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(^|\s)@\w+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _normalize_text(t: str) -> str:
    # Keep it conservative: remove URLs/handles, normalize whitespace.
    t = t.strip()
    t = _URL_RE.sub("", t)
    t = _HANDLE_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def _parse_posts(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = str(obj.get("text", "") or "")
        norm = _normalize_text(text)
        if not norm:
            continue
        rows.append(
            {
                "id": str(obj.get("id", "")),
                "ts": obj.get("ts"),
                "text": norm,
                "source": obj.get("source"),
                "url": obj.get("url"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No posts found")

    # Basic de-duplication (exact normalized text). This reduces cluster noise.
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # timestamps optional
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    else:
        df["ts"] = pd.NaT
    return df


def _trend_score(ts: pd.Series) -> float:
    """Heuristic: more posts + more recent posts => higher score."""
    n = int(ts.notna().sum())
    if n == 0:
        return 0.0
    now = datetime.now(timezone.utc)
    ages_h = ((now - ts.dropna()).dt.total_seconds() / 3600.0).clip(lower=0.0)
    recency = float(np.mean(np.exp(-ages_h / 24.0)))  # 24h half-life-ish
    volume = math.log(1 + n)
    burst = float(np.std(np.exp(-ages_h / 6.0)))  # 6h sensitivity
    return (0.7 * volume * recency) + (0.3 * burst)


_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_TOKEN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]{2,}")


def _looks_cjk(texts: List[str], sample: int = 30) -> bool:
    for t in texts[:sample]:
        if _CJK_RE.search(t):
            return True
    return False


def _vectorizer_for(texts: List[str]) -> TfidfVectorizer:
    # For JP/CJK, word-based tokenization performs poorly. Use char ngrams.
    if _looks_cjk(texts):
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )

    # Default: English-ish word model.
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )


def _top_keywords_from_cluster(texts: List[str], top_k: int = 8) -> List[str]:
    if _looks_cjk(texts):
        # Simple phrase mining for readable JP labels.
        counts: Dict[str, int] = {}
        for t in texts:
            for m in _CJK_TOKEN_RE.findall(t):
                counts[m] = counts.get(m, 0) + 1
        kws = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
        return [k for k, _ in kws[:top_k]]

    vec = _vectorizer_for(texts)
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    idx = np.argsort(-scores)[:top_k]
    inv_vocab = {i: t for t, i in vec.vocabulary_.items()}
    return [inv_vocab[i] for i in idx if i in inv_vocab]


def _representative_posts(texts: List[str], X: np.ndarray, top_k: int = 4) -> List[str]:
    # centroid similarity
    centroid = np.mean(X, axis=0, keepdims=True)
    sims = cosine_similarity(X, centroid).ravel()
    idx = np.argsort(-sims)[:top_k]
    return [texts[i] for i in idx]


def _auto_k(n: int) -> int:
    # Small heuristic: scale clusters with dataset size.
    return max(3, min(12, int(round(math.sqrt(max(1, n))))))


def _tfidf_cluster(df: pd.DataFrame, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    texts = df["text"].tolist()
    vec = _vectorizer_for(texts)
    X = vec.fit_transform(texts)
    k_eff = _auto_k(len(df)) if k <= 0 else k
    km = KMeans(n_clusters=max(2, min(k_eff, len(df))), n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    return X.toarray(), labels, km.cluster_centers_


def _semantic_cluster(df: pd.DataFrame):
    # optional deps; import lazily
    from sentence_transformers import SentenceTransformer
    import hdbscan

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(df["text"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(3, len(df) // 20), metric="euclidean")
    labels = clusterer.fit_predict(emb)
    return emb, labels


def detect_narratives(input_path: Path, method: str = "auto", k: int = 6) -> Dict[str, Any]:
    df = _parse_posts(input_path)

    method = method.lower().strip()
    if method in {"auto", "semantic"}:
        try:
            X, labels = _semantic_cluster(df)
            method = "semantic"
        except Exception as e:  # fallback to tfidf
            X, labels, _ = _tfidf_cluster(df, k=k)
            method = f"tfidf_fallback ({type(e).__name__})"
    elif method == "tfidf":
        X, labels, _ = _tfidf_cluster(df, k=k)
    else:
        raise ValueError("method must be one of: auto|tfidf|semantic")

    df = df.copy()
    df["label"] = labels

    narratives = []
    for lab in sorted(df["label"].unique()):
        if lab == -1:
            # noise cluster in HDBSCAN
            continue
        sub = df[df["label"] == lab]
        texts = sub["text"].tolist()
        Xsub = X[sub.index.values]
        keywords = _top_keywords_from_cluster(texts)
        reps = _representative_posts(texts, Xsub)
        narratives.append(
            {
                "id": int(lab),
                "label": " / ".join(keywords[:3]) if keywords else f"cluster-{lab}",
                "size": int(len(sub)),
                "keywords": keywords,
                "trend_score": float(_trend_score(sub["ts"])),
                "representative_posts": reps,
            }
        )

    narratives = sorted(narratives, key=lambda x: x["trend_score"], reverse=True)
    return {
        "meta": {
            "input": str(input_path),
            "method": method,
            "n_posts": int(len(df)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "narratives": narratives,
    }
