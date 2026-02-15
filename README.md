# Narrative Detect (prototype)

Offline-first **narrative detection + idea generation** tool.

- Input: a JSONL file of short posts (e.g., tweets, telegram messages, reddit titles)
- Output: detected narrative clusters + trending scores + product/market ideas per narrative

This is a working prototype intended for rapid iteration + reproducible demos.

## Why this is different (strengths)

Most "narrative dashboards" fail on one of these: (1) you can’t reproduce the results, (2) ingestion requires secret keys / scraping, or (3) it doesn’t work well on non‑English text.

Narrative Detect is built to be:

- **Offline-first + reproducible**: ships with a bundled JSONL dataset and deterministic outputs so judges can verify quickly.
- **Privacy / key-safe by default**: core pipeline needs **no API keys**; optional ingestion uses `.env` (never committed).
- **Direct ingestion when you want it**: `narrative ingest` supports RSS (no keys), X (bearer token), and Telegram (export JSON).
- **Higher-quality clustering with graceful fallback**: `--method auto` uses semantic embeddings when available and falls back to TF‑IDF.
- **JP/multilingual friendly**: TF‑IDF switches to **char n-grams** for CJK so Japanese clusters don’t collapse.
- **Trend-relevant summaries**: representative posts are **recency-weighted** (so clusters read like what’s trending now).

## Quickstart

```bash
cd target-repos/narrative-detect
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# run on the bundled sample dataset
narrative detect data/sample_posts.jsonl --out out/report.md
narrative ideas out/report.json --out out/ideas.md
```

Outputs land in `out/`.

## Clustering modes

### Recommended: auto (best quality)

```bash
pip install -e ".[semantic]"

narrative detect data/sample_posts.jsonl --method auto --out out/report.md
```

`auto` uses semantic embeddings + HDBSCAN when available and falls back to TF‑IDF + KMeans automatically.

### TF‑IDF only (fast / no extra deps)

```bash
narrative detect data/sample_posts.jsonl --method tfidf --out out/report.md
```

## Optional: web demo (Streamlit)

```bash
pip install -e ".[web]"
streamlit run web/app.py
```

## Data sources

### How we get “SNS narratives”

This repo is designed to be **data-source agnostic**:
- core pipeline: JSONL posts → narratives → ideas (always works offline on the bundled sample)
- optional connectors: generate the same JSONL format from real sources

In the hosted demo (Colab), we run on the bundled sample dataset (`data/sample_posts.jsonl`).

#### Direct ingestion (optional)

Install with:

```bash
pip install -e ".[ingest]"
```

Then you can generate `out/posts.jsonl`:

- RSS/Atom (no keys):

```bash
narrative ingest rss --url https://example.com/feed.xml --out out/posts.jsonl
```

- X recent search (requires API access + `X_BEARER_TOKEN` in `.env`):

```bash
narrative ingest x --query "solana (depin OR restaking)" --out out/posts.jsonl
```

- Telegram (open channels): **export-based** (Telegram Desktop → Export data → JSON):

```bash
narrative ingest telegram-export --telegram-export path/to/result.json --out out/posts.jsonl
```

Notes:
- **Never commit secrets**. Use `.env` locally (see `.env.example`).
- If you don’t have keys/exports, the tool still works on `data/sample_posts.jsonl`.

This prototype is **offline-first** and ships with a small bundled sample dataset (`data/sample_posts.jsonl`) so anyone can reproduce outputs.

The intent is that you swap in real Solana signals by generating the same JSONL format from any combination of:

- Social: X (selected KOLs), Discord/Telegram public channels, forums, blogs
- Dev activity: GitHub repos/commits/stars
- On-chain: program deployments, TX volume spikes, protocol usage, wallet behavior

(Keeping ingestion modular is deliberate: data access varies by environment and API keys.)

## Data format

JSONL records with at least:

```json
{"id":"1","ts":"2026-02-14T12:00:00Z","text":"Solana memecoins are back…"}
```

Fields supported: `id`, `ts` (ISO8601), `text`, `source`, `url`.

## Example detected narratives + ideas (from sample dataset)

Running the Quickstart on the bundled sample produces narratives like:

- `solana / depin / demand`
- `session keys / session / keys`
- `resistance / sybil / sybil resistance`
- `restaking / better / rehypothecation`
- `defi / agents coordinating / ai agents`

…and for each narrative the tool generates **3–5 build ideas**.

See:
- `out/report.md` for the narrative list + explanations + representative posts
- `out/ideas.md` for the ideas (tied to each narrative)

## How narrative detection works (prototype)

1. **Vectorize** posts (TF‑IDF by default; sentence embeddings in `--method semantic`).
2. **Cluster** into narratives (KMeans by default; HDBSCAN in semantic mode).
3. **Summarize** each cluster with top keywords + representative posts.
4. Compute a simple **trend score** from volume + recentness + burstiness:
   - `volume = log(1 + n_posts)`
   - `recency = mean(exp(-age_hours / 24))` (roughly 24h half-life)
   - `burst = std(exp(-age_hours / 6))` (more sensitive to sudden spikes)
   - `trend_score = 0.7 * volume * recency + 0.3 * burst`
5. Generate **idea prompts** using deterministic templates (no API keys required).

## License

MIT

## Hosted demo

- **Colab (recommended / zero setup):**
  https://colab.research.google.com/github/yukikm/narrative-detect/blob/main/demo/NarrativeDetect_Demo.ipynb

(If you prefer to run locally, see Quickstart above.)

## Evidence / sample outputs

- Deterministic sample outputs (generated from `data/sample_posts.jsonl`):
  - `out/report.md`
  - `out/report.json`
  - `out/ideas.md`
- You can reproduce them locally with:

```bash
narrative detect data/sample_posts.jsonl --out out/report.md
narrative ideas out/report.json --out out/ideas.md
```
