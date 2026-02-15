# Narrative Detect (prototype)

Offline-first **narrative detection + idea generation** tool.

- Input: a JSONL file of short posts (e.g., tweets, telegram messages, reddit titles)
- Output: detected narrative clusters + trending scores + product/market ideas per narrative

This is a working prototype intended for rapid iteration + reproducible demos.

## Quickstart

```bash
cd target-repos/narrative-detect
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# run on the bundled sample dataset
narrative detect data/sample_posts.jsonl --out out/report.md
narrative ideas out/report.json --out out/ideas.md
```

Outputs land in `out/`.

## Optional: semantic clustering (better quality)

```bash
pip install -e ".[semantic]"

narrative detect data/sample_posts.jsonl --method semantic --out out/report.md
```

If optional deps are missing, the tool automatically falls back to TF‑IDF + KMeans.

## Optional: web demo (Streamlit)

```bash
pip install -e ".[web]"
streamlit run web/app.py
```

## Data format

JSONL records with at least:

```json
{"id":"1","ts":"2026-02-14T12:00:00Z","text":"Solana memecoins are back…"}
```

Fields supported: `id`, `ts` (ISO8601), `text`, `source`, `url`.

## How narrative detection works (prototype)

1. **Vectorize** posts (TF‑IDF by default; sentence embeddings in `--method semantic`).
2. **Cluster** into narratives (KMeans by default; HDBSCAN in semantic mode).
3. **Summarize** each cluster with top keywords + representative posts.
4. Compute a simple **trend score** from volume + recentness + burstiness.
5. Generate **idea prompts** using deterministic templates (no API keys required).

## License

MIT
