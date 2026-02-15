# Narrative Detect (prototype)

Offline-first **narrative detection + idea generation** tool.

- Input: a JSONL file of short posts (e.g., tweets, telegram messages, reddit titles)
- Output: detected narrative clusters + trending scores + product/market ideas per narrative

This is a working prototype intended for rapid iteration + reproducible demos.

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

## Data sources

### How we get “SNS narratives”

Right now this repo **does not scrape X/Discord/Telegram directly**. It’s designed to be **data-source agnostic**: you provide a JSONL of posts, and the tool detects narratives + generates ideas.

In the hosted demo (Colab), we run on the bundled sample dataset (`data/sample_posts.jsonl`).

To run on real data, you typically generate the same JSONL format via one of these paths:
- **Exports**: X export / CSV → convert to JSONL
- **APIs** (requires your own keys): X API, Reddit API, Farcaster/Hubs, etc.
- **Public feeds**: RSS/blog feeds / GitHub events (no auth)

If you tell me the exact sources you want (e.g. X lists + a few Telegram channels), I can add an `ingest` command/module that produces `posts.jsonl` from those sources in a reproducible way (keeping keys optional and out of the repo).

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
