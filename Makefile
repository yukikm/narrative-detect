.PHONY: demo

demo:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .
	narrative detect data/sample_posts.jsonl --out out/report.md
	narrative ideas out/report.json --out out/ideas.md
	@echo "Wrote out/report.md and out/ideas.md"
