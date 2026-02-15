from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .detect import detect_narratives
from .ideas import generate_ideas_from_report
from .ingest import ingest_rss, ingest_telegram_export, ingest_x_recent_search

app = typer.Typer(add_completion=False, help="Narrative detection + idea generation (prototype)")
console = Console()


@app.command()
def detect(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="JSONL posts"),
    out: Path = typer.Option(Path("out/report.md"), help="Markdown report output path"),
    json_out: Path = typer.Option(Path("out/report.json"), help="Machine-readable report JSON"),
    method: str = typer.Option("auto", help="auto|tfidf|semantic"),
    k: int = typer.Option(6, help="Clusters for tfidf mode (ignored for semantic mode)")
):
    """Detect narratives (clusters) and compute simple trend scores."""
    out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    report = detect_narratives(input_path, method=method, k=k)

    # write json
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # write md
    md = []
    md.append(f"# Narrative report\n\nInput: `{input_path}`\nMethod: `{method}`\n")
    md.append("## Narratives\n")
    for n in report["narratives"]:
        md.append(f"### {n['label']}\n")
        md.append(f"- size: {n['size']}\n- trend_score: {n['trend_score']:.3f}\n- keywords: {', '.join(n['keywords'])}\n")
        md.append("\nRepresentative posts:\n")
        for p in n["representative_posts"]:
            md.append(f"- {p}")
        md.append("\n")
    out.write_text("\n".join(md), encoding="utf-8")

    console.print(f"Wrote [bold]{out}[/bold] and [bold]{json_out}[/bold]")


@app.command()
def ideas(
    report_json: Path = typer.Argument(..., exists=True, readable=True, help="report.json from `detect`"),
    out: Path = typer.Option(Path("out/ideas.md"), help="Markdown ideas output path"),
):
    """Generate product/market ideas from a narrative report."""
    out.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_json.read_text(encoding="utf-8"))
    md = generate_ideas_from_report(report)
    out.write_text(md, encoding="utf-8")
    console.print(f"Wrote [bold]{out}[/bold]")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="rss|x|telegram-export"),
    out: Path = typer.Option(Path("out/posts.jsonl"), help="Output JSONL path"),
    url: Optional[str] = typer.Option(None, help="RSS feed URL (for source=rss)"),
    query: Optional[str] = typer.Option(None, help="X recent search query (for source=x)"),
    max_results: int = typer.Option(50, help="X max results (10-100)"),
    telegram_export: Optional[Path] = typer.Option(
        None, help="Telegram Desktop export JSON (for source=telegram-export)"
    ),
):
    """Ingest posts from a data source into JSONL.

    Keys/API access:
    - rss: no keys needed
    - x: requires X_BEARER_TOKEN in env/.env
    - telegram-export: requires a local Telegram Desktop JSON export file

    Never commit secrets in .env.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    source = source.strip().lower()
    if source == "rss":
        if not url:
            raise typer.BadParameter("--url is required for source=rss")
        path = ingest_rss(url, out)
    elif source == "x":
        if not query:
            raise typer.BadParameter("--query is required for source=x")
        path = ingest_x_recent_search(query=query, out_path=out, max_results=max_results)
    elif source in {"telegram-export", "telegram_export", "telegram"}:
        if not telegram_export:
            raise typer.BadParameter(
                "--telegram-export is required for source=telegram-export"
            )
        path = ingest_telegram_export(telegram_export, out)
    else:
        raise typer.BadParameter("source must be one of: rss|x|telegram-export")

    console.print(f"Wrote [bold]{path}[/bold]")


if __name__ == "__main__":
    app()
