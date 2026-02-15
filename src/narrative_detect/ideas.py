from __future__ import annotations

from typing import Any, Dict, List


def _idea_templates(label: str, keywords: List[str]) -> List[str]:
    kw = ", ".join(keywords[:8])
    return [
        f"**Signal dashboard**: track `{label}` across sources; show volume, growth, and key influencers. Keywords: {kw}.",
        f"**Trading/ops playbook**: a weekly brief for `{label}` with catalysts, risks, and on-chain metrics to watch.",
        f"**Dev tool**: a lightweight SDK that tags incoming content as `{label}` and routes alerts to Slack/Telegram.",
        f"**Product wedge**: build a micro-product for the pain implied by `{label}` (e.g., UX, risk mgmt, onboarding), then expand into a suite.",
    ]


def generate_ideas_from_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Narrative ideas\n")

    narratives = report.get("narratives", [])
    for n in narratives:
        label = n.get("label", "(unknown)")
        keywords = n.get("keywords", [])
        lines.append(f"## {label}\n")
        lines.append(f"- size: {n.get('size')}\n- trend_score: {n.get('trend_score'):.3f}\n- keywords: {', '.join(keywords)}\n")
        lines.append("### Ideas\n")
        for it in _idea_templates(label, keywords):
            lines.append(f"- {it}")
        lines.append("\n### Representative posts\n")
        for p in n.get("representative_posts", [])[:4]:
            lines.append(f"- {p}")
        lines.append("\n")

    return "\n".join(lines)
