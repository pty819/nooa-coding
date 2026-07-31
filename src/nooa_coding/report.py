"""Structured output: JSON, SARIF, and HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ReportMetadata(BaseModel):
    """Metadata for a generated report."""

    session_id: str = ""
    generated_at: str = ""
    tool_name: str = "nooa-coding"
    tool_version: str = "0.1.0"


class SessionReport(BaseModel):
    """Complete session report data."""

    metadata: ReportMetadata
    summary: str = ""
    status: str = ""
    turns: int = 0
    changed_files: list[str] = []
    verifications: list[dict[str, Any]] = []
    token_usage: dict[str, Any] = {}
    events_count: int = 0
    duration_seconds: float = 0.0


# ─── JSON Export ─────────────────────────────────────────────────────────────


def export_json(report: SessionReport, indent: int = 2) -> str:
    """Export a session report as formatted JSON."""
    return report.model_dump_json(indent=indent)


def export_events_jsonl(events: list[Any]) -> str:
    """Export session events as JSONL."""
    lines: list[str] = []
    for event in events:
        if hasattr(event, "model_dump_json"):
            lines.append(event.model_dump_json())
        elif isinstance(event, dict):
            lines.append(json.dumps(event, default=str))
        else:
            lines.append(json.dumps({"text": str(event)}, default=str))
    return "\n".join(lines) + "\n" if lines else ""


# ─── SARIF Export ────────────────────────────────────────────────────────────


def _sarif_result(
    rule_id: str,
    message: str,
    path: str = "",
    line: int = 1,
    level: str = "warning",
) -> dict[str, Any]:
    """Create a SARIF result object."""
    result: dict[str, Any] = {
        "ruleId": rule_id,
        "level": level,
        "message": {"text": message},
    }
    if path:
        result["locations"] = [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": path},
                    "region": {"startLine": line},
                }
            }
        ]
    return result


def export_sarif(
    findings: list[dict[str, Any]],
    *,
    tool_name: str = "nooa-coding",
    tool_version: str = "0.1.0",
) -> str:
    """Export findings as SARIF 2.1.0 format.

    Each finding should have: rule_id, message, path (optional), line (optional), level (optional).
    """
    rules: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    for finding in findings:
        rule_id = finding.get("rule_id", "general")
        if rule_id not in rules:
            rules[rule_id] = {
                "id": rule_id,
                "shortDescription": {"text": finding.get("message", rule_id)[:100]},
            }
        results.append(
            _sarif_result(
                rule_id=rule_id,
                message=finding.get("message", ""),
                path=finding.get("path", ""),
                line=finding.get("line", 1),
                level=finding.get("level", "warning"),
            )
        )

    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": tool_name,
                        "version": tool_version,
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(sarif, indent=2)


def review_to_sarif(review_text: str, *, max_findings: int = 20) -> str:
    """Convert a code review text into SARIF format.

    Parses common review output patterns into findings.
    """
    findings: list[dict[str, Any]] = []
    current_file = ""
    current_line = 1

    for line in review_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect file references like "src/module.py:42" or "**src/module.py**"
        if ":" in line and any(
            ext in line for ext in (".py", ".js", ".ts", ".rs", ".go", ".java")
        ):
            parts = line.split(":")
            for i, part in enumerate(parts):
                if any(ext in part for ext in (".py", ".js", ".ts", ".rs", ".go")):
                    current_file = part.strip().strip("*`\"'")
                    if i + 1 < len(parts) and parts[i + 1].strip().isdigit():
                        current_line = int(parts[i + 1].strip())
                    break

        # Detect issue markers.
        lower = line.lower()
        level = "warning"
        rule_id = "review-finding"

        if any(marker in lower for marker in ("bug", "error", "critical", "security")):
            level = "error"
            rule_id = "potential-bug"
        elif any(marker in lower for marker in ("warning", "caution", "risk")):
            level = "warning"
            rule_id = "potential-issue"
        elif any(marker in lower for marker in ("suggest", "consider", "style")):
            level = "note"
            rule_id = "suggestion"
        else:
            continue

        if len(findings) >= max_findings:
            break

        findings.append(
            {
                "rule_id": rule_id,
                "message": line[:200],
                "path": current_file,
                "line": current_line,
                "level": level,
            }
        )

    return export_sarif(findings)


# ─── HTML Report ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NOOA Coding Session Report</title>
<style>
  :root {{ --bg: #1a1b26; --fg: #c0caf5; --accent: #7aa2f7; --green: #9ece6a; --red: #f7768e; --dim: #565f89; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--fg); margin: 0; padding: 2rem; line-height: 1.6; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }}
  h2 {{ color: var(--fg); margin-top: 2rem; }}
  .meta {{ color: var(--dim); font-size: 0.9rem; }}
  .stat {{ display: inline-block; background: #24283b; padding: 0.5rem 1rem; border-radius: 6px; margin: 0.25rem; }}
  .stat-value {{ font-size: 1.5rem; font-weight: bold; color: var(--accent); }}
  .stat-label {{ font-size: 0.8rem; color: var(--dim); }}
  .file-list {{ list-style: none; padding: 0; }}
  .file-list li {{ padding: 0.25rem 0; font-family: monospace; }}
  .file-list li:before {{ content: "✎ "; color: var(--green); }}
  .verification {{ padding: 0.5rem; margin: 0.5rem 0; border-radius: 4px; }}
  .verification.pass {{ background: rgba(158, 206, 106, 0.1); border-left: 3px solid var(--green); }}
  .verification.fail {{ background: rgba(247, 118, 142, 0.1); border-left: 3px solid var(--red); }}
  .summary {{ background: #24283b; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }}
  footer {{ margin-top: 3rem; color: var(--dim); font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1>🤖 NOOA Coding Session Report</h1>
<p class="meta">Session: {session_id} | Generated: {generated_at}</p>

<div>
  <div class="stat"><div class="stat-value">{status}</div><div class="stat-label">Status</div></div>
  <div class="stat"><div class="stat-value">{turns}</div><div class="stat-label">Turns</div></div>
  <div class="stat"><div class="stat-value">{files_changed}</div><div class="stat-label">Files Changed</div></div>
  <div class="stat"><div class="stat-value">{duration}</div><div class="stat-label">Duration</div></div>
</div>

<h2>Summary</h2>
<div class="summary">{summary}</div>

<h2>Changed Files</h2>
<ul class="file-list">
{changed_files_html}
</ul>

<h2>Verifications</h2>
{verifications_html}

<h2>Token Usage</h2>
<div>
{token_usage_html}
</div>

<footer>Generated by nooa-coding v{version}</footer>
</div>
</body>
</html>
"""


def export_html(report: SessionReport) -> str:
    """Generate an HTML report from session data."""
    changed_files_html = "\n".join(
        f"<li>{_escape_html(f)}</li>" for f in report.changed_files
    ) or "<li>No files changed</li>"

    verifications_html = ""
    for v in report.verifications:
        passed = v.get("passed", False)
        css_class = "pass" if passed else "fail"
        icon = "✓" if passed else "✗"
        cmd = _escape_html(v.get("command", ""))
        verifications_html += (
            f'<div class="verification {css_class}">'
            f"<strong>{icon}</strong> <code>{cmd}</code>"
            f"</div>\n"
        )
    if not verifications_html:
        verifications_html = "<p>No verifications run.</p>"

    token_usage_html = ""
    for key, value in report.token_usage.items():
        token_usage_html += (
            f'<div class="stat"><div class="stat-value">{value}</div>'
            f'<div class="stat-label">{_escape_html(key)}</div></div>\n'
        )
    if not token_usage_html:
        token_usage_html = "<p>No token usage recorded.</p>"

    duration = f"{report.duration_seconds:.1f}s" if report.duration_seconds else "N/A"

    return _HTML_TEMPLATE.format(
        session_id=_escape_html(report.metadata.session_id),
        generated_at=_escape_html(report.metadata.generated_at),
        status=_escape_html(report.status),
        turns=report.turns,
        files_changed=len(report.changed_files),
        duration=duration,
        summary=_escape_html(report.summary),
        changed_files_html=changed_files_html,
        verifications_html=verifications_html,
        token_usage_html=token_usage_html,
        version=report.metadata.tool_version,
    )


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ─── Convenience ─────────────────────────────────────────────────────────────


def write_report(
    report: SessionReport,
    output_dir: str | Path,
    *,
    formats: list[str] | None = None,
) -> list[Path]:
    """Write report in multiple formats. Returns list of written file paths.

    Supported formats: "json", "html", "sarif" (requires findings in verifications).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["json", "html"]
    written: list[Path] = []

    base_name = f"report-{report.metadata.session_id or 'session'}"

    if "json" in formats:
        path = output_dir / f"{base_name}.json"
        path.write_text(export_json(report), encoding="utf-8")
        written.append(path)

    if "html" in formats:
        path = output_dir / f"{base_name}.html"
        path.write_text(export_html(report), encoding="utf-8")
        written.append(path)

    if "sarif" in formats:
        # Convert verifications to findings.
        findings = [
            {
                "rule_id": "verification",
                "message": v.get("command", ""),
                "level": "error" if not v.get("passed", True) else "note",
            }
            for v in report.verifications
        ]
        path = output_dir / f"{base_name}.sarif"
        path.write_text(export_sarif(findings), encoding="utf-8")
        written.append(path)

    return written


__all__ = [
    "ReportMetadata",
    "SessionReport",
    "export_html",
    "export_json",
    "export_events_jsonl",
    "export_sarif",
    "review_to_sarif",
    "write_report",
]
