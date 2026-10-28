from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .common import read_json


def build_report(root):
    root = Path(root).resolve()
    manifests = sorted(root.rglob("manifest.json"))
    rows, points = [], []
    for path in manifests:
        trial = path.parent
        if not (trial / "summary.json").exists():
            continue
        m, s = read_json(path), read_json(trial / "summary.json")
        name = str(trial.relative_to(root)) or "."
        rows.append(
            {
                "trial": name,
                "arm": m["arm"],
                "repetition": m["repetition"],
                "status": s["status"],
                "score": s["best_score"],
                "seconds": s["elapsed_seconds"],
                "tokens": s["total_tokens"] if s["usage_known"] else None,
                "synthetic": m["synthetic"],
                "seed_sha256": m["seed_sha256"],
                "model": m["model"],
            }
        )
        for line in (trial / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["type"] in {"baseline", "round_completed"}:
                points.append(
                    {
                        "trial": name,
                        "arm": m["arm"],
                        "round": e["round"],
                        "seconds": e["elapsed_seconds"],
                        "score": e.get("best_score", e.get("score")),
                        "kind": e.get("plan", {}).get("kind", "seed"),
                        "accepted": e.get("decision", {}).get("accepted", False),
                    }
                )
    if not rows:
        raise ValueError("no completed trial summaries found")
    for filename, records in (("summary.csv", rows), ("trajectory.csv", points)):
        with (root / filename).open("w", newline="") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    colors = {"baseline": "#64748b", "hypothesis": "#087e8b", "rethink": "#d97706"}
    xmax = max([p["seconds"] for p in points] + [1])
    time_scale = 1 if xmax < 120 else 60
    time_label = "seconds" if time_scale == 1 else "minutes"
    ymax = max([p["score"] or 0 for p in points] + [1]) * 1.1
    svg = [
        '<svg viewBox="0 0 900 390" role="img" aria-label="Measured incumbent performance over elapsed time">',
        '<path d="M65 20V330H875" fill="none" stroke="#94a3b8"/>',
    ]
    for j in range(6):
        y = 330 - j / 5 * 300
        svg.append(f'<text x="52" y="{y + 4}" text-anchor="end">{ymax * j / 5:.2f}×</text>')
        svg.append(f'<path d="M65 {y}H875" stroke="#e2e8f0"/>')
        x = 65 + j / 5 * 810
        svg.append(f'<text x="{x}" y="353" text-anchor="middle">{xmax * j / 5 / time_scale:.1f}</text>')
    for row in rows:
        group = [p for p in points if p["trial"] == row["trial"] and p["score"] is not None]
        xy = [(65 + p["seconds"] / xmax * 810, 330 - p["score"] / ymax * 300) for p in group]
        color = colors[row["arm"]]
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        svg.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" opacity=".8"/>'
        )
        for p, (x, y) in zip(group, xy):
            title = html.escape(f"{p['trial']}, round {p['round']}, {p['kind']}, {p['score']:.3f}×")
            fill = "white" if p["kind"] == "diagnostic" else color
            svg.append(
                f'<circle cx="{x}" cy="{y}" r="4" fill="{fill}" stroke="{color}"><title>{title}</title></circle>'
            )
    svg += [
        f'<text x="470" y="382" text-anchor="middle">Elapsed {time_label} (including evaluation)</text>',
        "</svg>",
    ]
    synthetic = any(r["synthetic"] for r in rows)
    banner = (
        "SYNTHETIC SMOKE TEST — no measured GPU or agent improvement"
        if synthetic
        else "Experimental results — individual runs, no significance claim"
    )

    def display(value):
        if value is None:
            return "unknown"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(display(r[k]))}</td>"
            for k in ("trial", "status", "score", "seconds", "tokens")
        )
        + "</tr>"
        for r in rows
    )
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Kernel experiment</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172033}}
svg{{width:100%;font-size:12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.banner{{background:#fff1c2;padding:16px}}
</style><h1>Kernel hypothesis experiment</h1><p class="banner">{banner}</p>
<p>Gray: baseline · teal: hypothesis · orange: rethink. Hollow markers are diagnostic rounds.
Y axis: selected incumbent's measured reference/candidate speedup. Ratios can fluctuate with measurement noise.
Lines connect observations; they do not imply continuous measurement.</p>{"".join(svg)}
<table><thead><tr><th>Trial</th><th>Stop reason</th><th>Final ratio</th><th>Seconds</th><th>Tokens</th></tr></thead>
<tbody>{table}</tbody></table><p>Missing usage is unknown, not zero. Failed and invalid-seed runs remain in the table.
See summary.csv, trajectory.csv and each trial's manifest, raw events and evaluations.</p></html>"""
    (root / "report.html").write_text(text)
    return root / "report.html"
