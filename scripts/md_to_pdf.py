#!/usr/bin/env python
"""Render a findings markdown file to PDF with the project's report stylesheet.

Firefox headless is unusable on this board (no X server; RenderCompositorSWGL fails and
the process hangs), so WeasyPrint does the rendering. WeasyPrint 61 has no CSS grid, no
logical `*-block` shorthands and no `color-mix()`, so the stylesheet below stays on
flexbox and longhand properties.

    pip install markdown weasyprint pydyf==0.8.0   # newer pydyf breaks weasyprint 61.2
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
:root {
  --ink:#141820; --ink-2:#414b59; --muted:#6d7787;
  --rule:#d8dee7; --rule-strong:#b6bfcd; --surface-2:#f1f4f8;
  --accent:#1f66bd; --accent-soft:#eaf1fb; --warn:#b8471a; --warn-soft:#fcefe8;
  --sans:"IBM Plex Sans",sans-serif; --serif:"Source Serif 4",Georgia,serif;
  --mono:"IBM Plex Mono",monospace;
}
body { background:#fff; color:var(--ink); font-family:var(--serif);
       font-size:10.1pt; line-height:1.5; margin:0; }
p,ul,ol,blockquote,figcaption { max-width:62em; }
h1 { font-family:var(--sans); font-weight:700; font-size:23pt; line-height:1.12;
     letter-spacing:-.02em; margin:0 0 14px; max-width:22ch; }
h2 { font-family:var(--sans); font-size:13.5pt; font-weight:600; letter-spacing:-.012em;
     margin:26px 0 8px; padding-top:12px; border-top:2px solid var(--ink);
     break-after:avoid; }
h3 { font-family:var(--sans); font-size:10.5pt; font-weight:600; margin:16px 0 7px;
     break-after:avoid; }
p { margin:0 0 11px; }
ul,ol { padding-left:20px; margin:0 0 11px; } li { margin-bottom:6px; }
strong { font-weight:600; }
em { font-style:italic; }
code { font-family:var(--mono); font-size:.86em; background:var(--surface-2);
       padding:1px 4px; border-radius:2px; }
blockquote { border-left:3px solid var(--accent); background:var(--accent-soft);
             margin:16px 0; padding:12px 16px; break-inside:avoid; }
blockquote p:last-child { margin-bottom:0; }
table { border-collapse:collapse; font-family:var(--sans); font-size:8.5pt; width:100%;
        margin:14px 0; break-inside:avoid; }
th,td { text-align:left; padding:5px 12px 5px 0; border-bottom:1px solid var(--rule);
        vertical-align:baseline; }
/* Greek letters survive text-transform as look-alike capitals (rho -> "P" beside an
   actual p-value column), so headers keep their case. */
thead th { font-size:8.4pt; font-weight:600; letter-spacing:.02em; color:var(--muted);
           border-bottom:1px solid var(--rule-strong); }
tbody tr:last-child td { border-bottom:none; }
td:nth-child(n+2), th:nth-child(n+2) { font-variant-numeric:tabular-nums; }
img { display:block; width:100%; max-width:100%; height:auto; border:1px solid var(--rule);
      margin:16px 0; break-inside:avoid; }
hr { border:none; border-top:1px solid var(--rule); margin:22px 0; }
a { color:var(--accent); text-decoration:none; }
"""

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{css}</style></head><body>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--out", default="")
    ap.add_argument("--figures", default="results/figures",
                    help="comma-separated figure directories; later ones win on name clash")
    args = ap.parse_args()

    import markdown
    from weasyprint import HTML

    src = Path(args.md)
    text = src.read_text()
    # the markdown links figures relative to docs/; rewrite to the render directory
    # figures may live in several directories; the markdown references them by basename
    text = re.sub(r"\]\((?:\.\./)*results/(?:final/)?figures/", "](figs/", text)
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), src.stem)

    html = markdown.markdown(text, extensions=["tables", "attr_list"])
    doc = HEAD.format(title=title, css=CSS) + html + "\n</body></html>"

    work = Path("/tmp/rap_render")
    (work / "figs").mkdir(parents=True, exist_ok=True)
    for d in args.figures.split(","):
        for p in Path(d.strip()).glob("*.png"):
            (work / "figs" / p.name).write_bytes(p.read_bytes())
    (work / "doc.html").write_text(doc)

    out = Path(args.out) if args.out else src.with_suffix(".pdf")
    rendered = HTML(filename=str(work / "doc.html"), base_url=str(work)).render()
    print(f"pages: {len(rendered.pages)}")
    rendered.write_pdf(str(out))
    print("wrote", out)


if __name__ == "__main__":
    main()
