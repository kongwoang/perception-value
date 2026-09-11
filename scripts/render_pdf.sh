#!/bin/bash
# Render a report to PDF. Firefox headless is unusable on this board (no X server,
# RenderCompositorSWGL fails and it hangs), so WeasyPrint is the path:
#   pip install weasyprint pydyf==0.8.0     # newer pydyf changes PDF() and breaks 61.2
# WeasyPrint has no CSS grid, no *-block shorthands and no color-mix(); the report
# stylesheets carry a flexbox fallback layer for it.
set -euo pipefail
HTML=${1:?usage: render_pdf.sh <report.html> [out.pdf]}
OUT=${2:-${HTML%.html}.pdf}
DIR=$(cd "$(dirname "$HTML")" && pwd)
mkdir -p "$DIR/figs" && cp -f "$(dirname "$0")/../results/figures/"*.png "$DIR/figs/" 2>/dev/null || true
cd "$DIR" && /home/kongwoang/miniforge3/envs/edge/bin/python -c "
import sys; from weasyprint import HTML
d = HTML(filename=sys.argv[1], base_url='.').render()
print('pages:', len(d.pages)); d.write_pdf(sys.argv[2])
" "$(basename "$HTML")" "$OUT"
echo "wrote $OUT"
