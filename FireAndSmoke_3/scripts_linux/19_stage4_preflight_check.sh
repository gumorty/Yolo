#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/gu/projects/FireAndSmoke_3}"
OUT="${OUT:-$PROJECT/reports/stage4_preflight_check.json}"

python "$PROJECT/tools/stage4_preflight_check.py" --project "$PROJECT" --out "$OUT"
