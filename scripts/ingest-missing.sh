#!/usr/bin/env bash
# ingest-missing.sh — detect and ingest collections with new or untracked files
#
# Usage (run from project root):
#   ./scripts/ingest-missing.sh          # dry-run: show what needs ingesting
#   ./scripts/ingest-missing.sh --run    # run missing ingests sequentially
#   ./scripts/ingest-missing.sh --force  # force re-ingest all collections
#
# Ingests run sequentially to avoid GPU OOM from Docling's PDF parser.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(pwd)"
LOG_DIR="/tmp/grounded-code-mcp-ingest"
ANALYSIS_FILE="$(mktemp /tmp/grounded-ingest-analysis.XXXXXX.json)"
trap 'rm -f "$ANALYSIS_FILE"' EXIT

# Use the pipx venv python (has tomli/tomllib); fall back to system python3
PYTHON="$(ls "$HOME/.local/pipx/venvs/grounded-code-mcp/bin/python" 2>/dev/null || echo "python3")"

DRY_RUN=true
FORCE=false

for arg in "$@"; do
  case "$arg" in
    --run)   DRY_RUN=false ;;
    --force) FORCE=true; DRY_RUN=false ;;
  esac
done

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Analyse which collections have untracked or misnamed files
# ---------------------------------------------------------------------------
"$PYTHON" "$SCRIPT_DIR/_ingest_check.py" "$PROJECT_DIR" > "$ANALYSIS_FILE"

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
echo ""
echo "grounded-code-mcp — ingest status"
echo "================================================================"
printf "%-34s %6s %6s %6s  %s\n" "Collection" "Disk" "Tracked" "Missing" "Status"
echo "----------------------------------------------------------------"

"$PYTHON" - "$ANALYSIS_FILE" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for r in data:
    flag = "✓" if r["status"] == "ok" else "→"
    print(f"{flag} {r['collection']:<32}  {r['disk']:>6} {r['tracked']:>6} {r['untracked']:>6}  {r['status']}")
PYEOF

echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Dry-run: just report
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run — pass --run to ingest, --force to force re-ingest all."
  echo ""
  "$PYTHON" - "$ANALYSIS_FILE" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
needs = [r for r in data if r["status"] != "ok"]
if needs:
    print("Action needed:")
    for r in needs:
        if r["status"] == "WRONG_COLLECTION":
            print(f"  {r['src_dir']:<28} → ingested under wrong collection name — run with --force")
        else:
            print(f"  {r['src_dir']:<28} → {r['untracked']} untracked files")
else:
    print("All collections are up to date.")
PYEOF
  exit 0
fi

# ---------------------------------------------------------------------------
# Build ordered list of dirs to ingest
# ---------------------------------------------------------------------------
FORCE_FLAG="$FORCE"
mapfile -t INGEST_DIRS < <("$PYTHON" - "$ANALYSIS_FILE" "$FORCE_FLAG" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
force = sys.argv[2] == "true"
for r in data:
    if force and r["disk"] > 0:
        print(r["src_dir"])
    elif r["status"] in ("NEEDS_INGEST", "WRONG_COLLECTION"):
        print(r["src_dir"])
PYEOF
)

if [[ ${#INGEST_DIRS[@]} -eq 0 ]]; then
  echo "Nothing to ingest — all collections are up to date."
  exit 0
fi

echo "Will ingest (sequentially):"
for dir in "${INGEST_DIRS[@]}"; do
  echo "  $dir"
done
echo ""

# Abort if another ingest is already running
if pgrep -f "grounded-code-mcp ingest" > /dev/null 2>&1; then
  echo "WARNING: an ingest process is already running:"
  pgrep -fa "grounded-code-mcp ingest"
  echo ""
  echo "Wait for it to finish before starting another (GPU OOM risk)."
  exit 1
fi

# ---------------------------------------------------------------------------
# Run ingests sequentially
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
TOTAL=${#INGEST_DIRS[@]}
IDX=0

for src_dir in "${INGEST_DIRS[@]}"; do
  IDX=$((IDX + 1))
  LOG="$LOG_DIR/$(basename "$src_dir")-$(date +%Y%m%d-%H%M%S).log"
  echo "[$IDX/$TOTAL] Ingesting $src_dir ..."
  echo "  Log: $LOG"

  if [[ "$FORCE" == "true" ]]; then
    grounded-code-mcp ingest --force "$src_dir" 2>&1 | tee "$LOG" | \
      grep -E "✓|✗|Scanned|Skipped|Chunks|Error|failed" || true
  else
    grounded-code-mcp ingest "$src_dir" 2>&1 | tee "$LOG" | \
      grep -E "✓|✗|Scanned|Skipped|Chunks|Error|failed" || true
  fi

  echo "  Done."
  echo ""
done

echo "All ingests complete."
