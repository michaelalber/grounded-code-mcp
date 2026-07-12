#!/usr/bin/env bash
# Mirror sources/ from the remote knowledge-base host down to this machine.
#
# Configure the remote via environment variables (keep machine-specific values
# out of version control — e.g. export them in your shell profile or an
# untracked scripts/.sync-sources.env that you `source` before running):
#   export GKB_REMOTE_HOST=your-kb-host        # required — host or IP of the KB machine
#   export GKB_REMOTE_USER=youruser            # optional — defaults to $USER
#   export GKB_REMOTE_PATH=~/path/to/sources/  # optional — defaults to the repo layout
#
# Usage:
#   scripts/sync-sources.sh            # dry run — shows what would change, changes nothing
#   scripts/sync-sources.sh --apply    # performs the sync (mirrors, deletes local-only files)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load machine-specific config if present (gitignored — see header).
ENV_FILE="$SCRIPT_DIR/.sync-sources.env"
# shellcheck source=/dev/null
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

REMOTE_USER="${GKB_REMOTE_USER:-$USER}"
REMOTE_HOST="${GKB_REMOTE_HOST:?set GKB_REMOTE_HOST to your knowledge-base host (see header)}"
REMOTE_PATH="${GKB_REMOTE_PATH:-~/AppDev/michaelalber/codeberg/grounded-code-mcp/sources/}"
LOCAL_PATH="$REPO_ROOT/sources/"

RSYNC_OPTS=(-avz --delete)

if [[ "${1:-}" != "--apply" ]]; then
  RSYNC_OPTS+=(--dry-run)
  echo "Dry run — no files will be changed or deleted. Pass --apply to sync for real."
fi

rsync "${RSYNC_OPTS[@]}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
