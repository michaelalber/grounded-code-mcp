#!/usr/bin/env bash
# Mirror sources/ from the remote knowledge-base host down to this machine.
#
# Usage:
#   scripts/sync-sources.sh            # dry run — shows what would change, changes nothing
#   scripts/sync-sources.sh --apply    # performs the sync (mirrors, deletes local-only files)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

REMOTE_USER="malber"
REMOTE_HOST="your-kb-host"
REMOTE_PATH="~/AppDev/michaelalber/codeberg/grounded-code-mcp/sources/"
LOCAL_PATH="$REPO_ROOT/sources/"

RSYNC_OPTS=(-avz --delete)

if [[ "${1:-}" != "--apply" ]]; then
  RSYNC_OPTS+=(--dry-run)
  echo "Dry run — no files will be changed or deleted. Pass --apply to sync for real."
fi

rsync "${RSYNC_OPTS[@]}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
