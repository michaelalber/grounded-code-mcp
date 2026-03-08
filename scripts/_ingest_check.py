"""Analyse which collections have untracked or misnamed files in the manifest.

Called by ingest-missing.sh. Outputs a JSON array to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

INGESTABLE = {
    ".pdf", ".epub", ".md", ".markdown", ".html", ".htm",
    ".rst", ".asciidoc", ".adoc", ".mdx", ".txt", ".docx", ".doc",
}


def main(project_dir: Path) -> None:
    config_path = project_dir / "config.toml"
    manifest_path = project_dir / ".grounded-code-mcp" / "manifest.json"

    # --- Config ---
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    collection_prefix = cfg.get("vectorstore", {}).get("collection_prefix", "grounded_")
    sources_dir = cfg.get("knowledge_base", {}).get("sources_dir", "sources")
    raw_collections: dict[str, str] = dict(cfg.get("collections", {}))

    # Merge user config collections
    user_cfg_path = Path.home() / ".config" / "grounded-code-mcp" / "config.toml"
    if user_cfg_path.exists():
        with open(user_cfg_path, "rb") as f:
            user_cfg = tomllib.load(f)
        raw_collections.update(user_cfg.get("collections", {}))

    # --- Manifest ---
    tracked: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        tracked = manifest.get("sources", {})

    results = []

    for src_dir_str, collection_suffix in raw_collections.items():
        src_dir = project_dir / src_dir_str
        if not src_dir.exists():
            continue

        full_collection = collection_prefix + collection_suffix

        # Files on disk — paths relative to sources_dir to match manifest keys
        sources_root = project_dir / sources_dir
        disk_files = {
            str(f.relative_to(sources_root))
            for f in src_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in INGESTABLE
        }

        # Manifest keys are relative to sources_dir (e.g. "internal/file.pdf")
        # Strip the leading sources_dir prefix from the config key for matching
        collection_subdir = Path(src_dir_str).relative_to(sources_dir)
        dir_prefix = str(collection_subdir).rstrip("/") + "/"
        manifest_entries = {
            path: info
            for path, info in tracked.items()
            if path.startswith(dir_prefix)
        }

        # Entries tracked under a different (stale) collection name
        wrong_collection = {
            path
            for path, info in manifest_entries.items()
            if info.get("collection") != full_collection
        }

        untracked = disk_files - set(manifest_entries.keys())

        if wrong_collection:
            status = "WRONG_COLLECTION"
        elif untracked:
            status = "NEEDS_INGEST"
        else:
            status = "ok"

        results.append({
            "src_dir":          src_dir_str,
            "collection":       full_collection,
            "disk":             len(disk_files),
            "tracked":          len(manifest_entries),
            "untracked":        len(untracked),
            "wrong_collection": len(wrong_collection),
            "status":           status,
        })

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _ingest_check.py <project_dir>", file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]))
