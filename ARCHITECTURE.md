# grounded-code-mcp — Architecture Summary

> Phase 1 pre-implementation reference. Written read-only before any graph code exists.

---

## Directory Structure

```
grounded-code-mcp/
├── src/grounded_code_mcp/   # Production source (installed via hatchling)
│   ├── __main__.py          # Click CLI entry point
│   ├── server.py            # FastMCP server + tool handlers
│   ├── ingest.py            # Ingestion pipeline orchestrator
│   ├── chunking.py          # Code-aware document chunker
│   ├── config.py            # Pydantic v2 settings (TOML-based)
│   ├── embeddings.py        # Ollama embedding client
│   ├── manifest.py          # Ingest manifest (JSON, Pydantic v2)
│   ├── manifest_repair.py   # Manifest repair utilities
│   ├── parser.py            # Docling document parser
│   └── vectorstore.py       # VectorStore ABC + Qdrant/ChromaDB impls
├── tests/                   # pytest test suite
├── sources/                 # Knowledge base documents by collection
│   └── <collection>/        # e.g. sources/rust/, sources/python/
├── .grounded-code-mcp/      # Runtime data (gitignored)
│   ├── qdrant/              # Qdrant persistent storage
│   └── manifest.json        # Ingest manifest
├── config.toml              # Committed project configuration
├── config.toml.example      # User config template
└── pyproject.toml           # hatchling build; ruff, mypy, bandit, pytest config
```

---

## Server Entry Point

`server.py` — FastMCP server instance created at module level:

```python
mcp = FastMCP("grounded-code-mcp")
```

**Tool registration pattern:** Each MCP tool follows a two-function split:
- `_xxx_impl(...)` — pure implementation, no decorator, fully testable
- `@mcp.tool()` wrapper — thin delegator, not tested directly

**Five registered tools:** `search_knowledge`, `search_code_examples`, `list_collections`, `list_sources`, `get_source_info`

**Global state:** `_settings`, `_embedder`, `_manifest` — lazy-initialized via `initialize()` with a threading lock. Manifest hot-reloads on mtime change (supports concurrent CLI ingest + running server).

**Startup:** `run_server()` calls `initialize()` then `mcp.run(transport=..., host=..., port=...)`. Default transport: stdio. HTTP transport always binds `127.0.0.1`.

---

## Ingest Pipeline

`ingest_documents(settings, path, collection, force)` → `IngestionPipeline.ingest()`

```
scan_directory(path)
  → parser.parse(file)          # Docling → ParsedDocument (content: str, title, page_count)
  → chunker.chunk(content)      # list[Chunk]
  → embedder.embed_many(texts)  # list[EmbeddingResult] (batched, batch_size from config)
  → store.add_chunks(collection, chunks, embeddings)
  → manifest.add_entry(SourceEntry)
```

**Change detection:** SHA-256 hash in `manifest.json`. Skip if unchanged unless `--force`.

**Re-ingest:** `store.delete_chunks(old_chunk_ids)` before re-parsing.

**Memory management:** Batch streaming — embed `ingest_batch_size` chunks at a time (default 50), store, release.

**Collection resolution:** `Settings.get_collection_name(path)` matches source path against `[collections]` table prefix map → prepends `grounded_`. Fallback: first path component under `sources_dir`.

---

## Qdrant Collection Schema

Collection name format: `grounded_<suffix>` (e.g., `grounded_rust`)

Vector config:
- Dimension: 1024 (snowflake-arctic-embed2)
- Distance: COSINE

Payload fields per point:

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Chunk text |
| `source_path` | `str` | Relative path (e.g. `rust/the-rust-book.md`) |
| `chunk_index` | `int` | Position within source document |
| `heading_context` | `list[str]` | Heading breadcrumb trail |
| `is_code` | `bool` | True for fenced code blocks |
| `code_language` | `str\|None` | Language tag from fence |
| `is_table` | `bool` | True for Markdown tables |

Point IDs: UUID4 strings (random, not deterministic — re-ingest creates new IDs).

---

## Config Mechanism

`Settings.load()` merges two TOML layers (later overrides earlier):

1. `./config.toml` — committed, shared
2. `~/.config/grounded-code-mcp/config.toml` — user overrides (host, URL, private collections)

`[collections]` tables are deep-merged (union), not replaced.

No env var support in the existing config system — settings are TOML-only. The `GRAPH_JSON_PATH` env var (new, for the graph module) will be handled directly in `graph_store.py` via `os.environ.get()`.

---

## CLI Entry Points

All via `grounded_code_mcp/__main__.py` (Click group `cli`):

| Command | Key flags | Notes |
|---|---|---|
| `ingest [PATH]` | `--force`, `--collection` | `force=True` skips hash check |
| `convert [PATH]` | `--force`, `--dry-run`, `--no-ocr`, `--collection` | Spawns subprocesses per file |
| `serve` | `--transport`, `--host`, `--port`, `--debug` | |
| `search QUERY` | `--collection`, `-n`, `--min-score` | |
| `status` | — | |

Source targeting: `ingest PATH` + `--collection NAME` overrides auto-resolution. Without `--collection`, `Settings.get_collection_name(path)` drives it.

---

## Source Metadata Available at Ingest Time

`SourceEntry` (manifest):
- `path`: relative path from `sources_dir` (e.g., `rust/the-rust-book.md`)
- `sha256`, `collection`, `file_type`, `title`, `page_count`, `chunk_count`, `chunk_ids[]`, `ingested_at`

`Chunk` (in-memory only, not persisted beyond vector store payload):
- `chunk_id`, `content`, `chunk_index`, `start_char`, `end_char`
- `heading_context: list[str]`, `is_code`, `code_language`, `is_table`
- `source_path: str`

---

## Test Patterns

- One test file per module: `tests/test_<module>.py`
- Classes group related cases: `class TestXxx:` + `def test_<behavior>(self) -> None:`
- Fixtures in `conftest.py`: `temp_dir` (tmp Path), `sample_markdown_content`, `sample_config_toml`
- External services mocked (Ollama, Qdrant uses `:memory:` client)
- Integration tests marked `@pytest.mark.integration`
- No `assert` suppression (`S101` ignored for tests only via ruff per-file-ignores)

---

## Graph Module Placement Proposal

New code: `src/graph/__init__.py`, `src/graph/graph_store.py`, `src/graph/graph_builder.py`

Rationale for `src/graph/` (not project-root `graph/`):
- Consistent with existing convention (`src/grounded_code_mcp/`)
- Works with the existing editable install (`pip install -e ".[all,dev]"`) — `src/` is on `sys.path`
- `python -m graph.graph_builder` works out of the box post-editable install

Required change: add `"src/graph"` to `[tool.hatch.build.targets.wheel].packages` in `pyproject.toml`.

Tests: `tests/test_graph_store.py`, `tests/test_graph_builder.py` — same class/method naming conventions as existing tests.

`networkx` must be added to `pyproject.toml` `dependencies` (or `[project.optional-dependencies].graph`).

---

## Open Questions for Confirmation

1. **`graph/` placement**: `src/graph/` (consistent with existing layout) vs. project-root `graph/` (matches spec paths literally). I'll proceed with `src/graph/` unless you prefer the root.
2. **`networkx` dependency**: Add to main `dependencies` or a new `[graph]` optional extra?
3. **`sources/distilled/` directory**: The spec CLI references `./sources/distilled/tidy-first/`. Does this directory exist or will it be created as part of a later phase? (graph_builder only reads RELATIONSHIPS.md files from it — no impact on Phase 1 code, but affects test fixture design.)
