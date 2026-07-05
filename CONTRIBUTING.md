# Contributing to grounded-code-mcp

## The main extension point: adding a collection

A collection is a directory under `sources/` containing documents you trust — PDFs, Markdown files, HTML docs, EPUBs, DOCX files, or any mix. The server ingests them, chunks them, embeds them locally, and makes them searchable via the MCP tools.

### Step 1 — Create the source directory

```bash
mkdir sources/my-topic
```

Drop your documents in. Supported formats: PDF, DOCX, PPTX, HTML, Markdown, AsciiDoc, EPUB.

Subdirectories work — the ingester walks the tree recursively.

### Step 2 — Register the collection in `config.toml`

```toml
[collections]
"sources/my-topic" = "my_topic"
```

The key is the source path; the value is the collection name the MCP tools will use. The server prepends `grounded_` automatically — so `my_topic` becomes `grounded_my_topic` internally.

For **private collections** (not committed to the repo), add them to your personal config instead:

```toml
# ~/.config/grounded-code-mcp/config.toml
[collections]
"sources/my-private-docs" = "private"
```

### Step 3 — Ingest

```bash
# Ingest just the new collection
grounded-code-mcp ingest --collection my_topic

# Or re-ingest everything
grounded-code-mcp ingest
```

The ingester is incremental — unchanged files are skipped based on SHA-256 hashing. Use `--force` to re-ingest everything regardless.

> **Note:** Don't run multiple ingest jobs in parallel. Docling uses the GPU for PDF parsing — concurrent jobs cause CUDA out-of-memory errors.

### Step 4 — Verify

```bash
grounded-code-mcp status
grounded-code-mcp search "your topic" --collection my_topic
```

---

## What makes a good collection

Collections are most useful when they reflect a single coherent domain and come from sources you actually trust — books, official documentation, vetted standards. Avoid mixing unrelated topics in one collection; the semantic search quality degrades with noise.

**Good candidates:**
- A book you reference regularly (PDF)
- Official framework documentation (HTML or Markdown crawl)
- An internal engineering standards document

**Poor candidates:**
- Random blog posts scraped in bulk
- Draft or unreviewed documents
- Content duplicated across collections

---

## Changing the embedding model

All ingested collections must use the same embedding model. Changing it requires a full re-ingest:

```bash
# Update model in config.toml, then:
grounded-code-mcp ingest --force
```

Team members must coordinate — mixed embeddings from different models are not compatible.

---

## Development setup

```bash
git clone https://github.com/michaelalber/grounded-code-mcp.git
cd grounded-code-mcp
python3 -m venv .venv && .venv/bin/pip install -e ".[all,dev]"
```

Run tests:
```bash
.venv/bin/pytest
```

Lint and type-check:
```bash
.venv/bin/ruff check src/
.venv/bin/mypy src/
```

Use the `pipx`-installed binary for runtime commands (`grounded-code-mcp ingest`, `serve`, etc.). The dev venv is for tests and tooling only.

---

## Dependency notes

- `fastmcp >= 3.2.0` — v3.2.0 patched critical SSRF and command injection CVEs (CVE-2026-32871, CVE-2026-27124, CVE-2025-64340)
- `snowflake-arctic-embed2` — default embedding model; changing it requires a full re-ingest of all collections
