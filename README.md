# Grounded Code MCP

[![CI](https://github.com/michaelalber/grounded-code-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelalber/grounded-code-mcp/actions/workflows/ci.yml)
[![Security](https://github.com/michaelalber/grounded-code-mcp/actions/workflows/security.yml/badge.svg)](https://github.com/michaelalber/grounded-code-mcp/actions/workflows/security.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Ground your AI coding agent in the books, standards, and docs you actually trust.**

A local [MCP](https://modelcontextprotocol.io/) server that gives Claude Code, OpenCode, and other AI coding assistants retrieval access to your personal knowledge base — your books, standards documents, official docs, and curated references. Instead of generating answers from training data alone, the agent searches your sources first.

---

## Why I built this

AI coding assistants produce more useful output when the context they reason over matches your actual standards — not averaged training data. I work across .NET, Python, Rust, edge AI, and federal security domains. Each has authoritative sources I trust: specific books, NIST standards, official framework docs, internal engineering guidelines.

This project makes those sources searchable by any MCP-compatible agent. The agent queries the knowledge base before responding, grounding its answers in sources I've explicitly chosen. The result is output that reflects my preferences and standards, not a generic average.

**Key design decisions:**
- Fully local — embeddings run via Ollama, vectors stored in Qdrant. No data leaves the machine.
- Curated over comprehensive — 21 domain collections, each representing a deliberate choice of what to trust.
- Incremental — SHA-256 change detection means re-ingestion only processes what changed.
- Layered config — shared project config + per-machine user overrides, deep-merged at startup.

---

## Architecture

```
Documents (PDF, DOCX, HTML, MD, EPUB…)   RELATIONSHIPS.md files
        │                                         │
        ▼ (optional — run once on a GPU machine)  │
   [ convert → .md sidecars ]                     │
        │                                         │
        ▼                                         ▼
   [ Docling Parser ]                    [ GraphBuilder parser ]
        │                                         │
        ▼                                         │
   [ Semantic Chunker ]                           │
        │                                         │
        ▼                                         ▼
   [ Ollama Embedder ]               [ NetworkX DiGraph (JSON) ]
        │                                         │
        ▼                                         │
   [ Qdrant / ChromaDB ]─────────────────────────┘
        │                    vector + graph
        ▼
   [ FastMCP Server ]          ← 5 MCP tools: search_knowledge, search_code_examples,
        │                         list_collections, list_sources, get_source_info
        ▼
Claude Code / OpenCode / any MCP client
```

The pipeline has three separate processes. `convert` is a one-time GPU step that produces Markdown sidecars. `ingest` reads those sidecars (or parses documents directly when no sidecar exists) and upserts chunks into the vector store. `graph_builder` parses `RELATIONSHIPS.md` files from knowledge sources and builds a persistent concept graph. The MCP server runs as a persistent subprocess managed by the MCP client.

---

## Features

- **Concept graph (Graph RAG)** — a NetworkX DiGraph extracted from `RELATIONSHIPS.md` files in each knowledge source; captures named relationships between concepts and enables graph-traversal-augmented retrieval alongside vector search
- **Multi-format ingestion** — PDF, DOCX, PPTX, HTML, Markdown, AsciiDoc, EPUB via Docling
- **GPU-accelerated pre-convert** — `convert` command processes binary documents to Markdown sidecars; subsequent `ingest` runs read the sidecar and skip Docling entirely, making ingest CPU-only and fast after the first pass
- **Crash-isolated batch conversion** — each file is converted in its own subprocess; a Docling PDF crash doesn't abort the whole batch
- **Code-aware chunking** — preserves code blocks, tables, and heading hierarchy
- **Local embeddings** — Ollama with snowflake-arctic-embed2 (1024 dimensions, 8K context)
- **Dual vector store** — Qdrant (primary) or ChromaDB (Docker-free fallback)
- **Incremental updates** — SHA-256 hashing skips unchanged files
- **18 curated collections** — covering .NET, Python, Rust, architecture, security, AI/ML, edge, robotics, and more
- **Private collections** — add your own sources via user config without touching the project
- **Layered configuration** — project `config.toml` deep-merged with `~/.config/grounded-code-mcp/config.toml`

---

## Prerequisites

**Ollama** — runs the embedding model locally:

```bash
ollama serve
ollama pull snowflake-arctic-embed2
```

**Qdrant** — vector store (recommended):

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

ChromaDB is supported as a Docker-free fallback (`provider = "chromadb"` in config).

---

## Installation

**Production (pipx — recommended):**

```bash
pipx install git+https://github.com/michaelalber/grounded-code-mcp.git
```

**Development:**

```bash
git clone https://github.com/michaelalber/grounded-code-mcp.git
cd grounded-code-mcp
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Dev tools (`pytest`, `ruff`, `mypy`) run from `.venv/bin/`. Use the pipx binary for all runtime commands.

---

## Configuration

`config.toml` (committed to the repo) defines shared settings. Machine-specific overrides go in `~/.config/grounded-code-mcp/config.toml` — deep-merged at startup.

```bash
cp config.toml.example ~/.config/grounded-code-mcp/config.toml
```

Minimal user config:

```toml
[ollama]
host = "http://localhost:11434"

[vectorstore]
qdrant_url = "http://localhost:6333"
```

**Private collections** — add sources without touching the project config:

```toml
# ~/.config/grounded-code-mcp/config.toml
[collections]
"sources/my-team-docs" = "team_docs"
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full collection workflow.

---

## Usage

### Pre-convert documents (GPU-accelerated)

`convert` runs Docling on binary sources (PDF, DOCX, EPUB, PPTX) and writes a `.md` sidecar file next to each source. When a sidecar exists, `ingest` reads it directly and skips the Docling step — making repeated ingestion fast and CPU-only.

```bash
grounded-code-mcp convert                          # all collections
grounded-code-mcp convert --collection rust        # one collection
grounded-code-mcp convert path/to/file.pdf         # single file
grounded-code-mcp convert --force                  # re-convert even if sidecar already exists
grounded-code-mcp convert --dry-run                # list files that would be converted
grounded-code-mcp convert --no-ocr                 # disable OCR regardless of config
```

> Run `convert` before `ingest` on GPU machines. Each file is converted in an isolated subprocess — a Docling PDF crash doesn't abort the whole batch.

**Optional: Flash Attention 2** — on Ampere+ GPUs with the CUDA toolkit installed:

```bash
pip install flash-attn --no-build-isolation
```

Then enable in `~/.config/grounded-code-mcp/config.toml`:

```toml
[docling]
cuda_use_flash_attention2 = true
```

### Ingest documents

```bash
grounded-code-mcp ingest                        # all collections
grounded-code-mcp ingest --collection python    # one collection
grounded-code-mcp ingest --force                # ignore manifest, re-ingest everything
```

> Avoid parallel ingest jobs — Docling uses the GPU when no sidecar is present; concurrent jobs cause CUDA OOM.

### Check status

```bash
grounded-code-mcp status
```

### Search from the CLI

All search commands accept `--json` to emit machine-readable output — useful for shell scripts and AI agents that can't use MCP.

**Prose search:**

```bash
grounded-code-mcp search "async HTTP request"
grounded-code-mcp search "dependency injection" --collection patterns
grounded-code-mcp search "error handling" -n 10 --min-score 0.4
grounded-code-mcp search "CQRS" --collection architecture --json   # JSON output
```

**Code example search:**

```bash
grounded-code-mcp search-code "async context manager" --language python
grounded-code-mcp search-code "repository pattern" --language csharp -n 3 --json
```

**List and inspect sources:**

```bash
grounded-code-mcp list-sources                         # all collections
grounded-code-mcp list-sources --collection python     # one collection
grounded-code-mcp list-sources --json                  # JSON output

grounded-code-mcp source-info sources/python/cosmicpython.pdf
grounded-code-mcp source-info sources/python/cosmicpython.pdf --json
```

**Query the concept graph:**

```bash
grounded-code-mcp query-graph CQRS
grounded-code-mcp query-graph "clean architecture" --depth 2 --domain patterns
grounded-code-mcp query-graph CQRS --json
```

### Start the MCP server

```bash
grounded-code-mcp serve                                                     # stdio (default)
grounded-code-mcp serve --transport streamable-http --host 127.0.0.1 --port 4242
grounded-code-mcp serve --debug
```

### Connect to MCP clients

**Claude Code:**

```bash
claude mcp add --transport stdio --scope user grounded-code-mcp -- grounded-code-mcp serve
```

**OpenCode** (`~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "grounded-code-mcp": {
      "type": "local",
      "command": ["grounded-code-mcp", "serve"],
      "enabled": true
    }
  }
}
```

**Pi.dev** — see [Pi.dev extension](#pidev-extension) below.

---

## MCP Tools

Five tools are exposed to the agent. Pass the bare collection suffix — the server prepends `grounded_` automatically.

### `search_knowledge`

Search documentation across all collections or within a specific one.

```python
search_knowledge(
    query: str,                  # search query — 2–6 content words work best
    collection: str | None = None,  # bare suffix, e.g. "python", "rust", "internal"
    n_results: int = 5,
    min_score: float = 0.3,      # 0–1; raise to 0.5+ for tighter relevance
) -> list[dict]
```

### `search_code_examples`

Finds code-heavy chunks — useful when you want implementation patterns rather than prose.

```python
search_code_examples(
    query: str,                  # e.g. "async HTTP client", "repository pattern"
    language: str | None = None, # e.g. "python", "csharp", "rust"
    n_results: int = 5,
) -> list[dict]
```

### `list_collections`

```python
list_collections() -> list[dict]  # returns name + document count per collection
```

### `list_sources`

```python
list_sources(
    collection: str | None = None,  # optional filter
) -> list[dict]  # returns path, type, chunk count per source
```

### `get_source_info`

```python
get_source_info(
    source_path: str,  # path returned by list_sources
) -> dict  # title, type, chunks, ingestion date
```

---

## Pi.dev Extension

A TypeScript extension for [pi.dev](https://pi.dev) that exposes the knowledge base as five searchable tools. Each tool runs `grounded-code-mcp <subcommand> --json` as a subprocess and returns parsed JSON to pi's context — no MCP required, fully local.

### Installation

**Option A — local path (simplest)**

Add the extension directory to `~/.pi/settings.json`:

```json
{
  "extensions": [
    "/path/to/grounded-code-mcp/skill/extensions"
  ]
}
```

**Option B — test before installing**

```bash
pi -e /path/to/grounded-code-mcp/skill/extensions/index.ts
```

**Option C — git package (from inside pi)**

```
/install git:codeberg.org/michaelkalber/grounded-code-mcp?path=skill
```

### Tools

| Tool | Description |
|------|-------------|
| `grounded_search` | Vector search across all (or one) collection — returns prose chunks with score and source path |
| `grounded_search_code` | Code-block-only search with optional language filter |
| `grounded_list_sources` | Lists every ingested document — use to discover what's available |
| `grounded_source_info` | Metadata for a specific source: chunk count, SHA-256, ingestion date |
| `grounded_query_graph` | Graph traversal — finds concept relationships and linked sources |

### Example usage in pi

```
Search for FastAPI dependency injection patterns
→ grounded_search(query="dependency injection", collection="python")

Find Python async context manager examples
→ grounded_search_code(query="async context manager", language="python")

What documentation is indexed?
→ grounded_list_sources()

How does CQRS relate to clean architecture?
→ grounded_query_graph(concept="CQRS", depth=2)
```

Pass the bare collection suffix — the server prepends `grounded_` automatically.

---

## Concept Graph (Graph RAG)

Alongside vector embeddings, grounded-code-mcp builds a concept graph from `RELATIONSHIPS.md` files in each knowledge source. The graph is a directed NetworkX `DiGraph` persisted as JSON — it captures named relationships between concepts and enables graph-traversal-augmented retrieval: find related concepts by walking the graph, not just by cosine distance.

### RELATIONSHIPS.md formats

Two formats are supported in the same file:

**Quoted format** (general purpose):
```
"Concept A" → enables → "Concept B" [source-slug] [domain] [type] [optional description]
```

**Parenthetical format** (used by distilled sources, predicates normalised to lowercase):
```
(Concept A) --[PREDICATE]--> (Concept B)
```

Triples may appear as bare lines or inside fenced code blocks. Any relation name is accepted — there is no fixed allowlist.

### Building the graph

```bash
# Validate without writing
python -m graph.graph_builder --input sources/ --dry-run

# Build and persist (default: graph/concept_graph.json)
python -m graph.graph_builder --input sources/

# Point to a specific output file
GRAPH_JSON_PATH=/path/to/graph.json python -m graph.graph_builder --input sources/
```

Each run is idempotent: nodes for a source are replaced before new ones are inserted, so re-running on the same input produces the same result.

---

## Graph RAG — CLI Reference

All commands use the `grounded-code-mcp` binary installed via `pipx`. After any code change, reinstall with `pipx install . --force`.

```bash
# Full reingest — rebuilds Qdrant vectors and concept graph for all sources
grounded-code-mcp ingest --force

# Single source reingest + graph rebuild (targets one subdirectory)
grounded-code-mcp ingest --force sources/rust

# Full graph rebuild from all RELATIONSHIPS.md files (no reingest)
grounded-code-mcp build-graph

# Graph rebuild for a single source directory (no reingest)
grounded-code-mcp build-graph sources/rust

# Validate graph triples without writing (dry run, full sources)
grounded-code-mcp build-graph --dry-run

# Dry run for a single source
grounded-code-mcp build-graph --dry-run sources/rust

# Seed starter RELATIONSHIPS.md files for sources that are missing them
python -m graph.seed_graph

# Seed a single source by slug
python -m graph.seed_graph --source rust

# Dry run — preview what seed_graph would generate without writing
python -m graph.seed_graph --dry-run

# Direct graph query (CLI, not MCP) — explore the graph from the shell
python -m graph.graph_builder --input sources/ --dry-run
```

**Env var override** — point the graph to a non-default location:
```bash
GRAPH_JSON_PATH=/path/to/graph.json grounded-code-mcp build-graph
```

**MCP tool** — query the graph from an AI assistant session:
```
query_graph(concept="cqrs", depth=2, domain="architecture")
```
Returns matched nodes, relationships (triples), linked source slugs, and a plain-English summary of the concept's neighbourhood.

---

## Collections

17 curated collections covering the domains I work in. Each maps a `sources/` subdirectory to a collection name.

| Directory | Collection | What belongs here |
|-----------|-----------|-------------------|
| `sources/internal` | `internal` | Engineering standards — XP, TDD, CI/CD, DDD, OWASP, NIST AI |
| `sources/patterns` | `patterns` | Design patterns — GoF, CQRS, Clean Architecture, DI |
| `sources/architecture` | `architecture` | Software architecture — DDIA, SRE, 12-Factor, C4, arc42 |
| `sources/systems-thinking` | `systems_thinking` | Systems thinking — Meadows, feedback loops, chaos engineering |
| `sources/ui-ux` | `ui_ux` | Laws of UX, Nielsen, WCAG 2.2, ARIA, USWDS, GOV.UK |
| `sources/dotnet` | `dotnet` | .NET/C#, ASP.NET Core, Entity Framework, Telerik UI |
| `sources/python` | `python` | Python, FastAPI, Pydantic, FastMCP, pytest, cosmicpython |
| `sources/databases` | `databases` | SQL, PostgreSQL, relational theory |
| `sources/edge-ai` | `edge_ai` | AI engineering, RAG, embeddings, LLM application design, AI agents |
| `sources/automation` | `automation` | PLC, OPC UA, MODBUS, ICS security, Raspberry Pi |
| `sources/4d-legacy` | `4d_legacy` | 4D platform — source reference for 4D → .NET migration |
| `sources/php` | `php` | PHP manual, Laravel (5.5 / 6.x / 12.x) |
| `sources/javascript` | `javascript` | JS/TS, Vue 2/3, jQuery, ECMAScript spec |
| `sources/gov` | `gov` | NIST 800-53/171/218, DOE, Zero Trust, AI RMF, CUI |
| `sources/robotics` | `robotics` | ROS 2, MuJoCo, Isaac Lab, LeRobot, VLA models |
| `sources/rust` | `rust` | Rust ownership, async/Tokio, Cargo, error handling, Axum |
| `sources/api-design` | `api_design` | REST API design — Zalando, Google AIP, Microsoft guidelines |

Add private collections in `~/.config/grounded-code-mcp/config.toml` — they merge with the project list, not replace it.

---

## Development

```bash
.venv/bin/pytest                                      # run tests
.venv/bin/pytest --cov=grounded_code_mcp              # with coverage
.venv/bin/ruff check src/ tests/                      # lint
.venv/bin/ruff format --check src/ tests/             # format check
.venv/bin/mypy src/                                   # type check
.venv/bin/bandit -r src/ -c pyproject.toml            # security scan
```

All gates at once:

```bash
.venv/bin/pytest && .venv/bin/ruff format --check src/ tests/ && .venv/bin/ruff check src/ tests/ && .venv/bin/mypy src/ && .venv/bin/bandit -r src/ -c pyproject.toml
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the collection workflow and dependency notes.

---

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| MCP Framework | FastMCP `>=3.2.0` | |
| Document Parsing | Docling | Layout-aware; handles complex PDFs |
| Vector Store | Qdrant / ChromaDB | Qdrant primary; ChromaDB as Docker-free fallback |
| Concept Graph | NetworkX DiGraph | Persisted as JSON; supports BFS traversal, path-finding, domain/source filtering |
| Embeddings | Ollama + snowflake-arctic-embed2 | 1024-dim, 8K context, fully local |
| Configuration | TOML + Pydantic | Deep-merged layered config |
| CLI | Click + Rich | |
| Testing | pytest | 398 tests |
| Linting | ruff | |
| Type Checking | mypy | |
| Security Scan | bandit | |

---

## Security

- File type validation via allowlist (PDF, DOCX, PPTX, HTML, Markdown, AsciiDoc, EPUB)
- MIME type verification via magic bytes or UTF-8 validation
- File size limits (configurable; default 500 MB to support large vendor PDFs)
- Filename sanitization — path traversal prevention
- All inputs validated at system boundaries
- Dependency vulnerability scanning in CI via `pip-audit`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Ollama connection error` | `ollama serve` + `curl http://localhost:11434/api/tags` to verify |
| `Qdrant connection error` | `curl http://localhost:6333/healthz` to verify container is running |
| Ingestion OOM / GPU crash | Run one ingest at a time — parallel Docling jobs exhaust VRAM; or run `convert` first so ingest is CPU-only |
| `convert` fails on a specific file | Each file runs in an isolated subprocess; stderr shows the reason; re-run with the file path alone to debug |
| `convert` slow without GPU speedup | Install `flash-attn` and set `cuda_use_flash_attention2 = true` in `[docling]` (Ampere+ only) |
| Search returns no results | `grounded-code-mcp status` to verify ingestion; try `--min-score 0.3` |
| Low relevance scores | Pass a bare collection suffix, not the full `grounded_*` name |

---

## Author

**Michael K. Alber** — [github.com/michaelalber](https://github.com/michaelalber)

Software engineer working across .NET, Python, Rust, edge AI, and federal security domains. I build tools that make AI-assisted development more grounded, more opinionated, and more aligned with engineering standards that matter.

Related projects:
- [ai-toolkit](https://github.com/michaelalber/ai-toolkit) — skills, agents, and slash commands for Claude Code, OpenCode, and Pi

---

## License

MIT — see [LICENSE](LICENSE) for details.
