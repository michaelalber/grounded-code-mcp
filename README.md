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
Documents (PDF, DOCX, HTML, MD, EPUB…)
        │
        ▼ (optional — run once on a GPU machine)
   [ convert → .md sidecars ] ← GPU-accelerated Docling; writes foo.pdf.md next to each source
        │
        ▼
   [ Docling Parser ]          ← layout-aware extraction; skipped automatically when sidecar exists
        │
        ▼
   [ Semantic Chunker ]        ← heading-hierarchy-aware; code-block boundaries respected
        │
        ▼
   [ Ollama Embedder ]         ← snowflake-arctic-embed2, 1024-dim, fully local
        │
        ▼
   [ Qdrant / ChromaDB ]       ← persistent vector store; SHA-256 manifest for incremental updates
        │
        ▼
   [ FastMCP Server ]          ← 5 MCP tools: search_knowledge, search_code_examples,
        │                         list_collections, list_sources, get_source_info
        ▼
Claude Code / OpenCode / any MCP client
```

The pipeline has three separate processes. `convert` is a one-time GPU step that produces Markdown sidecars. `ingest` reads those sidecars (or parses documents directly when no sidecar exists) and upserts chunks into the vector store. The MCP server runs as a persistent subprocess managed by the MCP client.

---

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, HTML, Markdown, AsciiDoc, EPUB via Docling
- **GPU-accelerated pre-convert** — `convert` command processes binary documents to Markdown sidecars; subsequent `ingest` runs read the sidecar and skip Docling entirely, making ingest CPU-only and fast after the first pass
- **Crash-isolated batch conversion** — each file is converted in its own subprocess; a Docling PDF crash doesn't abort the whole batch
- **Code-aware chunking** — preserves code blocks, tables, and heading hierarchy
- **Local embeddings** — Ollama with snowflake-arctic-embed2 (1024 dimensions, 8K context)
- **Dual vector store** — Qdrant (primary) or ChromaDB (Docker-free fallback)
- **Incremental updates** — SHA-256 hashing skips unchanged files
- **20 curated collections** — covering .NET, Python, Rust, architecture, security, AI/ML, edge, robotics, and more
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

```bash
grounded-code-mcp search "async HTTP request"
grounded-code-mcp search "dependency injection" --collection patterns
grounded-code-mcp search "error handling" -n 10 --min-score 0.4
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

## Collections

20 curated collections covering the domains I work in. Each maps a `sources/` subdirectory to a collection name.

| Directory | Collection | What belongs here |
|-----------|-----------|-------------------|
| `sources/internal` | `internal` | Engineering standards — XP, TDD, CI/CD, DDD, OWASP, NIST AI |
| `sources/patterns` | `patterns` | Design patterns — GoF, CQRS, Clean Architecture, DI |
| `sources/architecture` | `architecture` | Software architecture — DDIA, SRE, 12-Factor, C4, arc42 |
| `sources/systems-thinking` | `systems_thinking` | Systems thinking — Meadows, feedback loops, chaos engineering |
| `sources/dotnet` | `dotnet` | .NET/C#, ASP.NET Core, Entity Framework, migration guides |
| `sources/python` | `python` | Python, FastAPI, Pydantic, FastMCP, pytest, cosmicpython |
| `sources/databases` | `databases` | SQL, PostgreSQL, relational theory |
| `sources/edge-ai` | `edge_ai` | AI engineering, RAG, embeddings, LLM application design |
| `sources/industrial-automation` | `automation` | PLC, OPC UA, MODBUS, ICS security, Raspberry Pi |
| `sources/4d-legacy` | `4d_legacy` | 4D platform — source reference for 4D → .NET migration |
| `sources/php` | `php` | PHP manual, Laravel (5.5 / 6.x / 12.x) |
| `sources/javascript` | `javascript` | JS/TS, Vue 2/3, jQuery, ECMAScript spec |
| `sources/ui-ux` | `ui_ux` | Laws of UX, Nielsen, WCAG 2.2, ARIA, USWDS, GOV.UK |
| `sources/gov` | `gov` | NIST 800-53/171/218, DOE, Zero Trust, AI RMF, CUI |
| `sources/robotics` | `robotics` | ROS 2, MuJoCo, Isaac Lab, LeRobot, VLA models |
| `sources/rust` | `rust` | Rust ownership, async/Tokio, Cargo, error handling, Axum |
| `sources/langsmith` | `langsmith` | LangSmith — tracing, evaluation, datasets, prompt engineering |
| `sources/langchain` | `langchain` | LangChain LCEL, chains, agents, retrievers, RAG patterns |
| `sources/langgraph` | `langgraph` | LangGraph — state machines, agent graphs, multi-agent orchestration |
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
| Embeddings | Ollama + snowflake-arctic-embed2 | 1024-dim, 8K context, fully local |
| Configuration | TOML + Pydantic | Deep-merged layered config |
| CLI | Click + Rich | |
| Testing | pytest | 276 tests |
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
