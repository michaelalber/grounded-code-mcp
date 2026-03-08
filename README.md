# grounded-code-mcp

A local MCP (Model Context Protocol) server providing RAG capabilities over a persistent knowledge base of technical documentation. Queryable by Claude Code, OpenCode, and other MCP-compatible clients.

## Philosophy

LLMs are sophisticated pattern matchers, not reasoning engines. This project mitigates known failure modes (hallucination, brittle compositionality, lack of grounding) by anchoring AI coding assistants to vetted, authoritative documentation via RAG.

**Design principles:**

- Ground responses in your curated knowledge base
- Treat AI output as "junior dev work" requiring review
- Structured outputs with validation
- Pair with TDD workflows — let tests catch AI mistakes

## Features

- **Multi-format document ingestion** — PDF, DOCX, PPTX, HTML, Markdown, AsciiDoc, EPUB via Docling
- **Code-aware semantic chunking** — Preserves code blocks, tables, and heading hierarchy
- **Local embeddings** — Ollama with snowflake-arctic-embed2 (1024 dimensions)
- **Dual vector store support** — Qdrant (primary) or ChromaDB (fallback)
- **Change detection** — SHA-256 hashing for incremental updates
- **MCP tools** — Search knowledge, find code examples, browse collections
- **Layered configuration** — Project config merged with per-user overrides

## Prerequisites

### Ollama

Install and start Ollama, then pull the embedding model:

```bash
# Install Ollama — see https://ollama.ai
ollama serve  # or: systemctl --user start ollama
ollama pull snowflake-arctic-embed2
```

### Qdrant

Install and start Qdrant (recommended):

```bash
# Docker
docker run -d -p 6333:6333 qdrant/qdrant

# or: https://qdrant.tech/documentation/quick-start/
```

ChromaDB is supported as a fallback (`provider = "chromadb"` in config).

## Installation

### Production (pipx — recommended)

```bash
pipx install git+https://github.com/michaelalber/grounded-code-mcp.git
```

After code changes: `pipx install . --force`

### Development

```bash
git clone https://github.com/michaelalber/grounded-code-mcp.git
cd grounded-code-mcp
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Dev tools (`pytest`, `ruff`, `mypy`) run from `.venv/bin/`. Use `grounded-code-mcp` from the pipx-installed binary for all runtime commands.

## Configuration

`config.toml` in the project root defines shared settings committed to the repo. Machine-specific settings (Ollama host, Qdrant URL, private collections) go in `~/.config/grounded-code-mcp/config.toml` — this file is deep-merged over the project config at startup.

Copy `config.toml.example` to get started:

```bash
cp config.toml.example ~/.config/grounded-code-mcp/config.toml
```

Typical user config:

```toml
[ollama]
host = "http://localhost:11434"

[vectorstore]
qdrant_url = "http://localhost:6333"
```

Private collections can be added in the user config using the same `[collections]` table — entries are merged, not replaced:

```toml
[collections]
"sources/private" = "private"
```

## Usage

### Ingest Documents

Populate a `sources/` subdirectory and ingest:

```bash
# Ingest a specific collection
grounded-code-mcp ingest sources/python

# Ingest all collections
grounded-code-mcp ingest

# Force full re-ingestion (ignores manifest)
grounded-code-mcp ingest --force

# Ingest a single collection by name
grounded-code-mcp ingest --collection python
```

> **Note:** Avoid running multiple ingests in parallel. Docling uses the GPU for PDF parsing — concurrent jobs cause CUDA out-of-memory errors.

### Check Status

```bash
grounded-code-mcp status
```

### Search

```bash
grounded-code-mcp search "async HTTP request"
grounded-code-mcp search "dependency injection" --collection patterns
grounded-code-mcp search "error handling" -n 10 --min-score 0.4
```

### Start MCP Server

```bash
# stdio (default — for Claude Code / OpenCode subprocess mode)
grounded-code-mcp serve

# HTTP (for remote clients or VM access)
grounded-code-mcp serve --transport streamable-http --host 127.0.0.1 --port 4242

grounded-code-mcp serve --debug
```

### Connect MCP Clients

**Claude Code** (user scope, available in all projects):

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

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_knowledge` | Search documentation with optional collection filter |
| `search_code_examples` | Find code examples by query and language |
| `list_collections` | List all available collections |
| `list_sources` | List ingested source documents |
| `get_source_info` | Get details about a specific source |

## Collections

Collections are mapped from source directories in `config.toml`. The server prepends `grounded_` to all collection names automatically.

| Directory | Collection (pass as `collection=`) | What belongs here |
|-----------|-------------------------------------|-------------------|
| `sources/internal` | `internal` | Engineering standards, XP/TDD/CD practices, agile, DDD, security frameworks |
| `sources/patterns` | `patterns` | Design patterns, clean code, dependency injection, CQRS |
| `sources/dotnet` | `dotnet_csharp` | .NET/C# APIs, ASP.NET Core, Entity Framework, migration guides |
| `sources/python` | `python` | Python language, testing, FastAPI, Pydantic, FastMCP, ML |
| `sources/databases` | `databases` | SQL, relational theory, PostgreSQL |
| `sources/edge-ai` | `edge_ai` | AI engineering, RAG pipelines, LLM application design, NLP |
| `sources/industrial-automation` | `industrial_automation` | PLC, OPC UA, MODBUS, ICS security, Raspberry Pi |
| `sources/4d-legacy` | `4d_legacy` | 4D platform docs for 4D → .NET migration (minimal LLM training coverage) |
| `sources/php` | `php_laravel` | PHP manual, PHP best practices, Laravel (multiple versions) |
| `sources/javascript` | `javascript_typescript` | JavaScript, TypeScript, Vue.js, jQuery, ECMAScript spec |

Add private collections in `~/.config/grounded-code-mcp/config.toml`.

Each collection directory contains a `README.md` describing what belongs there.

## Development

### Quality Checks

```bash
.venv/bin/pytest
.venv/bin/pytest --cov=grounded_code_mcp
.venv/bin/ruff format --check src/ tests/
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/
.venv/bin/bandit -r src/ -c pyproject.toml
```

All at once:

```bash
.venv/bin/pytest && .venv/bin/ruff format --check src/ tests/ && .venv/bin/ruff check src/ tests/ && .venv/bin/mypy src/ && .venv/bin/bandit -r src/ -c pyproject.toml
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| MCP Framework | FastMCP |
| Document Parsing | Docling |
| Vector Store | Qdrant (primary), ChromaDB (fallback) |
| Embeddings | Ollama + snowflake-arctic-embed2 |
| Configuration | TOML + Pydantic |
| CLI | Click + Rich |
| Testing | pytest |
| Linting | ruff |
| Type Checking | mypy |

## Security

- File type validation via allowlist (PDF, DOCX, PPTX, HTML, Markdown, AsciiDoc, EPUB)
- MIME type verification via magic bytes or UTF-8 validation
- File size limits (100 MB default, configurable)
- Filename sanitization (path traversal prevention)
- All inputs validated at system boundaries

## Troubleshooting

**Ollama connection error:**
```bash
ollama serve                          # start Ollama
ollama list                           # confirm model is pulled
curl http://localhost:11434/api/tags  # verify port
```

**Qdrant connection error:**
```bash
curl http://localhost:6333/healthz    # confirm Qdrant is running
```

**Ingestion failures:**
- Check file format is supported
- Run one ingest at a time — parallel ingests cause GPU OOM
- Use `--force` to re-ingest from scratch

**Search returns no results:**
- Verify ingestion: `grounded-code-mcp status`
- Lower the score threshold: `--min-score 0.3`
- Pass the bare collection suffix, not the full `grounded_*` name

## Contributing

Suggestions and feedback welcome via issues.

## Author

[Michael K Alber](https://github.com/michaelalber)

## License

MIT License — see [LICENSE](LICENSE) for details.
