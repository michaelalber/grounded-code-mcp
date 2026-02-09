# grounded-code-mcp

A local MCP (Model Context Protocol) server providing RAG capabilities over a persistent knowledge base of technical documentation. Queryable by Claude Code, OpenCode, and other MCP-compatible clients.

## Philosophy

LLMs are sophisticated pattern matchers, not reasoning engines. This project mitigates known failure modes (hallucination, brittle compositionality, lack of grounding) by anchoring AI coding assistants to vetted, authoritative documentation via RAG.

**Design principles:**

- Ground responses in your curated knowledge base
- Treat AI output as "junior dev work" requiring review
- Structured outputs with validation
- Pair with TDD workflows—let tests catch AI mistakes

## Features

- **Multi-format document ingestion** - PDF, DOCX, PPTX, HTML, Markdown, AsciiDoc, EPUB via Docling
- **Code-aware semantic chunking** - Preserves code blocks, tables, and heading hierarchy
- **Local embeddings** - Ollama with mxbai-embed-large (1024 dimensions)
- **Dual vector store support** - Qdrant (primary) or ChromaDB (fallback)
- **Change detection** - SHA-256 hashing for incremental updates
- **MCP tools** - Search knowledge, find code examples, browse collections

## Installation

```bash
# Clone the repository
git clone https://github.com/michaelalber/grounded-code-mcp.git
cd grounded-code-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install with dev dependencies
pip install -e ".[dev]"
```

## Prerequisites

### Ollama

Install and start Ollama, then pull the embedding model:

```bash
# Install Ollama (see https://ollama.ai)
# Start the service
systemctl --user start ollama  # or: ollama serve

# Pull the embedding model
ollama pull mxbai-embed-large
```

## Usage

### Ingest Documents

Create a `sources/` directory and add your documentation:

```bash
mkdir -p sources/python sources/patterns
# Add markdown, PDF, DOCX files...

# Ingest all documents
grounded-code-mcp ingest

# Ingest specific path
grounded-code-mcp ingest sources/python/

# Force re-ingestion
grounded-code-mcp ingest --force
```

### Check Status

```bash
grounded-code-mcp status
```

### Search

```bash
grounded-code-mcp search "async HTTP request"
grounded-code-mcp search "dependency injection" --collection patterns
grounded-code-mcp search "error handling" -n 10 --min-score 0.8
```

### Start MCP Server

```bash
grounded-code-mcp serve
grounded-code-mcp serve --debug
```

## Configuration

Create a `config.toml` file to customize settings:

```toml
[knowledge_base]
sources_dir = "sources"
data_dir = ".data"

[ollama]
base_url = "http://localhost:11434"
model = "mxbai-embed-large"
embedding_dim = 1024

[chunking]
text_chunk_size = 1000
text_chunk_max_size = 1500
text_chunk_overlap = 200
max_code_chunk_size = 3000

[vectorstore]
provider = "qdrant"  # or "chroma"
collection_prefix = "grounded_"
```

## MCP Tools

When running as an MCP server, the following tools are available:

| Tool | Description |
|------|-------------|
| `search_knowledge` | Search documentation with optional collection filter |
| `search_code_examples` | Find code examples by query and language |
| `list_collections` | List all available collections |
| `list_sources` | List ingested source documents |
| `get_source_info` | Get details about a specific source |

## Collection Mapping

Documents are organized into collections based on directory structure:

| Directory | Collection |
|-----------|------------|
| `sources/python/` | `grounded_python` |
| `sources/dotnet/` | `grounded_dotnet` |
| `sources/patterns/` | `grounded_patterns` |
| `sources/databases/` | `grounded_databases` |

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=grounded_code_mcp

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Security scan
bandit -r src/ -c pyproject.toml
```

## Tech Stack

- **MCP Framework:** FastMCP
- **Document Parsing:** Docling
- **Vector Store:** Qdrant / ChromaDB
- **Embeddings:** Ollama + mxbai-embed-large
- **Configuration:** TOML + Pydantic
- **CLI:** Click + Rich
- **Testing:** pytest

## Security

- File type validation (PDF, DOCX, PPTX, HTML, Markdown, AsciiDoc, EPUB)
- MIME type verification via magic bytes or UTF-8 validation
- File size limits (100MB default, configurable)
- Filename sanitization (path traversal prevention, special chars removed)
- All inputs validated at system boundaries

## Troubleshooting

**Ollama Connection Error:**
- Ensure Ollama is running: `ollama serve` or `systemctl --user start ollama`
- Check the embedding model is pulled: `ollama list`
- Verify Ollama is on port 11434: `curl http://localhost:11434/api/tags`

**Ingestion Failures:**
- Check file format is supported
- Ensure sufficient disk space for vector store
- Use `--force` flag to re-ingest: `grounded-code-mcp ingest --force`

**Search Returns No Results:**
- Verify documents were ingested: `grounded-code-mcp status`
- Try lowering `--min-score` threshold
- Check collection name matches: `grounded-code-mcp search "query" --collection NAME`

**MCP Server Connection Issues:**
- Verify the server starts: `grounded-code-mcp serve --debug`
- Check client MCP configuration points to correct transport

## Contributing

This is a personal/educational project, but suggestions and feedback are welcome via issues.

## Author

[Michael K Alber](https://github.com/michaelalber)

## License

MIT License - see [LICENSE](LICENSE) for details.
