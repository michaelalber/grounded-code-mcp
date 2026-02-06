# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

grounded-code-mcp is a local MCP (Model Context Protocol) server that provides RAG capabilities over a persistent knowledge base. The server enables AI coding assistants (Claude Code, OpenCode, etc.) to ground their responses in vetted, authoritative documentation.

## Philosophy

This project addresses LLM limitations (hallucination, brittle compositionality, lack of grounding) by anchoring AI responses to curated technical documentation via RAG.

**Core principles:**
- Ground responses in curated knowledge base content
- Treat AI output as requiring human review
- Use structured outputs with validation
- Pair with TDD workflows

## Tech Stack

| Component | Choice |
|-----------|--------|
| MCP Framework | FastMCP (<3) |
| Document Parsing | Docling |
| Vector Store | Qdrant (primary), ChromaDB (fallback) |
| Embeddings | Ollama + mxbai-embed-large (1024 dim) |
| Configuration | TOML + Pydantic |
| CLI | Click + Rich |
| Testing | pytest |
| Linting | ruff |
| Type Checking | mypy |

## Build & Test Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

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

## CLI Commands

```bash
grounded-code-mcp ingest [--force] [--collection NAME] [PATH]
grounded-code-mcp status
grounded-code-mcp serve [--debug]
grounded-code-mcp search "query" [--collection NAME] [-n NUM] [--min-score FLOAT]
```

## Architecture

```
src/grounded_code_mcp/
├── __main__.py      # CLI entry point (Click commands)
├── config.py        # Settings loading (TOML + Pydantic)
├── manifest.py      # File tracking with SHA-256 hashing
├── parser.py        # Docling document parsing
├── chunking.py      # Code-aware semantic chunking
├── embeddings.py    # Ollama client wrapper
├── vectorstore.py   # Qdrant/ChromaDB abstraction
├── ingest.py        # Ingestion pipeline orchestrator
└── server.py        # FastMCP server with MCP tools
```

## Key Patterns

### MCP Tools (server.py)
- `search_knowledge` - Search documentation with optional collection filter
- `search_code_examples` - Find code examples by query and language
- `list_collections` - List available collections
- `list_sources` - List ingested source documents
- `get_source_info` - Get details about a specific source

### Chunking Strategy (chunking.py)
- Text chunks: 1000-1500 chars with 200 char overlap
- Code blocks: Atomic up to 3000 chars; split on function boundaries if larger
- Tables: Atomic, never split
- Heading hierarchy preserved as context metadata

### Change Detection (manifest.py)
- SHA-256 hashing for file change detection
- Incremental ingestion (only re-process changed files)
- Chunk IDs tracked for targeted deletion on re-ingestion

### Security-by-Design
- Validate all inputs at system boundaries
- Sanitize filenames (remove path traversal, special chars)
- Validate file extensions: `.pdf`, `.epub`, `.md`, `.txt`, `.rst`, `.html`
- Verify MIME type via magic bytes or UTF-8 validation
- Max size: 100MB (configurable)
- Store uploads outside web root
- Follow OWASP guidelines for file handling, auth, and data protection

### YAGNI Principle
- No abstract interfaces until needed (Rule of Three)
- No repository pattern - direct JSON read/write
- No dependency injection containers
- No plugin architecture - simple match/case on file extension
- Add abstractions only when necessary

## Configuration

Default config in `config.toml`. Settings can be customized:

```toml
[knowledge_base]
sources_dir = "sources"
data_dir = ".data"

[ollama]
base_url = "http://localhost:11434"
model = "mxbai-embed-large"
embedding_dim = 1024

[vectorstore]
provider = "qdrant"  # or "chroma"
```

## Prerequisites

Ollama must be running with the embedding model:

```bash
ollama pull mxbai-embed-large
ollama serve  # or: systemctl --user start ollama
```
