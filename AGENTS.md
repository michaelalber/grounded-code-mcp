# AGENTS.md

This file contains guidelines for agentic coding agents operating in this repository.

## Project Overview

grounded-code-mcp is a local MCP (Model Context Protocol) server that provides RAG capabilities over a persistent knowledge base. The server enables AI coding assistants (Claude Code, OpenCode, etc.) to ground their responses in vetted, authoritative documentation.

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
| Type Checking | mypy (strict mode) |
| Security | bandit |
| Python | >=3.10 |

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

### Key Patterns

**MCP Tools (server.py):**
- `search_knowledge` - Search documentation with optional collection filter
- `search_code_examples` - Find code examples by query and language
- `list_collections` - List available collections
- `list_sources` - List ingested source documents
- `get_source_info` - Get details about a specific source

**Chunking Strategy (chunking.py):**
- Text chunks: 1000-1500 chars with 200 char overlap
- Code blocks: Atomic up to 3000 chars; split on function boundaries if larger
- Tables: Atomic, never split
- Heading hierarchy preserved as context metadata

**Change Detection (manifest.py):**
- SHA-256 hashing for file change detection
- Incremental ingestion (only re-process changed files)
- Chunk IDs tracked for targeted deletion on re-ingestion

## Build/Lint/Test Commands

### Setup
```bash
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_chunking.py

# Run a specific test class
pytest tests/test_chunking.py::TestDocumentChunker::test_empty_content

# Run a specific test method
pytest tests/test_chunking.py::TestDocumentChunker::test_simple_text

# Run with coverage
pytest --cov=grounded_code_mcp
```

### Linting and Type Checking
```bash
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

## Configuration

Default config in `config.toml`:

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

## Code Style Guidelines

### Imports
- Follow standard Python import ordering
- Separate standard library from third-party from local imports with blank lines
- Use `from __future__ import annotations` for forward references
- Prefer absolute imports over relative imports

### Formatting
- Follow PEP 8 style guidelines
- Use Ruff for linting, auto-formatting, and import sorting
- Line length is 100 characters max
- Ruff rules enabled: E, W, F, I (isort), B, C4, UP, S (bandit), T20, SIM, RUF

### Types
- Use type hints for all function parameters and return types
- Use Pydantic models for configuration
- Use TypedDict for structured data types
- Use dataclasses for simple data containers
- mypy runs in strict mode (`strict = true`)

### Naming Conventions
- Use snake_case for functions, methods, and variables
- Use PascalCase for classes and dataclasses
- Use UPPER_CASE for constants
- Use descriptive names; avoid abbreviations when not necessary

### Error Handling
- Use try/except blocks for operations that may fail
- Provide helpful error messages when catching exceptions
- Use specific exception types where possible
- Use `raise RuntimeError(...)` instead of bare `assert` for runtime checks (bandit S101)
- Log errors appropriately

### Documentation
- Use docstrings for all public functions and classes
- Follow Google-style docstrings
- Include parameter descriptions with types
- Include return value descriptions
- Include example usage when appropriate

### Testing
- All code should have corresponding tests
- Use fixtures for test data setup
- Test edge cases and error conditions
- Use pytest for testing framework
- Aim for high test coverage
- `asyncio_mode = "auto"` is configured — no `@pytest.mark.asyncio` decorator needed
- Default pytest options: `-v --tb=short`

### Asynchronous Code
- Use async/await syntax for async operations
- Follow the existing async patterns in the codebase

### Security
- Never include secrets in source code
- Use proper input validation
- Implement security checks with bandit
- Follow the project's security best practices
