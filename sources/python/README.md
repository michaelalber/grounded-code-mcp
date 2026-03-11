# `python` — Python Language & Ecosystem

Python language reference, modern idioms, type system, testing patterns, web frameworks,
data tooling, architecture patterns, and coding standards. Focus on Python 3.10+ and the
libraries most relevant to AI-assisted software development.

## What belongs here

- **Python language reference** — official Python docs, stdlib reference
- **Modern idiomatic Python** — type hints, async/await, dataclasses, pattern matching,
  Python 3.10+ features
- **Python PEPs and coding standards** — style guide (PEP 8), docstring conventions
  (PEP 257), type hint specifications (PEP 484, 526, and related)
- **Testing frameworks** — pytest patterns, fixtures, parametrize, coverage, mocking
- **Web frameworks** — FastAPI, Flask, or any Python web framework in use by your project
- **Data validation and settings** — Pydantic v2 models, validators, settings management
- **MCP development** — FastMCP server and client patterns
- **Architecture patterns** — DDD, ports & adapters, clean architecture, CQRS in Python
- **Data analysis and ML** — NumPy, pandas, deep learning frameworks, pipeline design

## Suggested sources

The following are high-quality, freely available resources suitable for ingestion:

| Source | Where to get it | Notes |
|---|---|---|
| Python docs | [docs.python.org](https://docs.python.org) or [github.com/python/cpython](https://github.com/python/cpython) `Doc/` | Branch per version (e.g. `3.13`) |
| Python PEPs | [peps.python.org](https://peps.python.org) or [github.com/python/peps](https://github.com/python/peps) `peps/` | RST format; sparse-clone individual PEPs |
| pytest docs | [github.com/pytest-dev/pytest](https://github.com/pytest-dev/pytest) `doc/` | RST format |
| FastAPI docs | [github.com/fastapi/fastapi](https://github.com/fastapi/fastapi) `docs/` | Markdown; branch `master` |
| Flask docs | [github.com/pallets/flask](https://github.com/pallets/flask) `docs/` | RST format |
| Pydantic docs | [github.com/pydantic/pydantic](https://github.com/pydantic/pydantic) `docs/` | Markdown; branch `main` |
| FastMCP docs | [github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp) `docs/` | Markdown |
| Architecture Patterns with Python | [github.com/cosmicpython/book](https://github.com/cosmicpython/book) | Free online book; Markdown |
| Fluent Python | O'Reilly (purchase or library) | PDF suitable for ingestion |
| Python Cookbook | O'Reilly (purchase or library) | PDF suitable for ingestion |

## Coverage areas

When populating this collection, aim to cover:

### Language & Standards
- Core syntax, built-in types, comprehensions, generators, decorators, context managers
- Type system: `typing` module, `Protocol`, `TypeVar`, `TypeAlias`, `ParamSpec`, `TypeGuard`
- Async/await, `asyncio`, `anyio`, structured concurrency patterns
- Dataclasses, `NamedTuple`, `TypedDict`, `Enum`
- PEP 8 style guide, PEP 257 docstring conventions, PEP 484/526 type annotation specs
- Standard library highlights: `pathlib`, `functools`, `itertools`, `collections`, `contextlib`

### Testing
- pytest fixtures, parametrize, markers, conftest patterns
- Mocking with `unittest.mock` and `pytest-mock`
- Integration and end-to-end test strategies
- Coverage configuration and enforcement
- Async test patterns with `pytest-asyncio`

### Web Frameworks & APIs
- Request/response lifecycle, routing, dependency injection
- Middleware, error handling, background tasks
- OpenAPI / schema generation
- Authentication and authorization patterns

### Data Validation & Configuration
- Pydantic v2: models, field validators, model validators, discriminated unions
- Settings management with `BaseSettings`
- JSON schema generation and serialization

### Architecture & Design
- Ports and adapters (hexagonal architecture)
- Domain-driven design in Python
- Repository pattern, service layer, unit of work
- Event-driven patterns, message buses

### Data & ML
- pandas, NumPy idioms and performance patterns
- Deep learning framework usage (training, inference, pipelines)
- RAG pipeline design: chunking, embedding, retrieval
- Model integration patterns with Ollama and cloud providers

## Format notes

- Markdown (`.md`) and reStructuredText (`.rst`) are both supported by the ingestion pipeline
- PDF files (books, whitepapers) are supported
- Prefer sparse clones of large repositories to avoid ingesting unrelated files
  (e.g. clone only the `docs/` or `doc/` subdirectory, or cherry-pick individual PEP `.rst` files)
- Remove `.git` directories before ingesting to keep the collection clean
