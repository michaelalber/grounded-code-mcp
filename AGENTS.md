# grounded-code-mcp — Project Context

> Global rules (TDD, security, quality gates, Python standards, AI behavior) are in
> `~/.config/opencode/AGENTS.md` and apply here automatically.
> This file contains only what is specific to this project.

---

## Project Overview

- **Name:** grounded-code-mcp
- **Purpose:** Local MCP server providing RAG over a persistent knowledge base of vetted technical documentation; eliminates hallucination by grounding AI coding assistant responses in authoritative sources.
- **Phase:** Maintain
- **Jira project key:** N/A — tracked via GitHub issues
- **Confluence space:** N/A
- **Definition of success:** Any AI coding session using this server produces grounded, citation-backed responses with zero reliance on training-data guesses for covered domains.

---

## Technology Stack

- **Language:** Python 3.10–3.12
- **Framework:** FastMCP (<3) for MCP server; Click + Rich for CLI
- **Vector store:** Qdrant (primary), ChromaDB (fallback)
- **Document parsing:** Docling ≥2.70.0
- **Embeddings:** Ollama — model: `snowflake-arctic-embed2` (1024-dim, 8192-token context)
- **Configuration:** TOML + Pydantic v2
- **Test framework:** pytest + pytest-asyncio + pytest-cov
- **CI/CD:** GitHub Actions — `.github/workflows/ci.yml` (lint, type-check, test matrix 3.10–3.12, dep-audit) + `security.yml` (Semgrep, Bandit, CodeQL, Trivy)
- **Package manager:** pip / hatchling build; runtime install via pipx

---

## Architecture

- **Pattern:** Ingest pipeline → vector search → MCP tool layer. Transport is stdio (default) or HTTP (local only, binds `127.0.0.1`).
- **Entry points:**
  - `src/grounded_code_mcp/__main__.py` — Click CLI (`ingest`, `serve`, `status`, `search`)
  - `src/grounded_code_mcp/server.py` — FastMCP server and all MCP tool handlers
- **Key directories:**
  - `src/grounded_code_mcp/` — production source (8 pipeline modules)
  - `tests/` — pytest unit tests; integration tests marked `@pytest.mark.integration`
  - `sources/` — knowledge base documents organised by collection subdirectory
  - `.grounded-code-mcp/` — runtime data: Qdrant storage, `manifest.json`
  - `scripts/` — utility scripts (doc downloaders)
  - `.github/workflows/` — CI definitions
- **Non-obvious constraints:**
  - CLI is installed via **pipx**, not `.venv`. After any code change run `pipx install . --force`.
  - Ollama must be running with `snowflake-arctic-embed2` pulled before ingest or search.
  - Qdrant must be running (Docker Compose or system service) for vector operations.
  - Ingest jobs must run **sequentially** — parallel ingest causes OOM. Never run two collections simultaneously.
  - Collection names in queries use the bare suffix (e.g., `"rust"`); the server prepends `grounded_` automatically.
  - Machine-specific config (Ollama host, Qdrant URL, port overrides) belongs in `~/.config/grounded-code-mcp/config.toml`, never in the committed `config.toml`.

---

## Key Files

| File | Why It Matters |
|---|---|
| `src/grounded_code_mcp/server.py` | All MCP tool handlers — the public API surface |
| `src/grounded_code_mcp/ingest.py` | Pipeline orchestrator: parse → chunk → embed → upsert |
| `src/grounded_code_mcp/chunking.py` | Code-aware chunking strategy; directly shapes retrieval quality |
| `src/grounded_code_mcp/config.py` | Settings loading with Pydantic; resolves committed + user override configs |
| `config.toml` | Collection map, chunking params, embedding model — shared across the team |
| `pyproject.toml` | Dependencies and ruff / mypy / bandit / pytest configuration |

---

## Persistent Decisions

| Date | Decision | Rationale |
|---|---|---|
| 2025-02 | Qdrant primary, ChromaDB fallback | Qdrant offers better performance and filtering; ChromaDB retained for Docker-free environments |
| 2025-02 | FastMCP (<3) pinned | Simplest path to MCP compliance; version pinned below 3 to avoid breaking API changes |
| 2025-02 | Docling for document parsing | Handles PDF, EPUB, HTML, Markdown with layout awareness; preserves table structure |
| 2025-02 | Ollama + snowflake-arctic-embed2 | Local-only embeddings with no cloud dependency; 1024-dim balances quality and speed |
| 2025-02 | Collections prefixed `grounded_` | Namespace isolation in shared Qdrant instances |
| 2025-02 | HTTP transport binds `127.0.0.1` only | Security by default — never expose to network without explicit override |
| 2025-02 | Separate RED / GREEN / REFACTOR commits | Verifiable TDD evidence on feature branches; RED commits never pushed to `main` |

---

## Open Loops

_None currently open._

---

## Team

| Name | Role | Notes |
|---|---|---|
| Michael Alber | Owner / sole maintainer | Reviews all changes |

---

## Available Tools

- **grounded-code-mcp MCP** — `search_knowledge`, `search_code_examples`, `list_sources`, `get_source_info` — use the running instance for knowledge lookups during development on this project
- **Bash** — run tests, linters, ingest commands, pipx reinstall
- **Read / Write / Edit / Grep / Glob** — file operations and code search

### Collections

Pass only the bare suffix — the server prepends `grounded_` automatically.

| `collection=` | What lives here |
|---|---|
| `"internal"` | Engineering standards: XP, TDD, CI/CD, DDD, Clean Architecture, OWASP, NIST AI |
| `"patterns"` | Design patterns: GoF, CQRS, DDD, Clean Architecture, DI, MADR; code smells + refactoring techniques |
| `"architecture"` | Software architecture: DDIA, SRE, 12-Factor, AOSA, C4, arc42, distributed systems |
| `"systems_thinking"` | Systems thinking: Meadows leverage points, feedback loops, chaos engineering |
| `"dotnet"` | .NET/C#, EF Core, ASP.NET Core, DI, migration guides |
| `"python"` | Python 3.13, FastAPI, FastMCP, Pydantic v2, pytest, Flask, cosmicpython |
| `"databases"` | SQL, PostgreSQL indexing, relational theory |
| `"edge_ai"` | AI/ML engineering, RAG, embeddings, NLP, AI agents |
| `"automation"` | Raspberry Pi, PLC, MODBUS, OPC UA, NIST 800-82, robotics |
| `"4d_legacy"` | 4D v18/v20 — source reference for 4D → .NET migration |
| `"php"` | PHP manual, Laravel 5.5 / 6.x / 12.x |
| `"javascript"` | JS/TS: Definitive Guide, TypeScript Handbook, Vue 2/3, ECMAScript 2024 |
| `"ui_ux"` | UI/UX: Laws of UX, Nielsen heuristics, WCAG 2.2, ARIA, GOV.UK, USWDS |
| `"gov"` | Federal/LANL: NIST 800-53/171/218, DOE, Zero Trust, AI RMF, CUI |
| `"robotics"` | ROS 2, MuJoCo, Isaac Lab, LeRobot, Spinning Up in Deep RL, VLA models |
| `"rust"` | Rust: ownership/borrowing/lifetimes, async/Tokio, Cargo, error handling, Axum |
| `"langsmith"` | LangSmith: tracing, evaluation, datasets, experiments, prompt engineering |
| `"langchain"` | LangChain: LCEL, chains, agents, retrievers, RAG patterns |
| `"langgraph"` | LangGraph: state machines, agent graphs, multi-agent orchestration |
| `"ssis"` | SSIS: packages, control flow, data flow, SSIS Catalog, expressions, deployment |
| `"api_design"` | REST API design: Zalando, Google AIP, Microsoft REST/Graph guidelines |

---

## Project Boot Ritual

At the start of every session:

1. Read this file (`AGENTS.md`), `intent.md`, and `constraints.md`.
2. No Jira — confirm the task with the user before starting.
3. State: current phase (Maintain), active task, top 3 constraints, open loops relevant to the task.
4. Do NOT begin work until context is confirmed.

---

## Runtime vs Development

The CLI is installed via **pipx** and available on PATH. Use it for all runtime commands:

```bash
grounded-code-mcp ingest ...
grounded-code-mcp serve --debug
```

After any code change, reinstall: `pipx install . --force`

Dev tools live in `.venv/` — **do not** use `.venv/bin/grounded-code-mcp`:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/pytest
.venv/bin/pytest --cov=grounded_code_mcp
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/
.venv/bin/bandit -r src/ -c pyproject.toml
```

---

## Git Workflow — Atomic TDD Commits

Stricter than the global standard to produce verifiable RED→GREEN evidence:

- **RED:** Write failing test → commit `test: add failing test for <behavior>`
- **GREEN:** Minimal code to pass → commit `feat|fix: <description>`
- **REFACTOR:** Improve structure, tests stay green → commit `refactor: <description>`
- Never combine new tests and new production code in a single commit
- Never push failing tests to `main` — RED commits are local / feature-branch only
- Run `.venv/bin/pytest` before every commit

---

## Project-Specific Code Conventions

Things easy to get wrong that tooling alone won't catch before you commit:

- `asyncio_mode = "auto"` is configured — do **not** add `@pytest.mark.asyncio` to test functions
- Use `raise RuntimeError(...)` instead of bare `assert` for runtime checks — bandit S101 will flag bare asserts
- Google-style docstrings for all public functions and classes

---

## Project-Specific Security

Beyond global rules:

- Validate file extensions against allowlist: `.pdf`, `.epub`, `.md`, `.txt`, `.rst`, `.html`
- Verify MIME type via magic bytes or UTF-8 validation
- Enforce file size limits (100 MB default, configurable) and sanitize filenames
- HTTP transport binds to `127.0.0.1` by default — never expose to an external network
