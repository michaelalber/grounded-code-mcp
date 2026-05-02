# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Global rules (TDD, security, quality gates, Python standards, AI behavior) are in
> `~/.claude/CLAUDE.md` and apply here automatically.
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
  - `src/grounded_code_mcp/__main__.py` — Click CLI (`ingest`, `convert`, `serve`, `status`, `search`)
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
  - `convert` runs Docling on binary sources and writes `foo.pdf.md` sidecars. Run before `ingest` on GPU machines; `ingest` then reads the sidecar and skips Docling entirely.
  - `flash-attn` is **not** in `pyproject.toml` — manual install only: `pip install flash-attn --no-build-isolation`. Requires CUDA toolkit + Ampere+ GPU. Enable via `cuda_use_flash_attention2 = true` in `[docling]`.

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
| 2026-04 | FastMCP upgraded to >=3.2.0 | CVE-2026-32871 (critical SSRF), CVE-2026-27124 (high OAuth), CVE-2025-64340 (medium command injection) — v3 API was compatible with existing usage; 276 tests pass |
| 2025-02 | Docling for document parsing | Handles PDF, EPUB, HTML, Markdown with layout awareness; preserves table structure |
| 2025-02 | Ollama + snowflake-arctic-embed2 | Local-only embeddings with no cloud dependency; 1024-dim balances quality and speed |
| 2025-02 | Collections prefixed `grounded_` | Namespace isolation in shared Qdrant instances |
| 2025-02 | HTTP transport binds `127.0.0.1` only | Security by default — never expose to network without explicit override |
| 2025-02 | Separate RED / GREEN / REFACTOR commits | Verifiable TDD evidence on feature branches; RED commits never pushed to `main` |
| 2026-04 | Removed Microsoft Learn PDF exports and Writing Style Guide from `dotnet` and `internal` collections | Live equivalents covered by the Microsoft Learn MCP; keeping static snapshots is redundant and creates drift |
| 2026-04 | Markdown sidecars (`foo.pdf.md`) + GPU-accelerated `convert` command | Decouples expensive Docling conversion (GPU) from ingest (CPU); sidecars let `ingest` use the fast plaintext path. `flash-attn` kept out of `pyproject.toml` — manual install only, Ampere+ GPU required. |
| 2026-05 | Removed `langchain` and `langgraph` collections | Well-covered by model training data; static snapshots create drift with fast-moving docs. `langsmith` retained (evaluation/tracing workflows are less common in training data). |

---

## Open Loops

- [ ] Untracked source directories in repo root (`async-book/`, `burn/`, `nomicon/`, `patterns/`, `rust-by-example/`) — pending decision on ingesting as Rust sub-collections
- [ ] `w3c-trace-context.html` untracked — pending ingestion target decision

---

## Team

| Name | Role | Notes |
|---|---|---|
| Michael Alber | Owner / sole maintainer | Reviews all changes |

---

## Available Tools

- `mcp__grounded-code-mcp__search_knowledge` — search vetted documentation by query and collection
- `mcp__grounded-code-mcp__search_code_examples` — find code examples by query and language
- `mcp__grounded-code-mcp__list_sources` — list ingested source documents
- `mcp__grounded-code-mcp__get_source_info` — get metadata for a specific source document

---

## Project Boot Ritual

At the start of every session:

1. Read this file (`CLAUDE.md`), `intent.md`, and `constraints.md`.
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

## Project-Specific Security

Beyond global rules:

- Validate file extensions against allowlist: `.pdf`, `.epub`, `.md`, `.txt`, `.rst`, `.html`
- Verify MIME type via magic bytes or UTF-8 validation
- Enforce file size limits (100 MB default, configurable) and sanitize filenames
- HTTP transport binds to `127.0.0.1` by default — never expose to an external network
