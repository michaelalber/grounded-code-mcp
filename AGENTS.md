# AGENTS.md

This file contains guidelines for agentic coding agents operating in this repository.

## Runtime vs Development

The CLI is installed via **pipx** and available on PATH. Use it directly for all runtime commands:
```bash
grounded-code-mcp ingest ...
grounded-code-mcp serve --debug
```

After code changes, reinstall: `pipx install . --force`

Dev tools (pytest, ruff, mypy) live in `.venv/`. **Do not** use `.venv/bin/grounded-code-mcp` — always use the pipx-installed binary.

## Build/Lint/Test Commands

### Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
.venv/bin/pytest

# Run a single test file
.venv/bin/pytest tests/test_chunking.py

# Run a specific test class
.venv/bin/pytest tests/test_chunking.py::TestDocumentChunker::test_empty_content

# Run a specific test method
.venv/bin/pytest tests/test_chunking.py::TestDocumentChunker::test_simple_text

# Run with coverage
.venv/bin/pytest --cov=grounded_code_mcp
```

### Linting and Type Checking
```bash
# Lint
.venv/bin/ruff check src/ tests/

# Type check
.venv/bin/mypy src/

# Security scan
.venv/bin/bandit -r src/ -c pyproject.toml
```

## Code Style Guidelines

### Imports
- Follow standard Python import ordering (stdlib, third-party, local)
- Separate groups with blank lines
- Use `from __future__ import annotations` for forward references
- Prefer absolute imports over relative imports

### Formatting
- Follow PEP 8 style guidelines
- Use Ruff for linting, auto-formatting, and import sorting
- Line length: 100 characters max
- Ruff rules enabled: E, W, F, I (isort), B, C4, UP, S (bandit), T20, SIM, RUF

### Types
- Type hints on all function parameters and return types
- Use Pydantic models for configuration
- Use TypedDict for structured data types
- Use dataclasses for simple data containers
- mypy runs in strict mode (`strict = true`)

### Naming
- `snake_case` for functions, methods, and variables
- `PascalCase` for classes and dataclasses
- `UPPER_CASE` for constants
- Use descriptive names; avoid unnecessary abbreviations

### Error Handling
- Use specific exception types, never bare `except:`
- Use `raise RuntimeError(...)` instead of bare `assert` for runtime checks (bandit S101)
- Provide helpful error messages
- Log errors appropriately

### Documentation
- Google-style docstrings for all public functions and classes
- Include parameter descriptions with types
- Include return value descriptions
- Include example usage when appropriate

### Testing
- Arrange-Act-Assert pattern for all tests
- Use fixtures for test data setup
- Test edge cases and error conditions
- `asyncio_mode = "auto"` is configured — no `@pytest.mark.asyncio` decorator needed
- Default pytest options: `-v --tb=short`

### Async
- Use async/await syntax for async operations
- Follow existing async patterns in the codebase

## Development Principles

### TDD
1. **Never write production code without a failing test first**
2. Cycle: RED (write failing test) → GREEN (minimal code to pass) → REFACTOR
3. Run tests before committing: `pytest`
4. Coverage target: 80% minimum for business logic, 95% for security-critical code

### Security-By-Design
- Validate all inputs at system boundaries
- Validate file extensions against allowlist (`.pdf`, `.epub`, `.md`, `.txt`, `.rst`, `.html`)
- Verify MIME type via magic bytes or UTF-8 validation
- Enforce file size limits (100MB configurable) and sanitize filenames
- Bind HTTP transport to `127.0.0.1` by default
- Never include secrets in source code — use environment variables
- All rules align with [OWASP Top 10 (2025)](https://owasp.org/Top10/2025/) guidance

### YAGNI (You Aren't Gonna Need It)
- Start with direct implementations
- Add abstractions only when complexity demands it
- Create interfaces only when multiple implementations exist
- No dependency injection containers
- No repository pattern — direct JSON read/write
- No plugin architecture — simple match/case on file extension

### Quality Gates
- **Cyclomatic Complexity**: Methods <10, classes <20
- **Code Coverage**: 80% minimum for business logic, 95% for security-critical code
- **Maintainability Index**: Target 70+
- **Code Duplication**: Maximum 3%

## Git Workflow — Atomic TDD Commits

- **Separate test and implementation commits** to create verifiable RED→GREEN evidence
- RED phase: Write failing test, commit with `test: add failing test for <behavior>`
- GREEN phase: Write minimal code to pass, commit with `feat|fix: <description>`
- REFACTOR phase: Improve structure (tests stay green), commit with `refactor: <description>`
- Never combine new tests and new production code in a single commit
- Don't commit failing tests to shared branches (RED commits are for local/feature branches)
- Commit message format: `test|feat|fix|refactor|style|ci|docs: brief description`
- Run `pytest` before every commit

## Tools

- **Bash**: Use for running tests, linters, and formatters
- **Read/Write/Edit**: For file operations
- **Grep/Glob**: For code search
- **Task**: For complex, multi-step operations
- **Skill**: For TDD cycles and architecture reviews

## Example Workflow

1. Write a failing test for the new feature
2. Run `pytest -k <test_name>` to confirm it fails (RED)
3. Commit: `git commit -m "test: add failing test for <behavior>"`
4. Write minimal code to make the test pass (GREEN)
5. Run full test suite: `pytest`
6. Run linters: `ruff check src/ tests/ && mypy src/`
7. Commit: `git commit -m "feat: <description>"`
8. Refactor if needed while keeping tests green (REFACTOR)
9. Commit: `git commit -m "refactor: <description>"`
