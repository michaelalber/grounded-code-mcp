# grounded-code-mcp — Constraints

---

## Must Do

- Load and confirm context (`AGENTS.md`, `intent.md`, `constraints.md`) before every session.
- Confirm the task with the user before starting — no active Jira issue exists.
- Write three verifiable acceptance criteria before delegating any significant subtask.
- Confirm understanding before executing any irreversible action (delete, deploy, push).
- Run `.venv/bin/pytest` before every commit and verify all tests pass.
- Run `pipx install . --force` after any code change before testing CLI behaviour.
- Run ingest jobs **one collection at a time** — never in parallel.
- Add a `# VERIFY:` comment rather than guess when uncertain about a function signature, API, or behaviour.

---

## Must NOT Do

- Do not begin a task that has no verifiable acceptance criteria.
- Do not re-litigate decisions already logged in `AGENTS.md` or `intent.md` Persistent Decisions.
- Do not use `.venv/bin/grounded-code-mcp` — always use the pipx-installed binary.
- Do not run two ingest jobs simultaneously — parallel ingest causes OOM.
- Do not commit `config.toml` with machine-specific overrides (Ollama host, Qdrant URL, port) — those belong in `~/.config/grounded-code-mcp/config.toml`.
- Do not hardcode secrets, tokens, or credentials — use environment variables or a secrets manager.
- Do not commit generated files, build artifacts, or `.env` files.
- Do not change the embedding model or vector dimension without explicit human approval — it breaks compatibility with all existing vector data.
- Do not modify chunking parameters without a measurable before/after retrieval quality comparison.

---

## Preferences

- Prefer brevity over completeness unless depth is explicitly requested.
- Prefer asking one clarifying question over assuming and proceeding.
- Prefer flagging a problem before executing a workaround.
- Prefer editing an existing file over creating a new one.
- Prefer the grounded-code-mcp knowledge base over training data for language-specific idioms — use `search_knowledge` before writing non-trivial Python.
- Prefer smaller atomic chunks that preserve table and function boundaries over large context blobs.

---

## Escalate Rather Than Decide

- Any change to the chunking strategy — affects all existing embeddings and may require full re-ingest.
- Any change to the embedding model or dimension — breaks vector store compatibility.
- Adding a new collection to `config.toml` — collection ownership and source vetting is a human decision.
- Any action that modifies or deletes vector store data.
- Any request where acceptance criteria cannot be met within stated constraints.
- Any security-relevant decision not explicitly covered by existing constraints.

---

## Code Quality Gates

- **Test coverage (business logic):** ≥ 80% — run `.venv/bin/pytest --cov=grounded_code_mcp`
- **Test coverage (security-critical paths — file validation, MIME checking, size limits):** ≥ 95%
- **Cyclomatic complexity (per method):** < 10
- **Code duplication:** ≤ 3%
- **Commit format:** Conventional Commits — `feat:`, `fix:`, `refactor:`, `chore:`, `test:`, `docs:`
- **Commit scope:** Atomic — one logical change per commit; test and implementation in separate commits
- **Lint:** `ruff check src/ tests/` — zero errors
- **Type check:** `mypy src/` — zero errors (strict mode enabled)
- **Security scan:** `bandit -r src/ -c pyproject.toml` — zero high or critical issues
