# grounded-code-mcp — Evals

---

## Eval Philosophy

Evals are not a finishing step — they are safety infrastructure. Write them before the agent starts. Run them after every model update. A passing test suite ≠ done; tests verify code correctness, evals verify output is actually good relative to project intent.

A passing eval is measurable, repeatable, and would survive scrutiny from the project owner (Michael Alber).

Evals answer: *"Is the output actually good?"* — not *"does it look reasonable?"*

---

## Test Cases

### Test Case 1: New Collection Ingest

- **Input / Prompt:** Add a new collection (e.g., `sources/rust-by-example/`) to `config.toml` and ingest it with `grounded-code-mcp ingest --collection rust_extended sources/rust-by-example/`
- **Known-Good Output:** Command exits 0; `manifest.json` updated with new file hashes; at least one search query against the new collection returns a result with score ≥ 0.5
- **Pass Criteria:**
  - [ ] `grounded-code-mcp ingest` exits with code 0
  - [ ] `grounded-code-mcp search "ownership borrowing" --collection rust_extended` returns ≥ 1 result
  - [ ] Top result score ≥ 0.5
  - [ ] No previously-ingested collections affected (run a smoke search on `internal` and `python` collections)
  - [ ] All tests pass: `.venv/bin/pytest`
- **Last Run:** — | **Result:** —
- **Notes:** —

---

### Test Case 2: Bug Fix in Search / Chunking Pipeline

- **Input / Prompt:** A reported issue where a specific document type (e.g., HTML) produces zero results or low-score results despite relevant content
- **Known-Good Output:** Root cause identified via a failing unit test; fix applied with minimal scope; re-ingestion of the affected collection restores search quality; no regressions in other collection search results
- **Pass Criteria:**
  - [ ] A failing test exists before the fix (RED commit present on feature branch)
  - [ ] `.venv/bin/pytest` fully green after the fix (GREEN commit)
  - [ ] `ruff check src/ tests/` — clean
  - [ ] `mypy src/` — clean
  - [ ] `bandit -r src/ -c pyproject.toml` — zero high/critical
  - [ ] Smoke search on unrelated collection (`internal`) still returns valid results
- **Last Run:** — | **Result:** —
- **Notes:** —

---

### Test Case 3: New MCP Tool or Tool Enhancement

- **Input / Prompt:** Add a new parameter or tool to `server.py` (e.g., a `min_score` filter on `search_code_examples`)
- **Known-Good Output:** New parameter documented in the function signature and docstring; unit test covers the new behaviour; existing MCP tool integration is not broken; `pipx install ".[all]" --force` followed by a live call to the tool produces the expected filtered output
- **Pass Criteria:**
  - [ ] Failing test written before production code
  - [ ] All tests pass: `.venv/bin/pytest`
  - [ ] `mypy src/` — no new type errors
  - [ ] Tool visible and correctly typed when the MCP server is introspected
  - [ ] No breaking change to the existing `search_knowledge` or `search_code_examples` tool signatures
- **Last Run:** — | **Result:** —
- **Notes:** —

---

## Taste Rules (Encoded Rejections)

| # | Pattern to Reject | Why It Fails | Rule |
|---|---|---|---|
| 1 | Output that "looks right" but isn't grounded in project context | Generic output requires cleanup that defeats delegation | Always anchor recommendations to a specific fact from `AGENTS.md` or the active task description |
| 2 | Production code written before a failing test | Violates the TDD contract and removes verifiability | RED commit must precede GREEN commit on every feature |
| 3 | Chunking or embedding parameter changes without a before/after search quality comparison | Silent regressions in retrieval quality — the pipeline looks healthy but answers degrade | Any parameter change must include a `search` smoke test on ≥ 2 representative queries |

---

## CI Gate

- **Install:** `pip install -e ".[all,dev]"` — zero errors
- **Tests:** `.venv/bin/pytest --cov=src/grounded_code_mcp --cov-report=term-missing` — all 276 pass
- **Coverage:** ≥ 80% business logic, ≥ 95% security-critical (file validation paths)
- **Lint:** `.venv/bin/ruff check src/ tests/` — clean
- **Type check:** `.venv/bin/mypy src/` — clean
- **Security scan:** `.venv/bin/bandit -r src/ -c pyproject.toml` — zero high or critical issues
- **Dependency audit:** `pip-audit --skip-editable --desc on` — no known CVEs (see `ci.yml` for any active ignore list)

> Append CI gate results as a sub-item of each Test Case entry on every run.

---

## Rejection Log

<!-- Append rejected outputs here. Never delete entries. Review weekly to extract new Taste Rules. -->
