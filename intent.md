# grounded-code-mcp — Intent

---

## Agent Architecture

**This project uses:** Coding harness

**Reason:** Task-level maintenance work — feature additions, bug fixes, new collection onboarding — each reviewed by the owner before merge.

---

## Primary Goal

Enable AI coding sessions to produce grounded, citation-backed responses in covered engineering domains — eliminating hallucination by anchoring every answer in vetted, authoritative documentation served through a local RAG pipeline.

---

## Values (What We Optimize For)

1. **Correctness** — wrong embeddings or broken chunking silently poison every downstream search result
2. **Security** — local-only by default; validated inputs; no secrets in code
3. **Maintainability** — clean pipeline, TDD discipline, incremental ingestion
4. **Performance** — fast search, bounded memory during ingest
5. **Speed of delivery** — least important; adding a collection requires no code changes

---

## Tradeoff Rules

| Conflict | Resolution |
|---|---|
| Speed vs. correctness | Correctness. A fast wrong answer is worse than a slow right one. |
| Completeness vs. brevity | Brevity unless depth is explicitly requested. |
| Embedding model accuracy vs. ingest speed | Accuracy — wrong embeddings corrupt the entire collection. |
| New collection vs. pipeline stability | Add incrementally; test search quality before marking a collection ready. |
| Chunking granularity vs. retrieval recall | Prefer smaller atomic chunks that preserve table and function boundaries over large blobs. |

---

## Decision Boundaries

### Decide Autonomously

- Formatting, structure, naming within established project conventions
- Tool selection for read-only exploration
- Refactoring within the approved, scoped task
- Running linters and tests at any point
- Choosing which existing test fixtures to reuse

### Escalate to Human

- Any change to the chunking strategy (affects all existing embeddings — may require re-ingest)
- Any change to the embedding model or dimension (breaks compatibility with existing vector data)
- Adding a new collection to `config.toml` (collection ownership and source vetting is a human decision)
- Any action that modifies or deletes vector store data
- Scope changes beyond the stated task
- When acceptance criteria cannot be met within stated constraints

---

## What "Good" Looks Like

A good output for this project:

- Tests are written before production code; RED commit precedes GREEN commit
- Code changes are minimal and do not alter behaviour outside the stated task scope
- New collections are accompanied by at least one search smoke-test asserting score ≥ 0.5
- Security-relevant code (file validation, MIME checking, size limits) has ≥ 95% test coverage
- Commit messages follow Conventional Commits with enough specificity to understand the change without reading the diff

---

## Anti-Patterns (What Bad Looks Like)

- Writing production code before a failing test exists
- "Improving" chunking or embedding parameters without a measurable before/after comparison
- Ingesting a new collection without verifying search quality on at least one representative query
- Adding `# type: ignore` or `noqa` suppressions without a comment explaining why
- Combining unrelated changes in a single commit

---

## Persistent Decisions

| Date | Decision | Rationale |
|---|---|---|
| 2025-02 | Coding harness architecture | Single-owner project; every change reviewed before merge; no need for autonomous multi-session loops |
| 2025-02 | No dark factory / domain-memory.md | All work is human-initiated and human-reviewed; state lives in git, not an agent backlog |

---

## Open Loops

- [ ] Untracked Rust source directories (`async-book/`, `burn/`, `nomicon/`, `patterns/`, `rust-by-example/`) — decide whether to ingest as additional Rust sub-collections or a separate `grounded_rust_extended` collection
- [ ] No collection yet for observability / distributed tracing standards
