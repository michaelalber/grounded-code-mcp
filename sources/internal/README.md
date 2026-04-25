# `internal` — Engineering Standards & Practices

This collection is the **authoritative grounding source** for software engineering
methodology, process, and team practices.

## What belongs here

- XP, TDD, continuous delivery, and agile methodology references
- Testing strategy and test-driven design guides
- Code quality, readability, and team practices
- Domain-driven design (strategic patterns)
- Security-by-design and AI safety/risk frameworks
- Technical writing standards and documentation style guides

## `tech-writing/` sub-directory

Populated by `download_tech_writing_docs.py`. Run from the repo root:

```bash
python download_tech_writing_docs.py
grounded-code-mcp ingest sources/internal --collection internal
```

### Free resources (auto-downloaded)

| Directory | Source | Purpose |
|---|---|---|
| `diataxis/` | diataxis.fr | Diátaxis framework — tutorial / how-to / reference / explanation model |
| `write-the-docs/` | writethedocs.org/guide | Community documentation practices, docs-as-code, tooling |
| `plain-language-gov/` | plainlanguage.gov/guidelines | US federal plain language — sentences, word choice, organisation |
| `18f-content-guide/` | github.com/18F/content-guide | GSA / 18F content standards — plain language, accessibility, inclusive language |
| `google-style-guide/` | developers.google.com/style | API reference, code samples, prose style, word list for developer docs |
| `microsoft-style-guide/` | learn.microsoft.com/en-us/style-guide | Voice, tone, end-user and developer content, UI text, A–Z word list |
| `gitlab-style-guide/` | docs.gitlab.com | In-house developer doc structure, word list, content architecture |

### Additional sources

PDFs of technical writing books and other documentation standards references
can be placed directly in `sources/internal/tech-writing/` and will be ingested
automatically.
