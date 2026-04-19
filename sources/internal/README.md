# `internal` — Engineering Standards & Practices

This collection is the **authoritative grounding source** for software engineering
methodology, process, and team practices. The canonical agent spec
(`xp-and-continuous-delivery-practices.md`) lives here.

## What belongs here

- XP, TDD, continuous delivery, and agile methodology books
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

### Books to purchase (place PDFs here manually)

| File to create | Book |
|---|---|
| `tech-writing/docs-for-developers.pdf` | *Docs for Developers* — Bhatti et al., Apress 2021 |
| `tech-writing/developing-quality-technical-information.pdf` | *Developing Quality Technical Information* — Haley et al., IBM Press 3rd ed. |
| `tech-writing/every-page-is-page-one.pdf` | *Every Page is Page One* — Mark Baker, XML Press 2013 |
| `tech-writing/the-sense-of-style.pdf` | *The Sense of Style* — Steven Pinker, Viking 2014 |
