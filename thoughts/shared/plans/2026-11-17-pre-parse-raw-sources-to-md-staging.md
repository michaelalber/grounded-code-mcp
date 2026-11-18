# Plan: Pre-parse Raw Sources → Stage Markdown → Ingest (Sidecar Removal)

**Created:** 2026-11-17
**Status:** draft — plan only, implementation not started (user asked for plan first)
**Ticket:** N/A
**Branch:** `main`
**Git base:** `254efcd`

---

## Overview

Remove the existing **sidecar feature** from the `grounded-code-mcp` ingestion pipeline and add a
**pre-parse → convert → stage → ingest** flow for **raw sources** (documents, whitepapers, ebooks)
that live in binary formats (`.pdf`, …).

Future end state: **all ingestion sources will be `.md` files**. The old
`book.pdf` + `book.pdf.md` sidecar pattern must not return.

Raw sources live at an **arbitrary path** (file or directory). The existing **`convert` CLI
command** is repurposed to:

1. Scan the given raw path (or default to sources root) for supported **binary** documents.
2. Parse each via **Docling** (GPU for PDFs) — single-file mode calls `DocumentParser` directly;
   directory mode spawns one subprocess per file (crash isolation).
3. **Write the parsed `.md`** into `sources/<collection>/`, **mirroring the raw directory
   structure** (dest = `sources_dir / file.relative_to(root).with_suffix(".md")`).
4. Honor `--force` (overwrite staged file), `--dry-run` (list only), `--no-ocr`.

After staging is in place, the human runs `grounded-code-mcp ingest` to consume the staged `.md`
files. Staging may take **multiple rounds** (different collections, or different source types —
ebooks, documentation, …) before the final ingest.

---

## Decisions (from scoping conversation)

1. **Raw location:** arbitrary path. Single file or directory.
2. **Staged destination:** `sources/<collection>/…`, mirroring the raw structure.
   - **Goal (confirmed):** raw source `raw/<collection>/<file>.pdf` → staged source
     `sources/<collection>/<file>.md`; the ingestion source is the staged `.md`.
   - Directory raw root `R` containing `R/<collection>/<file>.pdf` → staged to
     `sources/<collection>/<file>.md`.
   - Sub-folder mirroring: `R/<collection>/folder1/file.pdf` → staged to
     `sources/<collection>/folder1/file.md`. Sub-folder structure is not needed for
     ingestion, but mirroring keeps files organised for human review / addition / deletion.
   - Single file `R/<collection>/<file>.pdf` → staged to `sources/<collection>/<file>.md`.
3. **`convert` repurposed:** writes staged `.md` into `sources/` instead of sidecars next to
   binaries. `--collection` no longer fits the raw-path design and is **dropped**; collection
   now comes from the path structure (and `ingest` derives the collection from the staged path).
4. **Stage then ingest:** staging first, ingest after. Multiple staging rounds allowed.
5. **Sidecar removal:** `parser.sidecar_path()`, the sidecar-preference branch in `parse()`, and
   the sidecar-exclusion branch in `scan_directory()` are deleted.
6. **Dest-folder existence check:** before writing a staged file, ensure the destination
   directory exists (create with parents if missing).

---

## Current state (from code reading)

- `src/grounded_code_mcp/parser.py`:
  - `sidecar_path(source: Path) -> Path` (~line 110) — e.g. `book.pdf` → `book.pdf.md`.
  - Sidecar-preference branch in `parse()` (~lines 209–217) for non-plaintext formats.
  - Sidecar-skip logic in `scan_directory()` (~lines 483–485).
  - `DocumentParser(enable_ocr, enable_table_extraction, pdf_page_batch_size)`, helpers
    `_get_converter`, `_parse_with_docling`, `_get_pdf_page_count`,
    `_split_pdf_batch_to_temp`, `_parse_pdf_in_batches`.
  - `PLAINTEXT_EXTENSIONS = {".md", ".markdown", ".mdx", ".rst", ".txt", ".asciidoc", ".adoc"}`;
    `SUPPORTED_EXTENSIONS` includes binary formats.
- `src/grounded_code_mcp/__main__.py`: `convert` command (~lines 246–360) —
  `--collection`, `--force`, `--dry-run`, `--no-ocr`, `path` argument.
  Single-file mode: parse → write sidecar. Directory mode: subprocess per file, write sidecar.
- `src/grounded_code_mcp/ingest.py`: passes `pdf_page_batch_size` / `docling_settings` into
  `DocumentParser` when scanning `sources/`; plaintext path already exists for `.md`.
- `src/grounded_code_mcp/config.py`: `KnowledgeBaseSettings.pdf_page_batch_size`,
  `DoclingSettings`.
- `src/grounded_code_mcp/manifest.py`: `SourceEntry.page_count` (binary-parse-only meaning).
- Tests: `tests/test_cli.py::TestConvertCommand` (~lines 593–941), parser sidecar classes/tests
  (`TestSidecarPath` ~669, `TestSidecarAwareParse` ~682, scan sidecar tests ~614–648, `test_parse_md_*`
  sidecar-bypass tests, Docling/batching tests that stay).
- Docs/scripts referencing sidecars: README, ARCHITECTURE, CLAUDE, AGENTS, CONTRIBUTING,
  `.gitignore`, `scripts/_ingest_check.py` (`INGGESTABLE` set), `scripts/ingest-missing.sh`.
- `config.toml`: `[docling]` section, `pdf_page_batch_size = 50`, `max_file_size_mb = 500`.

---

## Desired end state

- `convert [PATH]` — `PATH` is a raw file or directory (arbitrary path) of binary documents.
  - Stages parsed `.md` into `sources/` mirroring raw structure.
  - Goal: raw `raw/<collection>/<file>.pdf` → staged `sources/<collection>/<file>.md`.
  - Sub-folder mirroring: `raw/<collection>/folder1/file.pdf` → staged
    `sources/<collection>/folder1/file.md` (mirrored, though not needed for ingestion).
  - Skips existing staged files unless `--force`; `--dry-run` lists; `--no-ocr` disables OCR.
  - Directory mode: subprocess per file with return-code-based failure counting (keep).
- Sidecar machinery deleted from parser (`sidecar_path`, parse preference, scan exclusion).
- `sources/` contains only parsed `.md` files; `ingest` consumes them via the plaintext path.
- Full quality gate green (pytest, ruff, ruff format, mypy, bandit).

---

## What we're NOT doing

- No new CLI command — repurpose `convert` in place.
- No ingest behavior change beyond what already exists (ingest already scans `sources/` and
  reads plaintext; binary wiring can stay until real binaries appear there).
- No real PDF conversion runs in this plan (code-only; staging runs are human-driven on GPU host).
- No changes to chunking / embedding / manifest semantics beyond what the sidecar removal forces.

---

## Implementation approach

Strict RED → GREEN per phase; tests define done.

---

## Phase 1 — RED: rewrite convert tests; drop sidecar tests

**Overview:** Tests first for the new staging semantics and the removed sidecar behavior.

### Step 1: Rewrite `tests/test_cli.py::TestConvertCommand`

- Keep: `test_convert_command_exists`.
- Rewrite `test_convert_dry_run_prints_files_without_writing` → assert staged `book.md` NOT written,
  `Would convert` in output.
- Rewrite `test_convert_skips_existing_sidecar` → `test_convert_skips_existing_staged_file` (staged
  `doc.md` exists → `1 skipped`, content untouched).
- Rewrite `test_convert_force_overwrites_sidecar` → `test_convert_force_overwrites_staged_file`
  (staged `doc.md` = `# Old`, `--force` → `# New content`).
- Keep `test_convert_skips_plaintext_files` (docstring: plaintext never passed to Docling).
- Keep OCR tests (`--no-ocr` flag, config-respected, override) — single-file based.
- Rewrite `test_convert_single_file_path_converts_that_file` → staged dest is `<stem>.md` inside
  `sources_dir`; assert `"1 converted"` and staged content.
- Keep `test_convert_single_plaintext_file_reports_nothing_to_convert`.
- Keep directory-mode tests: `test_convert_directory_spawns_subprocess_per_file`,
  `test_convert_directory_subprocess_crash_counted_as_failed`, `test_convert_directory_no_ocr_propagated_to_subprocess`.
- Add: `test_convert_stages_into_sources_mirroring_raw_structure` — raw dir containing
  `whitepapers/my-ebook.pdf` → staged at `sources/whitepapers/my-ebook.md`; plus a sub-folder
  case: `raw/<collection>/folder1/file.pdf` → staged at `sources/<collection>/folder1/file.md`.

### Step 2: Drop sidecar tests in `tests/test_parser.py`

- Remove `TestSidecarPath`, `TestSidecarAwareParse`, scan sidecar tests, `test_parse_md_does_not_check_sidecar`,
  `test_parse_rst_does_not_check_sidecar`, and the `sidecar_path` import.
- Keep: plaintext parsing, Docling parsing, PDF batching, docling settings, scan tests minus
  sidecar exclusions.

### Step 3: Verification (RED)

```
.venv/bin/pytest tests/test_cli.py::TestConvertCommand tests/test_parser.py -v
```

Expected: new/rewritten tests FAIL (old behavior still in place).

---

## Phase 2 — GREEN: remove sidecar machinery; repurpose `convert`

**Overview:** Minimal implementation to make Phase 1 tests pass.

### Step 1: `src/grounded_code_mcp/parser.py`

- Delete `sidecar_path()`.
- Delete sidecar-preference branch in `parse()`.
- Delete sidecar-exclusion branch in `scan_directory()`.

### Step 2: `src/grounded_code_mcp/__main__.py` — `convert` command

- Replace with staging implementation:
  - Options: `--force`, `--dry-run`, `--no-ocr` (drop `--collection`).
  - Argument: `path` (file or dir, `exists=True`).
  - Single file: root = file.parent; files = [file] if binary else [].
  - Dir: root = dir; files = scanned binaries (not plaintext).
  - No path: root = `settings.knowledge_base.sources_dir`; files = scanned binaries there
    (will be empty once only `.md` exists — consistent with new policy).
  - Staged dest: `settings.knowledge_base.sources_dir / file.relative_to(root).with_suffix(".md")`.
  - Before writing the staged file: ensure destination directory exists
    (`staged.parent.mkdir(parents=True, exist_ok=True)`) — dest-folder existence check.
  - Skip if staged exists and not `--force`; overwrite with `--force`.
  - `--dry-run`: print `Would convert: <file>`; write nothing.
  - Single-file: parse via `DocumentParser(enable_ocr=..., docling_settings=...)`; write content
    to staged.
  - Dir: subprocess per file (`python -m grounded_code_mcp convert …` with `--no-ocr`/`--force`
    forwarded); non-zero rc → failed.
  - Summary: converted / skipped / failed counts.

### Step 2 verification (GREEN)

```
.venv/bin/pytest tests/test_cli.py::TestConvertCommand tests/test_parser.py -v
```

Expected: all pass.

---

## Phase 3 — GREEN: full quality gate

```
.venv/bin/pytest
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/mypy src/
.venv/bin/bandit -r src/ -c pyproject.toml
```

Fix any fallout (ingest wiring, config, etc.).

---

## Phase 4 — Docs / config / scripts cleanup

- `README.md`, `ARCHITECTURE.md`, `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md`: update convert
  behavior, sidecar references, staging flow.
- `.gitignore`: drop sidecar patterns if present.
- `scripts/_ingest_check.py`: `INGGESTABLE` set → plaintext-only if appropriate.
- `config.toml`: no change required (`[docling]` still used by convert).

---

## Rollback

Per-phase rollback = `git checkout <file>` for the touched source/test files; no data migration.

---

## Notes

- `.venv` in this workspace currently lacks dev tooling (no pytest/click/rich installed) — install
  before running any verification: `.venv/bin/pip install -e ".[all,dev]"`.
- Full end-to-end staging (real GPU Docling run) is out of scope for code phases; those are human
  operator steps after Phase 3.
- pytest collection of `tests/test_cli.py::TestConvertCommand` must be runnable before Phase 1 RED.