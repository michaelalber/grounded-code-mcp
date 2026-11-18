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
   - **Resolved (2026-11-17):** single-file staging root = raw root (for
     `R/<collection>/<file>.pdf` the raw root is `R` = `file.parent.parent`), not the
     file's immediate parent. Phase 2 Step 2 draft line (`root = file.parent`) was
     corrected to `root = file.parent.parent`; the draft value would have staged to
     `sources/<file>.md`, dropping the collection and contradicting the confirmed goal.
   - **Depth note:** the `parent.parent` default assumes raw files live exactly
     `R/<collection>/<file>` deep. A shallower single file (`raw/foo.pdf`) stages to
     `sources/raw/foo.md` (collection `raw`); the operator should pass the parent dir
     instead. Documented in the `convert --help` text.
3. **`convert` repurposed:** writes staged `.md` into `sources/` instead of sidecars next to
   binaries. `--collection` no longer fits the raw-path design and is **dropped**; collection
   now comes from the path structure (and `ingest` derives the collection from the staged path).
4. **Stage then ingest:** staging first, ingest after. Multiple staging rounds allowed.
5. **Sidecar removal:** `parser.sidecar_path()`, the sidecar-preference branch in `parse()`, and
   the sidecar-exclusion branch in `scan_directory()` are deleted.
6. **Dest-folder existence check:** before writing a staged file, ensure the destination
   directory exists (create with parents if missing).
7. **Staging root is shared, never recomputed (fix: parent vs. subprocess inconsistency):**
   directory mode forwards its scan root to each subprocess via the environment variable
   `GROUNDED_CONVERT_ROOT` (keeps the public CLI surface at `--force`/`--dry-run`/`--no-ocr`
   only — no hidden flag); single-file mode uses `GROUNDED_CONVERT_ROOT` when set, else
   `file.parent.parent`. The staged destination is computed in exactly **one helper**
   `_staged_dest(file, root) = sources_dir / file.relative_to(root).with_suffix(".md")`,
   so parent and child can never disagree on the staged path for nested raw roots.
   - Both paths are `.resolve()`d first; `relative_to` failure (file outside root) raises
     `ValueError` → that file is counted as failed, the batch continues. This also blocks
     symlink/`..` paths escaping `sources_dir`.
   - **Atomic staged writes:** write to `staged.with_suffix(".md.tmp")` then `os.replace()`
     onto the final name, so a crash mid-write can never leave a truncated `.md` that a
     later ingest would chunk and embed (same crash-safety pattern as `ingest._save_manifest`).
8. **Manifest migration + orphan-staged-file policy (fix):** global `ingest` (no `path`
   argument) prunes manifest entries whose key is relative to `sources_dir` but whose file
   no longer exists, deleting their vector chunks first. One mechanism covers both old
   sidecar-keyed entries (`c/book.pdf`, `c/book.pdf.md`, once the binaries/sidecars leave
   `sources/`) and staged `.md` files the human deletes. Staged `.md` are the source of
   truth: delete the staged file + re-ingest → gone from the KB. **Guard:** pruning is
   skipped entirely when zero files are scanned, so a fresh clone with an empty `sources/`
   can never wipe the manifest.
9. **`.gitignore` policy (fix):** staged `.md` stay ignored — they are derived from
   copyrighted binaries, same policy as the PDFs they came from. Only the stale sidecar
   comment in `.gitignore` is removed; `scripts/sync-sources.sh` is updated to transfer
   staged `.md` instead of raw binaries.

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
- No ingest behavior change beyond the guarded stale-entry prune (Decision 8); ingest
  already scans `sources/` and reads plaintext; binary wiring can stay until real binaries
  appear there.
- No real PDF conversion runs in this plan (code-only; staging runs are human-driven on GPU host).
- No changes to chunking / embedding / manifest semantics beyond what the sidecar removal forces.

---

## Implementation approach

Strict RED → GREEN per phase; tests define done. Every step lists exact file:line anchors
(current `main`, git base `254efcd`) and fixture sketches so a small model can execute
mechanically. Run `.venv/bin/pytest` before every commit (RED commits stay local).

---

## Phase 1 — RED: rewrite convert tests; drop sidecar tests

### 1.1 Rewrite `tests/test_cli.py::TestConvertCommand` (class starts line 593)

Existing fixtures/mocks to copy from: `temp_dir` (conftest.py:25), pattern at
test_cli.py:603–625 (`MagicMock()` settings with `knowledge_base.sources_dir = temp_dir`,
`collections = {}`, `docling = DoclingSettings()`; `patch("grounded_code_mcp.__main__.Settings")`
and `patch("grounded_code_mcp.parser.DocumentParser")`; `CliRunner().invoke(cli, [...])`).

1. Keep `test_convert_command_exists` (line 596) unchanged.
2. Rewrite `test_convert_dry_run_prints_files_without_writing` (603): fixture =
   `raw = temp_dir/"raw"; (raw/"books").mkdir(parents=True); pdf = raw/"books"/"book.pdf";
   pdf.write_bytes(b"%PDF-1.4")`; settings `sources_dir = temp_dir/"sources"`
   (mkdir it). Invoke `["convert", str(raw), "--dry-run"]`. Assert exit 0,
   `"Would convert" in output`, and `not (sources/"books"/"book.md").exists()`.
3. Rewrite `test_convert_skips_existing_sidecar` → `test_convert_skips_existing_staged_file`
   (627): fixture adds staged `sources/"books"/"doc.md"` with `"# Existing"`. Invoke
   `["convert", str(raw)]`. Assert `"1 skipped"`, staged content still `"# Existing"`.
4. Rewrite `test_convert_force_overwrites_sidecar` → `test_convert_force_overwrites_staged_file`
   (653): staged `sources/"books"/"doc.md"` = `"# Old"`; mocked `DocumentParser().parse`
   returns `ParsedDocument(path=pdf, content="# New content", file_type="pdf")` (pattern at
   665–668). Invoke `["convert", str(pdf), "--force"]` where
   `pdf = raw/"books"/"doc.pdf"`. Assert staged file == `"# New content"`.
5. Keep `test_convert_skips_plaintext_files` (684), OCR tests (708, 739, 768) — single-file,
   unchanged except they must still pass with sidecar refs gone from `__main__.py`.
6. Rewrite `test_convert_single_file_path_converts_that_file` (797): staged dest is
   `sources/"books"/"book.md"` for `raw/"books"/"book.pdf"` (root = `file.parent.parent`).
   Assert `"1 converted"` and staged content from mocked parser.
7. Keep `test_convert_single_plaintext_file_reports_nothing_to_convert` (828).
8. Keep directory-mode tests (851, 879, 910) but update assertions: subprocess cmdline is
   now `[sys.executable, "-m", "grounded_code_mcp", "convert", ...]` **plus env var**
   `GROUNDED_CONVERT_ROOT` (see 1.10).
9. Add `test_convert_stages_into_sources_mirroring_raw_structure`: fixture
   `raw/"whitepapers"/"my-ebook.pdf"` and `raw/"whitepapers"/"folder1"/"file.pdf"`;
   invoke `["convert", str(raw), "--dry-run"]`; assert output lists both; with mocked
   subprocess/DocumentParser assert dests `sources/"whitepapers"/"my-ebook.md"` and
   `sources/"whitepapers"/"folder1"/"file.md"`.
10. Add `test_convert_directory_forwards_root_to_subprocess`: patch
    `grounded_code_mcp.__main__.subprocess.run`; invoke `["convert", str(raw)]` with 1 pdf;
    assert `call.kwargs["env"]["GROUNDED_CONVERT_ROOT"] == str(raw.resolve())` and
    `str(pdf)` is the last cmd arg.
11. Add `test_staged_dest_rejects_file_outside_root`: import `_staged_dest` from
    `grounded_code_mcp.__main__`; `pytest.raises(ValueError)` for
    `_staged_dest(Path("/elsewhere/x.pdf"), raw, sources)`; and a symlink-escape case
    (symlink inside raw pointing outside → still raises or dest stays under sources).
12. Add `test_convert_writes_staged_file_atomically`: single-file convert with mocked
    parser; assert staged `.md` exists and `list(sources.rglob("*.md.tmp")) == []`.

### 1.2 Drop sidecar tests in `tests/test_parser.py`

1. Delete `TestSidecarPath` (~669–680), `TestSidecarAwareParse` (~682+),
   scan sidecar tests (614–648: `test_scan_excludes_sidecar_when_parent_exists`,
   `test_scan_includes_orphaned_sidecar`, `test_scan_plain_md_not_filtered_as_sidecar`),
   `test_parse_md_does_not_check_sidecar`, `test_parse_rst_does_not_check_sidecar`,
   and the `sidecar_path` import (line 17).
2. Keep everything else (plaintext parse, Docling, batching, scan minus sidecar).

### 1.3 Add prune tests in `tests/test_ingest.py` (class `TestIngestionPipeline` at line 37;
fixtures `settings` (40), `mock_embedder` (64), `pipeline` (74) — reuse them)

1. `test_ingest_prunes_manifest_entries_for_missing_files`: ingest one real md file
   (`sources/a.md`) via `pipeline.ingest()`; then delete `sources/a.md`; add a second real
   file `sources/b.md`; call `pipeline.ingest()` again (global, no path). Assert
   `manifest.get_entry("a.md") is None`, `store.delete_chunks` called for its chunk_ids,
   and `get_entry("b.md")` present. (`store` is the real vector store from settings —
   check conftest/pipeline fixture wiring at test_ingest.py:74; if store is real Qdrant,
   patch `pipeline.store` with a MagicMock before the second ingest, as existing tests do.)
2. `test_ingest_prune_skipped_when_nothing_scanned`: manifest has entry `a.md`,
   `sources/` is empty; `pipeline.ingest()`; assert entry still present (guard:
   `files_scanned > 0` required).
3. `test_ingest_prune_keeps_absolute_path_entries`: manifest entry keyed by absolute path
   outside sources_dir (insert directly: `manifest.add_entry(SourceEntry(path=str(abs_path), ...))`);
   with one real file scanned, run ingest; assert absolute entry survives.

### 1.4 RED verification

```
.venv/bin/pytest tests/test_cli.py::TestConvertCommand tests/test_parser.py tests/test_ingest.py -v
```
Expected: new/rewritten tests FAIL against current code; pre-existing kept tests still PASS.

---

## Phase 2 — GREEN: implement

### 2.1 `src/grounded_code_mcp/parser.py`

1. Delete `sidecar_path()` — lines 110–119.
2. Delete sidecar-preference branch in `parse()` — lines 209–217.
3. Delete sidecar-skip in `scan_directory()` — lines 483–486.

### 2.2 `src/grounded_code_mcp/__main__.py` — rewrite `convert` (lines 251–355)

1. Remove `@click.option("--collection", ...)` (line 252) and `collection` param (257).
   Add module-level helper (place right above `convert`):
   ```python
   def _staged_dest(file: Path, root: Path, sources_dir: Path) -> Path:
       """Staged .md destination: sources_dir / file.relative_to(root).with_suffix('.md')."""
       rel = file.resolve().relative_to(root.resolve())  # ValueError if outside root
       return sources_dir / rel.with_suffix(".md")
   ```
2. Root resolution, one place: `root = Path(os.environ["GROUNDED_CONVERT_ROOT"])` if set;
   elif `path` is a file → `Path(path).parent.parent`; elif `path` is a dir → `Path(path)`;
   else `settings.knowledge_base.sources_dir`.
3. File selection: dir → `[f for f in scan_directory(root) if f.suffix.lower() not in PLAINTEXT_EXTENSIONS]`;
   file → `[p] if p.suffix.lower() not in PLAINTEXT_EXTENSIONS else []`.
4. Per file: `staged = _staged_dest(file, root, settings.knowledge_base.sources_dir)`
   inside try/except `ValueError` → print Failed, `failed += 1`, continue.
5. Skip if `staged.exists() and not force` → `skipped += 1`.
6. `--dry-run` → print `Would convert: {file}`, `converted += 1`, continue.
7. Write: `staged.parent.mkdir(parents=True, exist_ok=True)`;
   `tmp = staged.with_suffix(".md.tmp")`; `tmp.write_text(result.content, encoding="utf-8")`;
   `os.replace(tmp, staged)`. (`import os` at top.)
8. Single-file mode: parse via `DocumentParser(enable_ocr=enable_ocr, docling_settings=settings.docling)`
   (existing pattern, lines 300–303).
9. Dir mode: `subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "GROUNDED_CONVERT_ROOT": str(root)})`
   with `cmd = [sys.executable, "-m", "grounded_code_mcp", "convert", ("--no-ocr" if disable_ocr), ("--force" if force), str(file)]`
   (filter Nones). Keep rc-based failed counting and stderr print (lines 338–350).
10. Keep summary print (line 353).

### 2.3 `src/grounded_code_mcp/ingest.py` — guarded prune (insertion point: after line 133
`stats.files_scanned = len(files)`, before the `for file_path in files:` loop at ~136)

```python
# Prune stale manifest entries (global ingest only, guarded against empty scans).
if path is None and stats.files_scanned > 0:
    for key in list(self.manifest.sources.keys()):
        p = Path(key)
        if p.is_absolute():
            continue  # outside sources_dir — never prune
        if not (self.settings.knowledge_base.sources_dir / p).exists():
            entry = self.manifest.sources[key]
            if entry.chunk_ids:
                self.store.delete_chunks(entry.collection, entry.chunk_ids)
            self.manifest.remove_entry(key)
            logger.info("Pruned stale manifest entry: %s", key)
```
(Uses existing `manifest.sources` dict — manifest.py:89, `remove_entry` at 125, and the same
`_get_relative_path`-style absolute check as ingest.py:374.)

### 2.4 GREEN verification

```
.venv/bin/pytest tests/test_cli.py::TestConvertCommand tests/test_parser.py tests/test_ingest.py -v
```
Expected: all pass.

---

## Phase 3 — Full quality gate

```
.venv/bin/pytest
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/mypy src/
.venv/bin/bandit -r src/ -c pyproject.toml
```
Fix any fallout. `os.replace` needs no new import beyond `os`; `subprocess.run` already has
`nosec` annotations — the added `env=` kwarg does not change that.

---

## Phase 4 — Docs / config / scripts (mechanical edits)

1. `.gitignore`: delete only the comment line
   `# Docling pre-converted Markdown sidecars are already covered by sources/*/*` and the
   following `# sources/**/*.pdf.md ...` line. No pattern changes.
2. `scripts/sync-sources.sh`: find the rsync/scp filter that ships binaries (grep for
   `pdf|epub`); change to ship only `*.md` under `sources/`.
3. `scripts/_ingest_check.py`: line ~21 `INGESTABLE` import from
   `grounded_code_mcp.parser import SUPPORTED_EXTENSIONS` → import `PLAINTEXT_EXTENSIONS`;
   fallback set (lines 23–26) → `{".md", ".markdown", ".mdx", ".rst", ".txt", ".asciidoc", ".adoc"}`.
4. `README.md` lines 146–159: replace the `convert` examples block — remove the
   `convert --collection rust` line (152); rewrite the paragraph (148) to: "`convert <raw-path>`
   parses binary documents with Docling and stages Markdown into `sources/<collection>/…`;
   run `ingest` afterwards to index the staged files."
5. `ARCHITECTURE.md`: line 122 table row — remove `--collection` from the `convert [PATH]`
   options cell; lines 34/55/63–64 — replace "sidecar" wording with "staged Markdown".
6. `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md`: grep `sidecar`; replace each mention with the
   staging flow sentence from README (keep it to one line each).
7. `config.toml`: no change.

---

## Rollback

- Code: per-phase `git checkout <file>` for touched source/test files.
- Data: prune (2.3) is the only destructive step; guarded (never runs on empty scan).
  Rollback = restore old `sources/` layout + `ingest --force`, or
  `rebuild_collection <name>` per affected collection.

---

## Acceptance criteria (yes — added; they are the definition of done)

1. `grounded-code-mcp convert --help` shows only `--force`, `--dry-run`, `--no-ocr` and the
   `PATH` argument (no `--collection`, no `--root`).
2. `grep -rn sidecar src/ tests/` returns nothing.
3. `convert <raw-dir>` with nested `raw/<coll>/folder/file.pdf` stages to
   `sources/<coll>/folder/file.md` (verified by `--dry-run` output and by one real
   single-file run in tests).
4. Re-running `convert` on the same raw dir without `--force` prints `N skipped`, files
   unchanged; with `--force` they are overwritten.
5. Global `ingest` after deleting a staged `.md` removes its manifest entry and chunks;
   global `ingest` on an empty `sources/` leaves the manifest untouched.
6. Full gate green: pytest, ruff check, ruff format --check, mypy, bandit (Phase 3 commands).
7. Docs/scripts: `grep -rn "sidecar\|--collection.*convert" README.md ARCHITECTURE.md
   CLAUDE.md AGENTS.md CONTRIBUTING.md scripts/` returns nothing.

## Notes

- `.venv` may lack dev tooling: `.venv/bin/pip install -e ".[all,dev]"` first.
- Real GPU Docling staging runs are human operator steps after Phase 3 — out of scope here.
