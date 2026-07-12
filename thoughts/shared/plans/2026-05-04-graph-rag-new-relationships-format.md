# Plan: Graph RAG — New RELATIONSHIPS.md Format

**Created:** 2026-05-04
**Research:** `thoughts/shared/research/2026-05-04-graph-rag-grounded-code-mcp.md`
**Status:** implemented — merged to `main` (Phase 1 GREEN `04923f0`, Phase 2 `e649059`/`afaa43f`). Branch since deleted.
**Superseded in part:** the 10-verb `VALID_RELATIONS` specced below was later expanded to 95 verbs in `b097744`; treat that commit as the authoritative vocabulary, not this plan.
**Ticket:** N/A
**Branch:** `feat/graph-rag-new-relationships-format`
**Git base:** `399b423`

---

## Overview

The new distillation prompts (`prompts/distill-new-source.md`, `prompts/retrofit-relationships.md`)
define a canonical RELATIONSHIPS.md format that differs from the original seeded format in three
ways: (1) parenthetical slugs only — `(slug) --[verb]--> (slug)`; (2) a controlled vocabulary of
10 approved relation verbs, normalized to hyphenated form in storage; (3) section headers carrying
a `<!-- domain: X -->` comment that supplies the domain for all triples in that section. This plan
updates the graph build pipeline and `query_graph` tool to enforce and exploit this richer format.

All 69+ existing RELATIONSHIPS.md files will be retrofitted and re-ingested with `--force` before
or after these code changes land. The plan is code-only; file regeneration is a human-run batch
step outside this plan's scope.

---

## Current state (from research)

- `VALID_RELATIONS` has 5 underscore-verbs (`depends_on`, `enables`, `conflicts_with`,
  `is_example_of`, `reinforces`) — never enforced during parse or merge.
- `VALID_DOMAINS` has 7 values unrelated to the new 15-domain taxonomy.
- `VALID_TYPES` has 5 values (`pattern`, `principle`, `practice`, `anti-pattern`, `constraint`).
- `_parse_triples()` accepts any relation string; domain is read from the third `[bracket]`
  tag on each triple line; there is no section-level domain inheritance.
- `_query_graph_impl()` traverses only `concept_ids[0]` (single root), ignoring any additional
  matched concepts.
- `seed_graph.py` emits quoted format triples that will fail validation once enforcement is added.
- `sources/databases/RELATIONSHIPS.md` uses quoted format and is the only committed source file.

---

## Desired end state

- `VALID_RELATIONS` contains the 10 approved hyphenated verbs from the new prompts; invalid
  verbs in parsed triples emit a WARNING and increment a `triples_invalid_rel` counter; the
  triple is skipped.
- `VALID_DOMAINS` contains the 15 approved domain values from the new prompts.
- `VALID_TYPES` retains the existing 4 values that still apply (`pattern`, `principle`,
  `practice`, `anti-pattern`); `constraint` is removed.
- `_parse_triples()` reads `## Section Name  <!-- domain: X -->` headers and applies the
  domain to all subsequent triples until the next section header.
- Relation verbs (both quoted and parenthetical formats) are normalized to hyphenated form
  before storage, e.g. `"depends on"` → `"depends-on"`, `"IS_EXAMPLE_OF"` → `"is-example-of"`.
- `_query_graph_impl()` traverses all matched concept IDs (not just the first), merging
  their neighborhoods; the `query_graph` tool docstring updated to reflect this.
- `seed_graph.py` emits parenthetical-format triples with domain-tagged section headers and
  approved verbs only.
- `sources/databases/RELATIONSHIPS.md` converted to canonical parenthetical format.
- All existing tests pass; new test cases cover the changed behaviors.

---

## What we're NOT doing

- We are NOT adding enforcement to `merge_nodes()` in `GraphStore` — validation stays in the
  parser layer only.
- We are NOT changing the graph JSON schema, persistence format, or hot-reload mechanism.
- We are NOT running the batch retrofit of the 69 existing RELATIONSHIPS.md files — that is a
  human-run step using the prompts.
- We are NOT adding a `[graph]` section to `config.toml` — graph parameters stay as constants
  and env vars.
- We are NOT changing hybrid retrieval logic in `_search_knowledge_impl()` — graph expansion
  source matching is unaffected by this plan.
- We are NOT renaming edge `rel` keys in `get_edges_for_nodes()` output (`from`/`rel`/`to`
  stays as-is).
- We are NOT deprecating the quoted format parser — both formats continue to parse; only
  enforcement (verb validation) is new.

---

## Implementation approach

Five phases in dependency order: constants first (other phases depend on knowing the valid
vocabulary), then parser behavior (the core behavioral change), then seed generation (depends
on knowing the canonical format), then query tool improvement (depends on richer domain data
being reliably present), then the one committed source file (smoke-test of the full pipeline).

Each phase is RED → GREEN: failing tests written first, then minimal implementation to pass.

---

## Phase 1 — Update `VALID_*` constants in `graph_store.py`

**Overview:** Replace the three frozensets with values matching the new prompt vocabulary.
Constants are still reference-only at this phase — no enforcement added yet.

### Step 1: Write failing tests

**File:** `tests/test_graph_store.py`

Add test class `TestValidConstants`:

- `test_valid_relations_contains_approved_verbs` — assert each of the 10 hyphenated verbs
  is in `VALID_RELATIONS`:
  `enables`, `depends-on`, `conflicts-with`, `is-an-example-of`, `requires`,
  `prevents`, `alternative-to`, `causes`, `improves`, `replaces`
- `test_valid_relations_does_not_contain_old_underscore_verbs` — assert `reinforces`,
  `depends_on`, `conflicts_with`, `is_example_of` are NOT in `VALID_RELATIONS`
- `test_valid_domains_contains_new_taxonomy` — assert each of the 15 new domain values
  is in `VALID_DOMAINS`:
  `testing`, `dotnet`, `python`, `rust`, `databases`, `api-design`, `edge-ai`,
  `automation`, `architecture`, `security`, `javascript`, `ui-ux`, `systems-thinking`,
  `robotics`, `php`
- `test_valid_domains_does_not_contain_old_values` — assert `data-access`,
  `agent-behavior`, `quality`, `patterns`, `constraints` are NOT in `VALID_DOMAINS`
- `test_valid_types_does_not_contain_constraint` — assert `constraint` NOT in `VALID_TYPES`
- `test_valid_types_contains_four_values` — assert `VALID_TYPES` has exactly 4 members:
  `pattern`, `principle`, `practice`, `anti-pattern`

**Verification (RED):**
```
.venv/bin/pytest tests/test_graph_store.py::TestValidConstants -v
```
Expected: all new tests FAIL.

### Step 2: Implement

**File:** `src/graph/graph_store.py`

Replace `VALID_RELATIONS` frozenset (lines 41–49):
```
REMOVE: frozenset{"depends_on", "enables", "conflicts_with", "is_example_of", "reinforces"}
ADD:    frozenset{
            "enables", "depends-on", "conflicts-with", "is-an-example-of",
            "requires", "prevents", "alternative-to", "causes", "improves", "replaces",
        }
```

Replace `VALID_DOMAINS` frozenset (lines 19–29):
```
REMOVE: frozenset{"architecture", "testing", "data-access", "agent-behavior",
                  "quality", "patterns", "constraints"}
ADD:    frozenset{
            "testing", "dotnet", "python", "rust", "databases", "api-design",
            "edge-ai", "automation", "architecture", "security", "javascript",
            "ui-ux", "systems-thinking", "robotics", "php",
        }
```

Replace `VALID_TYPES` frozenset (lines 31–39):
```
REMOVE: frozenset{"pattern", "principle", "practice", "anti-pattern", "constraint"}
ADD:    frozenset{"pattern", "principle", "practice", "anti-pattern"}
```

**Verification (GREEN):**
```
.venv/bin/pytest tests/test_graph_store.py::TestValidConstants -v
.venv/bin/pytest tests/test_graph_store.py -v
```
Expected: all tests PASS (including pre-existing tests — constants are reference-only, no
behavioral change in `GraphStore` methods).

**Implementation note:** Phase 1 is a pure constant replacement. No other files change.
Phases 2–5 depend on these constants being correct first.

---

## Phase 2 — Update `_parse_triples()`: section-domain inheritance + verb normalization + enforcement

**Overview:** Three related behavioral changes in `graph_builder.py`, delivered together
because they all touch `_parse_triples()` and the changes interact:
1. Parse `<!-- domain: X -->` from section headers → propagate to triples in that section.
2. Normalize all relation verbs to hyphenated lowercase (`depends on` → `depends-on`,
   `IS_EXAMPLE_OF` → `is-an-example-of`... well, map via normalization — see note below).
3. Enforce: if the normalized verb is not in `VALID_RELATIONS`, emit WARNING and skip the
   triple (increment `triples_invalid_rel` in `BuildStats`).

**Normalization rule:** Replace any run of spaces or underscores with a hyphen, then
lowercase. E.g. `"IS_EXAMPLE_OF"` → `"is-example-of"` (not `"is-an-example-of"` — the
approved verb is `"is-an-example-of"`, so `is-example-of` would fail validation and be
skipped as invalid. The human authoring the RELATIONSHIPS.md is responsible for writing
`is an example of` which normalizes to `is-an-example-of`.)

### Step 1: Write failing tests

**File:** `tests/test_graph_builder.py`

Add to `TestParseTriples`:

- `test_section_header_domain_applied_to_following_triples` — content with
  `## Testing  <!-- domain: testing -->` header followed by a triple; assert node domain
  is `"testing"`.
- `test_section_header_domain_overrides_previous_section` — two sections with different
  domain tags; assert triples in each section carry their respective section domain.
- `test_section_domain_does_not_override_triple_inline_domain_when_present` — a triple
  that has an explicit `[domain]` bracket tag inside a domain-tagged section; assert
  the **inline triple domain takes precedence** over the section domain. (Design choice:
  inline tag wins, section tag is a default.)
- `test_relation_verb_normalized_to_hyphenated` — parenthetical triple with
  `--[DEPENDS_ON]-->` or `--[depends on]-->`; assert stored `rel` is `"depends-on"`.
- `test_relation_verb_multi_word_normalized` — `--[IS AN EXAMPLE OF]-->`; assert
  stored `rel` is `"is-an-example-of"`.
- `test_invalid_relation_verb_emits_warning_and_skips` — triple with
  `--[INVENTED_VERB]-->`; assert `skipped == 0` but `triples_invalid_rel == 1`
  and no edge produced. (It is NOT a malformed triple — it parses but fails vocab check.)
- `test_invalid_relation_verb_increments_build_stats` — call `build()` with a file
  containing one valid and one invalid-verb triple; assert
  `stats.triples_invalid_rel == 1` and `stats.triples_parsed == 1`.
- `test_any_relation_is_accepted` (existing, line 93) — **update**: this test currently
  asserts `invented_rel` is accepted. Change the assertion to expect `skipped == 0` but
  `triples_invalid_rel == 1` and no edge. The test name can stay.

**Verification (RED):**
```
.venv/bin/pytest tests/test_graph_builder.py::TestParseTriples -v
```
Expected: new tests FAIL; existing tests that assert `invented_rel` passes will also FAIL.

### Step 2: Implement

**File:** `src/graph/graph_builder.py`

**Change 1 — `BuildStats`:** Add field `triples_invalid_rel: int = 0` to the dataclass
(after `triples_skipped`).

**Change 2 — Section-header regex:** Add a compiled regex constant before `_parse_triples`:
```python
_SECTION_DOMAIN_RE = re.compile(r"##[^<]+<!--\s*domain:\s*([\w-]+)\s*-->")
```

**Change 3 — `_parse_triples()` signature:** Add `stats` parameter:
```
REMOVE: def _parse_triples(content, default_source_slug) -> tuple[list, list, int]
ADD:    def _parse_triples(content, default_source_slug, *, stats=None) -> tuple[list, list, int]
```
(Pass `stats` from `build()` so `triples_invalid_rel` can be incremented there.)

**Change 4 — Loop body changes in `_parse_triples()`:**

At the top of the loop, before the `if not line` check, add:
```python
# Track current section domain from headers like: ## Foo  <!-- domain: X -->
if line.startswith("#"):
    m_sec = _SECTION_DOMAIN_RE.search(line)
    if m_sec:
        current_section_domain = m_sec.group(1).strip()
    continue
```
Add `current_section_domain: str = ""` initialization before the loop.

After extracting `rel` and before appending nodes/edges, add verb normalization:
```python
# Normalize: spaces and underscores → hyphens, then lowercase
rel_normalized = re.sub(r"[\s_]+", "-", rel).lower()
```

Add enforcement after normalization (import `VALID_RELATIONS` from `graph_store` — it is
already imported via `GraphStore`; add `VALID_RELATIONS` to the import line):
```python
if rel_normalized not in VALID_RELATIONS:
    logger.warning("Invalid relation verb %r (normalized: %r) — skipping triple", rel, rel_normalized)
    if stats is not None:
        stats.triples_invalid_rel += 1
    continue
rel = rel_normalized
```

Add section-domain inheritance when setting node domain. Current code reads
`raw_domain` from the fourth bracket tag. Change:
```
REMOVE: domain = raw_domain.strip() if raw_domain and raw_domain.strip() else ""
ADD:    domain = raw_domain.strip() if raw_domain and raw_domain.strip() else current_section_domain
```

**Change 5 — `build()` body:** Pass `stats` into `_parse_triples()` calls and aggregate
`triples_invalid_rel`:
```python
nodes, edges, skipped = _parse_triples(content, default_source_slug=source_slug, stats=stats)
stats.triples_skipped += skipped
# triples_invalid_rel already incremented inside _parse_triples via stats reference
```

**Change 6 — `build()` CLI print output:** Add `triples_invalid_rel` to the summary line:
```
REMOVE: f", {stats.nodes_added} nodes added"
ADD:    f", {stats.nodes_added} nodes added, {stats.triples_invalid_rel} invalid verbs skipped"
```

**Verification (GREEN):**
```
.venv/bin/pytest tests/test_graph_builder.py -v
.venv/bin/pytest tests/test_graph_store.py -v
```
Expected: all tests PASS.

**Full suite check:**
```
.venv/bin/pytest -v
```
Expected: no regressions.

**Implementation note:** The existing test `test_any_relation_is_accepted` (line 93) asserts
that `invented_rel` passes through without skipping. This test must be updated in Step 1
before Step 2 can turn green. The two changes are intentionally coupled in this phase.

---

## Phase 3 — Update `seed_graph.py` to emit canonical format

**Overview:** Rewrite all `_SEED_CONTENT` entries and `_generic_template()` to use the
parenthetical format with approved verbs, domain-tagged section headers, and no quoted
format. The public API (`seed()`, `generate_seed_content()`,
`find_sources_missing_relationships()`) is unchanged.

### Step 1: Write failing tests

**File:** `tests/test_seed_graph.py`

Add test class `TestSeedContentFormat`:

- `test_seed_content_uses_parenthetical_format` — call `generate_seed_content("python")`
  (or any well-known slug); assert the returned string contains `--[` and `-->` and does
  NOT contain `" → "` (quoted format arrow).
- `test_seed_content_contains_domain_tagged_section_header` — assert the returned string
  matches `re.search(r"##[^<]+<!--\s*domain:", content)`.
- `test_seed_content_relations_are_valid` — for each known slug in `_SEED_CONTENT`, call
  `generate_seed_content(slug)`, parse with `_parse_triples`, and assert
  `triples_invalid_rel == 0` (import the updated `_parse_triples` and pass a dummy
  `BuildStats`).
- `test_generic_template_uses_parenthetical_format` — call `generate_seed_content("unknown-slug")`;
  assert contains `--[` and `-->` and does NOT contain `" → "`.

**Verification (RED):**
```
.venv/bin/pytest tests/test_seed_graph.py::TestSeedContentFormat -v
```
Expected: new tests FAIL; pre-existing `test_seed_graph.py` tests may or may not fail
depending on whether they check format — inspect output and note which fail.

### Step 2: Implement

**File:** `src/graph/seed_graph.py`

Rewrite each entry in `_SEED_CONTENT` dict. For each collection, produce a block with:
- At least one domain-tagged section header: `## Section  <!-- domain: DOMAIN -->`
- All triples in parenthetical format with approved verbs

Example replacement for `"internal"`:
```
## Engineering Practices  <!-- domain: testing -->

(test-driven-development) --[enables]--> (refactoring)
(red-green-refactor) --[is-an-example-of]--> (test-driven-development)
(continuous-integration) --[enables]--> (fast-feedback)
(trunk-based-development) --[enables]--> (continuous-integration)
(pair-programming) --[enables]--> (knowledge-sharing)

## Architecture  <!-- domain: architecture -->

(clean-architecture) --[depends-on]--> (dependency-inversion)
(dependency-inversion) --[enables]--> (testability)
(yagni) --[conflicts-with]--> (speculative-abstraction)
(simple-design) --[improves]--> (test-driven-development)
(boy-scout-rule) --[improves]--> (simple-design)
```

Apply the same pattern to all 8 existing known slugs: `internal`, `patterns`,
`architecture`, `dotnet`, `python`, `databases`, `rust`, `edge-ai`.

Rewrite `_generic_template()`:
```
REMOVE: f'"[Concept A]" → enables → "[Concept B]" [{source_slug}] [architecture] [principle]\n...'
ADD:    return (
    "## Core Concepts  <!-- domain: architecture -->\n\n"
    f"(concept-a) --[enables]--> (concept-b)\n"
    f"(concept-b) --[depends-on]--> (concept-c)\n"
    "# TODO: replace placeholder concepts with real relationships from this source\n"
)
```

**Verification (GREEN):**
```
.venv/bin/pytest tests/test_seed_graph.py -v
```
Expected: all tests PASS.

**Full suite check:**
```
.venv/bin/pytest -v
```
Expected: no regressions.

**Implementation note:** Phase 3 only touches `seed_graph.py` and its tests. The seeded
content no longer needs the quoted-format parser path to be exercised — but that path
remains in `_parse_triples()` for backward compatibility with any legacy files.

---

## Phase 4 — Update `_query_graph_impl()` to traverse all matched concepts

**Overview:** Currently `_query_graph_impl()` uses only `concept_ids[0]` as the traversal
root, discarding any additional matched concepts. With domain reliably populated via section
headers, two improvements are warranted: (a) traverse all matched concept IDs and merge
their neighborhoods; (b) expose `all_concept_ids` in the return payload so callers can see
what was matched. The inline summary is also updated to report all traversal roots.

### Step 1: Write failing tests

**File:** `tests/test_server_graph.py`

Add to `TestQueryGraphKnownConcept`:

- `test_multi_concept_match_traverses_all_roots` — graph with three nodes: `tdd`,
  `tdd-cycle`, `refactoring`; edge `tdd` → `refactoring`; edge `tdd-cycle` → `refactoring`.
  Query `"tdd"` — should match both `tdd` and `tdd-cycle` via substring. Assert both
  `tdd` and `tdd-cycle` appear in `matched_nodes` (i.e. both were used as roots).
- `test_matched_concept_ids_in_result` — query that matches one concept; assert the
  result dict contains key `"matched_concept_ids"` with a non-empty list.
- `test_multi_root_linked_sources_are_merged` — graph with two source slugs on two
  matched-but-unconnected roots; assert both sources appear in `linked_sources`.
- `test_summary_lists_all_matched_concepts` — query matching two concepts; assert both
  concept IDs appear in the `summary` string (or the summary mentions "2 concept(s) matched").

**Verification (RED):**
```
.venv/bin/pytest tests/test_server_graph.py::TestQueryGraphKnownConcept -v
```
Expected: new tests FAIL.

### Step 2: Implement

**File:** `src/grounded_code_mcp/server.py`

In `_query_graph_impl()` (starting line 507):

**Change 1 — Traverse all concept IDs, not just the first:**
```
REMOVE: primary_id = concept_ids[0]
        neighbors = graph.get_neighbors(primary_id, depth=depth)
        primary_candidates = [m for m in graph.search_nodes(primary_id) if m["id"] == primary_id]
        primary_node: dict[str, Any] = primary_candidates[0] if primary_candidates else {"id": primary_id}
        all_nodes: list[dict[str, Any]] = [primary_node, *neighbors]

ADD:    all_nodes_dict: dict[str, dict[str, Any]] = {}
        for cid in concept_ids:
            candidates = [m for m in graph.search_nodes(cid) if m["id"] == cid]
            root_node: dict[str, Any] = candidates[0] if candidates else {"id": cid}
            all_nodes_dict[cid] = root_node
            for neighbor in graph.get_neighbors(cid, depth=depth):
                all_nodes_dict.setdefault(neighbor["id"], neighbor)
        all_nodes: list[dict[str, Any]] = list(all_nodes_dict.values())
        # Keep primary_id for backward-compatible summary language
        primary_id = concept_ids[0]
        primary_node = all_nodes_dict.get(primary_id, {"id": primary_id})
```

**Change 2 — Add `matched_concept_ids` to return dict:**
```
REMOVE: return {
            "matched_nodes": [...],
            "relationships": relationships,
            "linked_sources": linked_sources,
            "summary": summary,
        }
ADD:    return {
            "matched_concept_ids": concept_ids,
            "matched_nodes": [...],
            "relationships": relationships,
            "linked_sources": linked_sources,
            "summary": summary,
        }
```

**Change 3 — Update summary to mention multi-root traversal:**
In the summary construction block, replace the single-root description:
```
REMOVE: parts = [f"The graph describes '{primary_id}' as a {node_type}"]
ADD:    if len(concept_ids) > 1:
            parts = [f"Graph traversal from {len(concept_ids)} matched concept(s): "
                     f"{', '.join(concept_ids[:3])}{'...' if len(concept_ids) > 3 else ''}"]
        else:
            parts = [f"The graph describes '{primary_id}' as a {node_type}"]
```

**Change 4 — Update `query_graph` MCP tool docstring:**
```
REMOVE: "Traversal depth from matched node (default 2, max 3)."
ADD:    "Traversal depth from each matched concept node (default 2, max 3). "
        "All concepts matching the query are used as traversal roots."
```

**Verification (GREEN):**
```
.venv/bin/pytest tests/test_server_graph.py -v
```
Expected: all tests PASS.

**Full suite check:**
```
.venv/bin/pytest -v
```
Expected: no regressions.

**Implementation note:** The `matched_concept_ids` key is additive — existing callers
reading `matched_nodes`, `relationships`, `linked_sources`, `summary` are unaffected.
Existing tests that check `result["matched_nodes"]` do not need updating; only tests that
assert the exact return dict keys need the new key added to their assertions if they check
exhaustively (review `test_server_graph.py` for any `== {...}` dict equality checks).

---

## Phase 5 — Convert `sources/databases/RELATIONSHIPS.md` to canonical format

**Overview:** The only committed source triple file is still in quoted format. Convert it to
canonical parenthetical format with domain-tagged section headers, using the approved verbs.
This is the pipeline smoke-test: run `build-graph --dry-run` before and after to confirm
zero skipped/invalid triples.

### Step 1: Dry-run baseline (no test file change)

```bash
.venv/bin/python -m graph.graph_builder \
  --input sources/databases \
  --dry-run
```
Note the current `triples_parsed`, `triples_skipped`, `triples_invalid_rel` counts.
Expected: `triples_invalid_rel > 0` because current verbs (`enables`, `is_example_of`)
include underscored forms that fail Phase 2 enforcement... actually `enables` IS valid and
`is_example_of` normalizes to `is-example-of` (not `is-an-example-of`), so several triples
will be flagged.

### Step 2: Rewrite the file

**File:** `sources/databases/RELATIONSHIPS.md`

Rewrite using canonical format. Map existing concepts and verbs:

```
REMOVE: <!-- SEED: auto-generated, review before committing -->
        "Index" → enables → "Query Performance" [databases] [data-access] [practice]
        "B-Tree Index" → is_example_of → "Index" [databases] [data-access] [pattern]
        ...

ADD:    ## Indexing  <!-- domain: databases -->

        (index) --[enables]--> (query-performance)
        (b-tree-index) --[is-an-example-of]--> (index)
        (covering-index) --[enables]--> (index-only-scan)

        ## Data Integrity  <!-- domain: databases -->

        (normalisation) --[enables]--> (data-integrity)
        (transaction) --[enables]--> (acid-guarantees)
        (write-ahead-log) --[enables]--> (crash-recovery)

        ## Scalability  <!-- domain: databases -->

        (replication) --[enables]--> (high-availability)
        (partitioning) --[enables]--> (horizontal-scalability)
        (connection-pooling) --[enables]--> (resource-efficiency)

        ## Security  <!-- domain: databases -->

        (prepared-statement) --[prevents]--> (sql-injection)
```

### Step 3: Verify dry-run passes cleanly

```bash
.venv/bin/python -m graph.graph_builder \
  --input sources/databases \
  --dry-run
```
Expected: `triples_invalid_rel == 0`, `triples_skipped == 0`.

### Step 4: Run the full test suite

```
.venv/bin/pytest -v
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/mypy src/
```
Expected: all pass.

**Implementation note:** Phase 5 is a data file change, not a code change. It serves as
proof that the Phase 2 enforcement works end-to-end on a real file. The dry-run is the
primary verification; no new test file is added for this phase.

---

## Testing strategy

| Phase | Test file | New test class / cases | Pre-existing coverage |
|---|---|---|---|
| 1 | `test_graph_store.py` | `TestValidConstants` (6 new cases) | All existing `GraphStore` tests must still pass |
| 2 | `test_graph_builder.py` | 7 new cases in `TestParseTriples`; 1 updated case | All existing builder + build() tests must pass |
| 2 | `test_graph_builder.py` | 1 new case in `TestBuild` for `triples_invalid_rel` stat | — |
| 3 | `test_seed_graph.py` | `TestSeedContentFormat` (4 new cases) | All existing seed tests must pass |
| 4 | `test_server_graph.py` | 4 new cases in `TestQueryGraphKnownConcept` | All existing query_graph + hybrid search tests must pass |

Full quality gate before opening PR:
```bash
.venv/bin/pytest --cov=grounded_code_mcp --cov=graph -v
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/mypy src/
.venv/bin/bandit -r src/ -c pyproject.toml
```

---

## Rollback plan

All changes are in source-controlled files. Each phase can be independently reverted:

| Phase | Rollback |
|---|---|
| 1 | `git checkout src/graph/graph_store.py` |
| 2 | `git checkout src/graph/graph_builder.py` |
| 3 | `git checkout src/graph/seed_graph.py` |
| 4 | `git checkout src/grounded_code_mcp/server.py` |
| 5 | `git checkout sources/databases/RELATIONSHIPS.md` |

The graph JSON (`graph/concept_graph.json`) is gitignored and regenerated from
`RELATIONSHIPS.md` files via `grounded-code-mcp ingest --force`. If the graph JSON needs
to be restored to a known-good state before this plan's changes, regenerate from the
current source files after reverting the code.

There is no database migration; no deployment step; no persistent state change outside the
graph JSON (which is a build artifact).

---

## Notes

- **Verb normalization edge case:** The verb `"is an example of"` (4 words) normalizes to
  `"is-an-example-of"` ✓. The old underscore form `"is_example_of"` normalizes to
  `"is-example-of"` ✗ (not in `VALID_RELATIONS`). This is intentional: old files with
  `is_example_of` will produce `triples_invalid_rel` warnings after Phase 2, which is the
  signal to the human operator that those files need retrofitting.
- **`test_any_relation_is_accepted` update:** This is the only pre-existing test that
  directly conflicts with Phase 2's new enforcement. It must be updated in Phase 2 Step 1
  (as part of the RED commit) before the GREEN implementation runs.
- **`test_paren_format_normalises_relation_to_lowercase` (line 136):** Currently asserts
  `edges[0]["rel"] == "optimises_for"` — an underscore form. After Phase 2 normalization,
  `OPTIMISES_FOR` → `optimises-for`, which is not in `VALID_RELATIONS`, so the triple is
  skipped. Update this test in Phase 2 Step 1 to use an approved verb (`ENABLES`,
  `DEPENDS_ON` → `depends-on`, etc.) or assert it is skipped with `triples_invalid_rel == 1`.
- **`test_multi_word_relation_quoted_format` (line 103) and `test_multi_word_relation_phrase_quoted_format` (line 113):** Currently assert `rel == "depends on"` and `rel == "is an example of"` (space-form). After Phase 2 normalization, these become `"depends-on"` and `"is-an-example-of"`. Update both assertions in Phase 2 Step 1.
- **`test_multi_word_predicate_paren_format` (line 195) and `test_list_item_prefix_stripped_before_parsing` (line 205):** Both use `REDUCES BLAST RADIUS OF` / `REDUCES_BLAST_RADIUS_OF` — not in `VALID_RELATIONS`. Update in Phase 2 Step 1 to assert the triple is counted as `triples_invalid_rel` and produces no edge.
- **`test_paren_format_normalises_relation_to_lowercase` and `test_hyphenated_relation_quoted_format` (line 185) (`pre-aggregates`):** `pre-aggregates` is not in `VALID_RELATIONS` → will be skipped. Update in Phase 2 Step 1.
- **After all 69+ RELATIONSHIPS.md files are retrofitted** (human batch step, outside this
  plan), run `grounded-code-mcp ingest --force` on each affected collection to rebuild the
  graph. The `triples_invalid_rel` counter in the CLI output will confirm clean ingestion.
