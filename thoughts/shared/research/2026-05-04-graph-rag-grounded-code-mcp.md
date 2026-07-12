---
date: 2026-05-04T00:00:00
repository: grounded-code-mcp
topic: "Graph RAG functionality"
tags: [research, graph-rag, knowledge-graph, vector-search, hybrid-retrieval]
git_commit: 399b423e06aabf5bd34c9fefa575e4c3e016c4dd
status: complete
---

# Research: Graph RAG Functionality

## Research question

How is the concept graph built, stored, queried, and integrated with vector search to produce
hybrid Graph RAG results in grounded-code-mcp?

## Summary

The Graph RAG layer is a **concept-graph overlay on top of vector search** — it does not replace
vector retrieval, it expands it. When a query matches a concept node in the graph, the system
traverses the graph neighborhood to discover related `source_slug` values, then performs a second
broader vector search limited to those sources. Results are labeled `[vector]` or
`[graph-expanded: via <concept>]` and merged.

The graph itself is a **hand-authored, static directed graph** of concept triples parsed from
`RELATIONSHIPS.md` files co-located with source document directories. There are no LLM or NLP
calls anywhere in the graph pipeline — construction is pure regex parsing, and query-time concept
matching is pure string/slug comparison. The graph is persisted as a NetworkX node-link JSON file
(`graph/concept_graph.json`) and hot-reloaded on mtime change at query time.

The graph module (`src/graph/`) is a first-class installable package alongside
`src/grounded_code_mcp/`. Graph construction is triggered only by `ingest --force`; normal ingest
never touches the graph. The `query_graph` MCP tool exposes pure graph traversal; hybrid retrieval
is embedded inside `search_knowledge`.

## Detailed findings

### 1. Graph data model

Nodes are plain `dict[str, Any]` stored as NetworkX node attributes (`graph_store.py:138`):

```
id: str          # slugified concept, e.g. "test-driven-development"
source_slug: str # collection/source dir slug
domain: str      # one of VALID_DOMAINS (architecture, testing, data-access, …)
type: str        # one of VALID_TYPES
description: str # optional free-text
```

Edges (`graph_store.py:197`): `{"from": str, "rel": str, "to": str}` (output form);
`{"source": str, "target": str, "rel": str}` (input form to `merge_nodes`).

`VALID_DOMAINS` (`graph_store.py:19`), `VALID_TYPES` (`graph_store.py:31`), and
`VALID_RELATIONS` (`graph_store.py:41`) are defined as frozensets but **not enforced** during
parse or merge — they are reference constants only.

### 2. Graph construction pipeline

No LLM or NLP. Entirely hand-authored and regex-parsed.

1. **Author** `RELATIONSHIPS.md` in each source dir (two accepted triple formats —
   `graph_builder.py:16–36`)
2. **Seed bootstrap** via `seed_graph.py:generate_seed_content()` — hardcoded starter triples for
   8 known slugs; unknown slugs get a placeholder template (`seed_graph.py:28–134`)
3. **Parse** via `_parse_triples()` (`graph_builder.py:51`) — regex line-by-line; prose lines and
   fence markers silently skipped; relations normalized to lowercase
4. **Populate** via `build(input_path, store)` (`graph_builder.py:124`) — recursively finds all
   `RELATIONSHIPS.md`, calls `store.remove_source()` then `store.merge_nodes()` per file,
   then `store.save()`

### 3. Graph storage

- **Backend**: NetworkX `DiGraph` in memory, persisted as JSON via `node_link_data/node_link_graph`
  (`graph_store.py:125, 130`)
- **Default path**: `graph/concept_graph.json` relative to CWD (`graph_store.py:17`)
- **Override**: `GRAPH_JSON_PATH` env var, validated for `.json` extension and no path traversal
  (`graph_store.py:61–84`)
- **Hot reload**: server compares file mtime on every call; reloads without restart
  (`server.py:148–190`)
- **Live file**: `graph/concept_graph.json` (~32,399 lines) is present in the repo

### 4. `query_graph` MCP tool

Signature (`server.py:665`):
```python
def query_graph(concept: str, depth: int = 2, domain: str | None = None) -> dict[str, Any]
```

Implementation `_query_graph_impl` (`server.py:507`):
1. Load/hot-reload graph
2. Match query → concept IDs via `_extract_concept_ids()` (three-tier: exact slug, substring,
   per-word fallback) (`server.py:193`)
3. Take `primary_id = concept_ids[0]` as traversal root
4. `graph.get_neighbors(primary_id, depth=depth)` — bidirectional BFS (`graph_store.py:146`)
5. Optional domain filter post-traversal (`server.py:544`)
6. `graph.get_edges_for_nodes(all_node_ids)` for edges within result set
7. Build inline summary string — **no LLM call** (`server.py:551`)

Return shape: `{matched_nodes, relationships, linked_sources, summary}`

Depth clamped to `[1, MAX_GRAPH_DEPTH=3]` (`server.py:29, 513`). No breadth cap.

### 5. Hybrid retrieval in `search_knowledge`

Two-pass vector search with graph-guided source expansion (`server.py:252–365`):

**Pass 1** — direct vector search, top `n_results`, track `seen_chunk_ids`

**Pass 2** — graph expansion:
1. `_extract_concept_ids(query, graph)` → matched concept IDs
2. `graph.get_neighbors(concept_id, depth=2)` for each concept
3. Collect `source_slug` values from neighbors → `expanded_slugs` dict (slug → via-concept)
4. Broader second vector search: `n_results = (n_results + expansion_n_results) * 3`
5. Filter hits by `source_path` prefix matching `expanded_slugs`; exclude `seen_chunk_ids`
6. Sort expansion hits by vector score; take top `expansion_n_results` (default 5)

**Merge**: vector results first (`[vector]`), expansion appended (`[graph-expanded: via <concept>]`).
No re-ranking or score normalization between pools. Graph expansion fails silently (`server.py:353`).

### 6. Ingest pipeline integration

Graph rebuild only on `--force` (`ingest.py:170`):
- `ingest PATH --force` → `_rebuild_graph_single()` (`ingest.py:291`) — one `RELATIONSHIPS.md`
- `ingest --force` (no path) → `_rebuild_graph_global()` (`ingest.py:311`) — all files recursively
- Missing `RELATIONSHIPS.md` → `WARNING` log, no failure (`ingest.py:299`)
- Normal ingest never touches the graph

### 7. Configuration surface

No `[graph]` section in `config.toml`. All graph config is:

| Mechanism | Value | Location |
|---|---|---|
| `GRAPH_JSON_PATH` env var | Override JSON path | `graph_store.py:91` |
| `MAX_GRAPH_DEPTH` | `3` (hardcoded) | `server.py:29` |
| `EXPANSION_N_RESULTS_DEFAULT` | `5` (hardcoded) | `server.py:30` |

## Code references

### Core graph module
- `src/graph/graph_store.py` — `GraphStore`, `slugify`, `VALID_*` constants, persistence
- `src/graph/graph_builder.py` — `build()`, `_parse_triples()`, `BuildStats`
- `src/graph/seed_graph.py` — `seed()`, `generate_seed_content()`, `SeedStats`
- `src/graph/__init__.py` — package marker (empty)

### Server integration
- `src/grounded_code_mcp/server.py:148` — `_get_graph_store()` (mtime hot-reload)
- `src/grounded_code_mcp/server.py:193` — `_extract_concept_ids()`
- `src/grounded_code_mcp/server.py:252` — `_search_knowledge_impl()` (hybrid retrieval)
- `src/grounded_code_mcp/server.py:507` — `_query_graph_impl()`
- `src/grounded_code_mcp/server.py:665` — `query_graph` MCP tool handler

### Ingest integration
- `src/grounded_code_mcp/ingest.py:291` — `_rebuild_graph_single()`
- `src/grounded_code_mcp/ingest.py:311` — `_rebuild_graph_global()`

### CLI
- `src/grounded_code_mcp/__main__.py:334` — `build-graph` Click command

### Storage
- `graph/concept_graph.json` — live NetworkX node-link JSON (~32k lines)
- `sources/databases/RELATIONSHIPS.md` — only committed source triple file

### Tests
- `tests/test_graph_store.py` — `GraphStore` unit tests (437 lines)
- `tests/test_graph_builder.py` — `_parse_triples` and `build()` (499 lines)
- `tests/test_seed_graph.py` — `seed_graph` module (312 lines)
- `tests/test_ingest_graph.py` — ingest pipeline graph rebuild (339 lines)
- `tests/test_server_graph.py` — hybrid search and `query_graph` MCP tool (728 lines)

## Key design patterns

1. **Silent degradation**: graph unavailability never raises — `_get_graph_store()` returns `None`,
   expansion is skipped, pure vector results returned (`server.py:148–190, 353`)
2. **Mtime hot-reload**: graph JSON reloaded on file change without server restart, same pattern
   as manifest reload (`server.py:176–185`)
3. **Deferred imports**: graph module imports inside method bodies in `ingest.py` and
   `__main__.py` — no hard dependency at module load time
4. **Two-pass hybrid**: vector-first, graph-expanded-second; no score fusion; expansion hits
   ranked by their own vector scores (`server.py:300–364`)
5. **Autouse fixture isolation**: `conftest.py:11–21` patches `_get_graph_store` to `None` for
   all legacy tests, preventing graph expansion from corrupting pre-graph call-count assertions
6. **Edge dict key asymmetry**: `merge_nodes` input uses `source/target/rel`; `get_edges_for_nodes`
   output uses `from/rel/to` (`graph_store.py:197, 235`)
7. **No TOML config for graph**: all graph parameters are hardcoded constants or env vars — no
   `[graph]` section in `config.toml`
8. **Regex-only concept extraction**: no LLM or NLP at any stage; concept matching at query time
   is slug substring matching (`server.py:193–224`)

## Open questions

1. **`VALID_RELATIONS` not enforced**: `graph_store.py:41–49` defines a frozenset of valid
   relation types but `_parse_triples` accepts any lowercase string. Should enforcement be added
   to `merge_nodes` or the parser? Or is the open vocabulary intentional?
2. **No breadth cap in `get_neighbors`**: at `depth=3` on a dense graph, the result set could be
   very large. Is a max-node cap needed, or is the current graph sparse enough that this is not
   a concern?
3. **`query_graph` uses only `primary_id = concept_ids[0]`** (`server.py:534`): if multiple
   concepts match the query, only the first is traversed. Should multi-concept traversal be
   supported, or is single-root intentional?
4. **Graph construction is fully manual**: `RELATIONSHIPS.md` files must be hand-authored or
   seeded. Is LLM-assisted concept/relation extraction from document chunks a planned feature,
   or is the static hand-authored approach a permanent design decision?
5. **`graph/concept_graph.json` is committed** (gitignored per recent commit `399b423`): the
   `.gitignore` now excludes it. Confirm whether the graph JSON should be treated as a build
   artifact (regenerated from `RELATIONSHIPS.md` on deploy) or as a committed snapshot.
