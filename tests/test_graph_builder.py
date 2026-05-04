"""Tests for GraphBuilder: triple parsing, idempotency, dry-run, CLI."""

from pathlib import Path

import pytest

from graph.graph_store import GraphStore


def _write_relationships(directory: Path, content: str) -> Path:
    """Write a RELATIONSHIPS.md file in directory and return its path."""
    rel_file = directory / "RELATIONSHIPS.md"
    rel_file.write_text(content, encoding="utf-8")
    return rel_file


def _make_store(temp_dir: Path) -> GraphStore:
    store = GraphStore(path=temp_dir / "graph.json")
    store.load()
    return store


# ---------------------------------------------------------------------------
# _parse_triples
# ---------------------------------------------------------------------------


class TestParseTriples:
    def test_valid_triple_unicode_arrow(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"Vertical Slice" → enables → "CQRS" [tidy-first] [architecture] [pattern] [Slices enable CQRS boundaries]'
        nodes, edges, skipped = _parse_triples(content, "tidy-first")

        assert skipped == 0
        node_ids = {n["id"] for n in nodes}
        assert "vertical-slice" in node_ids
        assert "cqrs" in node_ids
        assert len(edges) == 1
        assert edges[0]["rel"] == "enables"

    def test_valid_triple_ascii_arrow(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"Test-Driven Development" -> depends_on -> "Red-Green-Refactor"'
        _nodes, edges, skipped = _parse_triples(content, "xp")

        assert skipped == 0
        assert len(edges) == 1
        assert edges[0]["source"] == "test-driven-development"
        assert edges[0]["target"] == "red-green-refactor"

    def test_uses_default_source_slug_when_not_in_triple(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"Clean Architecture" → reinforces → "Dependency Inversion"'
        nodes, _edges, _skipped = _parse_triples(content, "clean-arch-book")

        assert all(n["source_slug"] == "clean-arch-book" for n in nodes)

    def test_overrides_source_slug_from_triple(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"Concept A" → enables → "Concept B" [external-source] [patterns] [pattern]'
        nodes, _edges, _skipped = _parse_triples(content, "default-slug")

        assert all(n["source_slug"] == "external-source" for n in nodes)

    def test_stores_domain_and_type_from_triple(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"CQRS" → enables → "Event Sourcing" [book] [architecture] [pattern]'
        nodes, _edges, _skipped = _parse_triples(content, "book")

        cqrs_node = next(n for n in nodes if n["id"] == "cqrs")
        assert cqrs_node["domain"] == "architecture"
        assert cqrs_node["type"] == "pattern"

    def test_malformed_triple_is_skipped_with_warning(
        self, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from graph.graph_builder import _parse_triples

        content = '"A" → → "B"'  # arrow but no relation token
        with caplog.at_level("WARNING"):
            nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 1
        assert len(nodes) == 0
        assert len(edges) == 0
        assert any("malformed" in r.message.lower() for r in caplog.records)

    def test_any_relation_is_accepted(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"A" → invented_rel → "B"'
        _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 1
        assert edges[0]["rel"] == "invented_rel"

    def test_multi_word_relation_quoted_format(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"Full Recovery Model" → depends on → "Transaction Log Backup"'
        _nodes, edges, skipped = _parse_triples(content, "databases")

        assert skipped == 0
        assert len(edges) == 1
        assert edges[0]["rel"] == "depends on"

    def test_multi_word_relation_phrase_quoted_format(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"BCP" → is an example of → "Bulk Data Transfer"'
        _nodes, edges, skipped = _parse_triples(content, "databases")

        assert skipped == 0
        assert len(edges) == 1
        assert edges[0]["rel"] == "is an example of"

    def test_valid_triple_paren_format(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = "(Reliability) --[IS_ACHIEVED_BY]--> (Fault Tolerance)"
        nodes, edges, skipped = _parse_triples(content, "ddia")

        assert skipped == 0
        node_ids = {n["id"] for n in nodes}
        assert "reliability" in node_ids
        assert "fault-tolerance" in node_ids
        assert len(edges) == 1
        assert edges[0]["rel"] == "is_achieved_by"

    def test_paren_format_normalises_relation_to_lowercase(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = "(LSM-Tree) --[OPTIMISES_FOR]--> (Write Throughput)"
        _nodes, edges, _skipped = _parse_triples(content, "ddia")

        assert edges[0]["rel"] == "optimises_for"

    def test_paren_format_slugifies_multiword_concepts(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = "(Snapshot Isolation) --[PREVENTS]--> (Read Skew)"
        nodes, _edges, _skipped = _parse_triples(content, "src")

        node_ids = {n["id"] for n in nodes}
        assert "snapshot-isolation" in node_ids
        assert "read-skew" in node_ids

    def test_prose_line_without_arrow_skipped_silently(
        self, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from graph.graph_builder import _parse_triples

        content = 'Source: some-book.pdf\nPurpose: graph-RAG ingestion.\n"A" → enables → "B"'
        with caplog.at_level("WARNING"):
            _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 1
        assert not any("malformed" in r.message.lower() for r in caplog.records)

    def test_horizontal_rule_skipped_silently(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '---\n"A" → enables → "B"\n---'
        _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 1

    def test_blockquote_metadata_skipped_silently(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '> Source: Some Book\n> For graph-RAG ingestion.\n"A" → enables → "B"'
        _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 1

    def test_hyphenated_relation_quoted_format(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"indexed view" → pre-aggregates → "data"'
        _nodes, edges, skipped = _parse_triples(content, "databases")

        assert skipped == 0
        assert len(edges) == 1
        assert edges[0]["rel"] == "pre-aggregates"

    def test_multi_word_predicate_paren_format(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = "(Least Privilege) --[REDUCES BLAST RADIUS OF]--> (Compromised Credential)"
        _nodes, edges, skipped = _parse_triples(content, "architecture")

        assert skipped == 0
        assert len(edges) == 1
        assert edges[0]["rel"] == "reduces blast radius of"

    def test_list_item_prefix_stripped_before_parsing(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = "- (Least Privilege) --[REDUCES_BLAST_RADIUS_OF]--> (Compromised Credential)"
        _nodes, edges, skipped = _parse_triples(content, "architecture")

        assert skipped == 0
        assert len(edges) == 1
        assert edges[0]["rel"] == "reduces_blast_radius_of"

    def test_backtick_wrapped_parens_parsed(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = "- `(Least Privilege)` --[REDUCES_BLAST_RADIUS_OF]--> `(Compromised Credential)`"
        _nodes, edges, skipped = _parse_triples(content, "architecture")

        assert skipped == 0
        assert len(edges) == 1

    def test_html_comment_skipped_silently(
        self, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from graph.graph_builder import _parse_triples

        content = '<!-- Distilled from: some-book.md -->\n"A" → enables → "B"'
        with caplog.at_level("WARNING"):
            _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 1
        assert not any("malformed" in r.message.lower() for r in caplog.records)

    def test_fenced_code_block_markers_skipped_silently(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = "```\n(A) --[ENABLES]--> (B)\n```"
        nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 1
        assert len(nodes) == 2

    def test_paren_format_inside_code_block_is_parsed(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = (
            "## Section\n"
            "```\n"
            "(Write-Ahead Log) --[ENABLES]--> (Crash Recovery)\n"
            "(Write-Ahead Log) --[ENABLES]--> (Replication)\n"
            "```\n"
        )
        _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 2

    def test_malformed_paren_format_skipped_with_warning(
        self, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from graph.graph_builder import _parse_triples

        content = "(A) --[BAD_FORMAT--> (B)"
        with caplog.at_level("WARNING"):
            _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 1
        assert len(edges) == 0
        assert any("malformed" in r.message.lower() for r in caplog.records)

    def test_both_formats_in_same_file(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = (
            '"Tidy First" → enables → "Refactoring"\n'
            "(Write-Ahead Log) --[ENABLES]--> (Crash Recovery)\n"
        )
        _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 2

    def test_blank_lines_and_comments_are_skipped_silently(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = "\n\n# This is a comment\n\n"
        nodes, edges, skipped = _parse_triples(content, "src")

        assert nodes == []
        assert edges == []
        assert skipped == 0

    def test_multiple_triples_in_file(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"A" → enables → "B"\n"B" → depends_on → "C"\n"C" → reinforces → "A"\n'
        _nodes, edges, skipped = _parse_triples(content, "src")

        assert skipped == 0
        assert len(edges) == 3

    def test_same_concept_in_multiple_triples_produces_one_node(self, temp_dir: Path) -> None:
        from graph.graph_builder import _parse_triples

        content = '"A" → enables → "B"\n"A" → reinforces → "C"\n'
        nodes, _edges, _skipped = _parse_triples(content, "src")

        node_ids = [n["id"] for n in nodes]
        assert node_ids.count("a") == 1


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


class TestBuild:
    def test_adds_nodes_and_edges_from_single_file(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "tidy-first"
        src_dir.mkdir()
        _write_relationships(src_dir, '"Tidy First" → enables → "Refactoring"\n')

        store = _make_store(temp_dir)
        build(src_dir, store)

        assert store.node_count == 2
        assert store.edge_count == 1

    def test_derives_source_slug_from_directory_name(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "rust-book"
        src_dir.mkdir()
        _write_relationships(src_dir, '"Ownership" → enables → "Memory Safety"\n')

        store = _make_store(temp_dir)
        build(src_dir, store)

        nodes = store.get_by_source("rust-book")
        assert len(nodes) == 2

    def test_remove_source_called_before_add(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "my-source"
        src_dir.mkdir()

        # First build: 2 nodes
        _write_relationships(src_dir, '"Old Concept A" → enables → "Old Concept B"\n')
        store = _make_store(temp_dir)
        build(src_dir, store)
        store.save()
        assert store.node_count == 2

        # Update file: different concepts
        _write_relationships(src_dir, '"New Concept X" → enables → "New Concept Y"\n')
        store2 = _make_store(temp_dir)
        store2.load()
        build(src_dir, store2)

        # Old concepts must be gone; only new ones remain
        ids = {n["id"] for n in store2.get_by_source("my-source")}
        assert "old-concept-a" not in ids
        assert "new-concept-x" in ids

    def test_idempotent_on_identical_rerun(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "src"
        src_dir.mkdir()
        _write_relationships(src_dir, '"A" → enables → "B"\n')

        store = _make_store(temp_dir)
        build(src_dir, store)
        build(src_dir, store)

        assert store.node_count == 2
        assert store.edge_count == 1

    def test_dry_run_does_not_write_graph_file(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "src"
        src_dir.mkdir()
        _write_relationships(src_dir, '"A" → enables → "B"\n')

        store = _make_store(temp_dir)
        stats = build(src_dir, store, dry_run=True)

        assert not store.path.exists()
        assert stats.dry_run is True

    def test_dry_run_reports_triples_parsed(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "src"
        src_dir.mkdir()
        _write_relationships(src_dir, '"A" → enables → "B"\n"C" → reinforces → "D"\n')

        store = _make_store(temp_dir)
        stats = build(src_dir, store, dry_run=True)

        assert stats.triples_parsed == 2
        assert stats.files_processed == 1

    def test_processes_multiple_relationships_files_recursively(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        root = temp_dir / "sources"
        root.mkdir()
        (root / "rust").mkdir()
        (root / "python").mkdir()
        _write_relationships(root / "rust", '"Ownership" → enables → "Safety"\n')
        _write_relationships(root / "python", '"AsyncIO" → enables → "Concurrency"\n')

        store = _make_store(temp_dir)
        stats = build(root, store)

        assert stats.files_processed == 2
        assert store.node_count == 4

    def test_single_relationships_file_as_input(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "src"
        src_dir.mkdir()
        rel_file = _write_relationships(src_dir, '"A" → enables → "B"\n')

        store = _make_store(temp_dir)
        build(rel_file, store)

        assert store.node_count == 2

    def test_save_called_after_build(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "src"
        src_dir.mkdir()
        _write_relationships(src_dir, '"A" → enables → "B"\n')

        store = _make_store(temp_dir)
        build(src_dir, store)

        assert store.path.exists()

    def test_malformed_lines_do_not_stop_processing(
        self, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "src"
        src_dir.mkdir()
        _write_relationships(
            src_dir,
            '"A" → → "B"\n"A" → enables → "B"\n',  # first line has arrow but no relation
        )

        store = _make_store(temp_dir)
        with caplog.at_level("WARNING"):
            stats = build(src_dir, store)

        assert store.node_count == 2
        assert stats.triples_skipped == 1

    def test_missing_input_path_returns_empty_stats(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        store = _make_store(temp_dir)
        stats = build(temp_dir / "nonexistent", store)

        assert stats.files_processed == 0
        assert store.node_count == 0

    def test_builds_from_paren_format_relationships_file(self, temp_dir: Path) -> None:
        from graph.graph_builder import build

        src_dir = temp_dir / "ddia"
        src_dir.mkdir()
        _write_relationships(
            src_dir,
            "```\n"
            "(Reliability) --[IS_ACHIEVED_BY]--> (Fault Tolerance)\n"
            "(Scalability) --[REQUIRES]--> (Load Parameters)\n"
            "```\n",
        )

        store = _make_store(temp_dir)
        stats = build(src_dir, store)

        assert store.node_count == 4
        assert store.edge_count == 2
        assert stats.triples_parsed == 2
        assert stats.triples_skipped == 0
