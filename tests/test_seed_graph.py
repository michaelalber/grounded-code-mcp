"""Tests for graph/seed_graph.py bootstrap script.

seed_graph generates starter RELATIONSHIPS.md files for source directories
that lack them.  No real Qdrant or Ollama is needed — sources are discovered
by scanning the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(sources_dir: Path, name: str, *, with_relationships: bool = False) -> Path:
    """Create a source directory with at least one document file."""
    d = sources_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "doc.md").write_text(f"# {name}\n\nContent for {name}.")
    if with_relationships:
        (d / "RELATIONSHIPS.md").write_text('"Existing" → enables → "Concept"\n')
    return d


# ---------------------------------------------------------------------------
# find_sources_missing_relationships
# ---------------------------------------------------------------------------


class TestFindSourcesMissingRelationships:
    def test_returns_dirs_without_relationships(self, temp_dir: Path) -> None:
        """Directories that contain files but no RELATIONSHIPS.md are returned."""
        from graph.seed_graph import find_sources_missing_relationships

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=False)
        _make_source(sources_dir, "python", with_relationships=False)

        missing = find_sources_missing_relationships(sources_dir)

        names = {p.name for p in missing}
        assert "rust" in names
        assert "python" in names

    def test_excludes_dirs_with_relationships(self, temp_dir: Path) -> None:
        """Directories that already have RELATIONSHIPS.md are excluded."""
        from graph.seed_graph import find_sources_missing_relationships

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=True)
        _make_source(sources_dir, "python", with_relationships=False)

        missing = find_sources_missing_relationships(sources_dir)

        names = {p.name for p in missing}
        assert "rust" not in names
        assert "python" in names

    def test_empty_sources_dir_returns_empty_list(self, temp_dir: Path) -> None:
        """Empty sources directory returns no results."""
        from graph.seed_graph import find_sources_missing_relationships

        sources_dir = temp_dir / "sources"
        sources_dir.mkdir()

        missing = find_sources_missing_relationships(sources_dir)

        assert missing == []

    def test_non_existent_sources_dir_returns_empty_list(self, temp_dir: Path) -> None:
        """Nonexistent directory does not raise — returns empty list."""
        from graph.seed_graph import find_sources_missing_relationships

        missing = find_sources_missing_relationships(temp_dir / "nonexistent")

        assert missing == []

    def test_nested_source_directories_discovered(self, temp_dir: Path) -> None:
        """Subdirectories within collections are also checked."""
        from graph.seed_graph import find_sources_missing_relationships

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir / "automation", "python-opcua-docs", with_relationships=False)

        missing = find_sources_missing_relationships(sources_dir)

        names = {p.name for p in missing}
        assert "python-opcua-docs" in names

    def test_directories_with_only_relationships_file_excluded(self, temp_dir: Path) -> None:
        """A directory whose only file IS RELATIONSHIPS.md should not appear as a source."""
        from graph.seed_graph import find_sources_missing_relationships

        sources_dir = temp_dir / "sources"
        orphan = sources_dir / "orphan"
        orphan.mkdir(parents=True)
        (orphan / "RELATIONSHIPS.md").write_text('"A" → enables → "B"\n')

        missing = find_sources_missing_relationships(sources_dir)

        names = {p.name for p in missing}
        assert "orphan" not in names


# ---------------------------------------------------------------------------
# generate_seed_content
# ---------------------------------------------------------------------------


class TestGenerateSeedContent:
    def test_output_starts_with_seed_header(self, temp_dir: Path) -> None:
        """Every generated file starts with the SEED header comment."""
        from graph.seed_graph import SEED_HEADER, generate_seed_content

        content = generate_seed_content("rust")

        assert content.startswith(SEED_HEADER)

    def test_content_contains_at_least_one_triple(self, temp_dir: Path) -> None:
        """Generated content must include at least one valid relationship triple."""
        from graph.seed_graph import generate_seed_content

        content = generate_seed_content("internal")

        has_arrow = "→" in content or "->" in content
        assert has_arrow, "Expected at least one relationship triple in generated content"

    def test_well_known_slug_produces_domain_specific_content(self, temp_dir: Path) -> None:
        """Well-known source slugs get domain-specific relationships, not a generic stub."""
        from graph.seed_graph import generate_seed_content

        known_slugs = ["internal", "patterns", "architecture", "dotnet", "python"]
        generic_slug = "unknown-random-slug-xyz"

        generic_content = generate_seed_content(generic_slug)
        for slug in known_slugs:
            content = generate_seed_content(slug)
            assert content != generic_content, (
                f"Expected '{slug}' to produce different content from the generic template"
            )

    def test_unknown_slug_returns_generic_template(self, temp_dir: Path) -> None:
        """Unknown source slugs get a generic starter template."""
        from graph.seed_graph import generate_seed_content

        content = generate_seed_content("totally-unknown-source-99")

        assert "SEED" in content


# ---------------------------------------------------------------------------
# seed() — dry-run mode
# ---------------------------------------------------------------------------


class TestSeedDryRun:
    def test_dry_run_does_not_write_files(self, temp_dir: Path) -> None:
        """--dry-run must never write RELATIONSHIPS.md to disk."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=False)

        seed(sources_dir, dry_run=True)

        assert not (sources_dir / "rust" / "RELATIONSHIPS.md").exists()

    def test_dry_run_returns_stats_with_sources_found(self, temp_dir: Path) -> None:
        """Dry-run returns SeedStats counting how many sources need seeding."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=False)
        _make_source(sources_dir, "python", with_relationships=False)

        stats = seed(sources_dir, dry_run=True)

        assert stats.sources_found == 2
        assert stats.files_written == 0

    def test_dry_run_skips_sources_with_existing_relationships(self, temp_dir: Path) -> None:
        """Dry-run excludes sources that already have RELATIONSHIPS.md."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=True)
        _make_source(sources_dir, "python", with_relationships=False)

        stats = seed(sources_dir, dry_run=True)

        assert stats.sources_found == 1


# ---------------------------------------------------------------------------
# seed() — write mode
# ---------------------------------------------------------------------------


class TestSeedWrite:
    def test_writes_relationships_md_for_missing_sources(self, temp_dir: Path) -> None:
        """seed() writes RELATIONSHIPS.md for each source that lacks one."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=False)

        seed(sources_dir)

        assert (sources_dir / "rust" / "RELATIONSHIPS.md").exists()

    def test_written_file_starts_with_seed_header(self, temp_dir: Path) -> None:
        """Written file must begin with the SEED header comment."""
        from graph.seed_graph import SEED_HEADER, seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "python", with_relationships=False)

        seed(sources_dir)

        content = (sources_dir / "python" / "RELATIONSHIPS.md").read_text()
        assert content.startswith(SEED_HEADER)

    def test_does_not_overwrite_existing_relationships(self, temp_dir: Path) -> None:
        """seed() skips sources that already have RELATIONSHIPS.md — never overwrites."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=True)
        original = (sources_dir / "rust" / "RELATIONSHIPS.md").read_text()

        seed(sources_dir)

        assert (sources_dir / "rust" / "RELATIONSHIPS.md").read_text() == original

    def test_returns_stats_with_files_written(self, temp_dir: Path) -> None:
        """seed() returns SeedStats with files_written count."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=False)
        _make_source(sources_dir, "python", with_relationships=False)
        _make_source(sources_dir, "dotnet", with_relationships=True)  # should be skipped

        stats = seed(sources_dir)

        assert stats.files_written == 2
        assert stats.sources_found == 2

    def test_seed_creates_parseable_triples(self, temp_dir: Path) -> None:
        """Written RELATIONSHIPS.md can be parsed by graph_builder without errors."""
        from graph.graph_builder import _parse_triples
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "internal", with_relationships=False)

        seed(sources_dir)

        content = (sources_dir / "internal" / "RELATIONSHIPS.md").read_text()
        _nodes, edges, skipped = _parse_triples(content, "internal")

        assert len(edges) > 0
        assert skipped == 0, f"Expected 0 malformed triples, got {skipped}"


# ---------------------------------------------------------------------------
# seed() — specific source flag
# ---------------------------------------------------------------------------


class TestSeedSpecificSource:
    def test_targets_only_specified_source(self, temp_dir: Path) -> None:
        """--source flag seeds only the named source directory."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=False)
        _make_source(sources_dir, "python", with_relationships=False)

        seed(sources_dir, source="rust")

        assert (sources_dir / "rust" / "RELATIONSHIPS.md").exists()
        assert not (sources_dir / "python" / "RELATIONSHIPS.md").exists()

    def test_specific_source_not_found_raises(self, temp_dir: Path) -> None:
        """Specifying a source slug that does not exist in sources_dir raises ValueError."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        sources_dir.mkdir()

        with pytest.raises(ValueError, match="nonexistent-source"):
            seed(sources_dir, source="nonexistent-source")

    def test_specific_source_already_has_relationships_is_skipped(
        self, temp_dir: Path
    ) -> None:
        """If the specified source already has RELATIONSHIPS.md, seed() skips it."""
        from graph.seed_graph import seed

        sources_dir = temp_dir / "sources"
        _make_source(sources_dir, "rust", with_relationships=True)
        original = (sources_dir / "rust" / "RELATIONSHIPS.md").read_text()

        stats = seed(sources_dir, source="rust")

        assert stats.files_written == 0
        assert (sources_dir / "rust" / "RELATIONSHIPS.md").read_text() == original


# ---------------------------------------------------------------------------
# TestSeedContentFormat
# ---------------------------------------------------------------------------


class TestSeedContentFormat:
    def test_seed_content_uses_parenthetical_format(self, temp_dir: Path) -> None:
        from graph.seed_graph import generate_seed_content

        content = generate_seed_content("python")
        assert "--[" in content
        assert "-->" in content
        assert " → " not in content, "Quoted format arrow found — should use parenthetical only"

    def test_seed_content_contains_domain_tagged_section_header(self, temp_dir: Path) -> None:
        import re

        from graph.seed_graph import generate_seed_content

        content = generate_seed_content("python")
        assert re.search(r"##[^<]+<!--\s*domain:", content), (
            "No domain-tagged section header found in seed content"
        )

    def test_seed_content_relations_are_valid(self, temp_dir: Path) -> None:
        from dataclasses import dataclass, field

        from graph.graph_builder import BuildStats, _parse_triples
        from graph.seed_graph import _SEED_CONTENT, generate_seed_content

        for slug in _SEED_CONTENT:
            content = generate_seed_content(slug)
            stats = BuildStats()
            _nodes, _edges, _skipped = _parse_triples(content, slug, stats=stats)
            assert stats.triples_invalid_rel == 0, (
                f"Seed content for {slug!r} has {stats.triples_invalid_rel} invalid relation verb(s)"
            )

    def test_generic_template_uses_parenthetical_format(self, temp_dir: Path) -> None:
        from graph.seed_graph import generate_seed_content

        content = generate_seed_content("unknown-slug-xyz")
        assert "--[" in content
        assert "-->" in content
        assert " → " not in content, "Quoted format arrow found in generic template"
