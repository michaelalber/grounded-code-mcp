"""Tests for document parsing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grounded_code_mcp.parser import (
    DocumentParseError,
    DocumentParser,
    ParsedDocument,
    UnsupportedFormatError,
    get_file_type,
    is_supported_format,
    scan_directory,
)


class TestIsSupportedFormat:
    """Tests for is_supported_format function."""

    @pytest.mark.parametrize(
        "suffix",
        [
            ".pdf", ".docx", ".doc", ".pptx", ".html", ".md", ".markdown", ".epub",
            ".rst", ".txt", ".mdx",
        ],
    )
    def test_supported_formats(self, suffix: str) -> None:
        """Test that supported formats return True."""
        assert is_supported_format(Path(f"test{suffix}")) is True

    @pytest.mark.parametrize("suffix", [".jpg", ".png", ".py", ".json"])
    def test_unsupported_formats(self, suffix: str) -> None:
        """Test that unsupported formats return False."""
        assert is_supported_format(Path(f"test{suffix}")) is False

    def test_case_insensitive(self) -> None:
        """Test that format check is case insensitive."""
        assert is_supported_format(Path("test.PDF")) is True
        assert is_supported_format(Path("test.DOCX")) is True


class TestGetFileType:
    """Tests for get_file_type function."""

    def test_standard_extensions(self) -> None:
        """Test standard extension mapping."""
        assert get_file_type(Path("test.pdf")) == "pdf"
        assert get_file_type(Path("test.docx")) == "docx"
        assert get_file_type(Path("test.html")) == "html"

    def test_normalized_extensions(self) -> None:
        """Test extension normalization."""
        assert get_file_type(Path("test.markdown")) == "md"
        assert get_file_type(Path("test.adoc")) == "asciidoc"
        assert get_file_type(Path("test.htm")) == "html"
        assert get_file_type(Path("test.doc")) == "docx"
        assert get_file_type(Path("test.mdx")) == "md"

    def test_rst_and_txt_file_types(self) -> None:
        """Test that .rst and .txt return their own type strings."""
        assert get_file_type(Path("test.rst")) == "rst"
        assert get_file_type(Path("test.txt")) == "txt"


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    def test_is_empty_with_content(self) -> None:
        """Test is_empty returns False when content exists."""
        doc = ParsedDocument(path=Path("test.md"), content="# Hello")
        assert doc.is_empty is False

    def test_is_empty_without_content(self) -> None:
        """Test is_empty returns True for empty content."""
        doc = ParsedDocument(path=Path("test.md"), content="")
        assert doc.is_empty is True

    def test_is_empty_whitespace_only(self) -> None:
        """Test is_empty returns True for whitespace-only content."""
        doc = ParsedDocument(path=Path("test.md"), content="   \n\t  ")
        assert doc.is_empty is True


class TestDocumentParser:
    """Tests for DocumentParser class."""

    def test_parse_markdown(self, temp_dir: Path) -> None:
        """Test parsing a markdown file."""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Test Document\n\nSome content here.")

        parser = DocumentParser()
        result = parser.parse(md_file)

        assert result.content == "# Test Document\n\nSome content here."
        assert result.title == "Test Document"
        assert result.file_type == "md"

    def test_parse_markdown_no_title(self, temp_dir: Path) -> None:
        """Test parsing markdown without a heading."""
        md_file = temp_dir / "test.md"
        md_file.write_text("Just some content without a title.")

        parser = DocumentParser()
        result = parser.parse(md_file)

        assert result.title is None

    def test_parse_nonexistent_file(self, temp_dir: Path) -> None:
        """Test parsing a file that doesn't exist."""
        parser = DocumentParser()

        with pytest.raises(FileNotFoundError):
            parser.parse(temp_dir / "nonexistent.md")

    def test_parse_unsupported_format(self, temp_dir: Path) -> None:
        """Test parsing an unsupported format."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("a,b,c")

        parser = DocumentParser()

        with pytest.raises(UnsupportedFormatError) as exc_info:
            parser.parse(csv_file)

        assert exc_info.value.path == csv_file

    def test_parse_with_docling_mock(self, temp_dir: Path) -> None:
        """Test parsing with mocked Docling."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")

        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Converted Content"
        mock_doc.title = "Document Title"
        mock_doc.pages = [1, 2, 3]

        mock_result = MagicMock()
        mock_result.document = mock_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        parser = DocumentParser()

        with patch.object(parser, "_get_converter", return_value=mock_converter):
            result = parser.parse(pdf_file)

        assert result.content == "# Converted Content"
        assert result.title == "Document Title"
        assert result.page_count == 3
        assert result.file_type == "pdf"

    def test_parse_with_docling_error(self, temp_dir: Path) -> None:
        """Test handling Docling errors."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"invalid pdf")

        mock_converter = MagicMock()
        mock_converter.convert.side_effect = RuntimeError("Parse error")

        parser = DocumentParser()

        with (
            patch.object(parser, "_get_converter", return_value=mock_converter),
            pytest.raises(DocumentParseError) as exc_info,
        ):
            parser.parse(pdf_file)

        assert exc_info.value.path == pdf_file

    def test_parse_many(self, temp_dir: Path) -> None:
        """Test parsing multiple documents."""
        (temp_dir / "doc1.md").write_text("# Doc 1")
        (temp_dir / "doc2.md").write_text("# Doc 2")

        parser = DocumentParser()
        results = parser.parse_many([temp_dir / "doc1.md", temp_dir / "doc2.md"])

        assert len(results) == 2
        assert results[0].title == "Doc 1"
        assert results[1].title == "Doc 2"

    def test_parse_many_skip_errors(self, temp_dir: Path) -> None:
        """Test parse_many skips errors when configured."""
        (temp_dir / "good.md").write_text("# Good Doc")
        (temp_dir / "bad.csv").write_text("a,b,c")

        parser = DocumentParser()
        results = parser.parse_many(
            [temp_dir / "good.md", temp_dir / "bad.csv"],
            skip_errors=True,
        )

        assert len(results) == 1
        assert results[0].title == "Good Doc"

    def test_parse_many_raises_on_error(self, temp_dir: Path) -> None:
        """Test parse_many raises when skip_errors is False."""
        (temp_dir / "bad.csv").write_text("a,b,c")

        parser = DocumentParser()

        with pytest.raises(UnsupportedFormatError):
            parser.parse_many([temp_dir / "bad.csv"], skip_errors=False)

    def test_parse_many_skips_empty(self, temp_dir: Path) -> None:
        """Test parse_many skips empty documents."""
        (temp_dir / "empty.md").write_text("")
        (temp_dir / "good.md").write_text("# Content")

        parser = DocumentParser()
        results = parser.parse_many([temp_dir / "empty.md", temp_dir / "good.md"])

        assert len(results) == 1

    def test_parse_asciidoc_directly_without_docling(self, temp_dir: Path) -> None:
        """Test that .asciidoc files are read directly, bypassing Docling."""
        adoc_file = temp_dir / "test.asciidoc"
        adoc_file.write_text("= My Title\n\nSome content here.")

        parser = DocumentParser()
        # Patching _get_converter to raise proves Docling is never called
        with patch.object(parser, "_get_converter", side_effect=RuntimeError("Docling should not be called")):
            result = parser.parse(adoc_file)

        assert "Some content here." in result.content
        assert result.file_type == "asciidoc"

    def test_parse_adoc_extension_directly_without_docling(self, temp_dir: Path) -> None:
        """Test that .adoc files are read directly, bypassing Docling."""
        adoc_file = temp_dir / "test.adoc"
        adoc_file.write_text("= Chapter Title\n\nBody text.")

        parser = DocumentParser()
        with patch.object(parser, "_get_converter", side_effect=RuntimeError("Docling should not be called")):
            result = parser.parse(adoc_file)

        assert "Body text." in result.content
        assert result.file_type == "asciidoc"

    def test_parse_rst_directly_without_docling(self, temp_dir: Path) -> None:
        """Test that .rst files are read directly, bypassing Docling."""
        rst_file = temp_dir / "test.rst"
        rst_file.write_text("My Title\n========\n\nSome rst content.")

        parser = DocumentParser()
        with patch.object(parser, "_get_converter", side_effect=RuntimeError("Docling should not be called")):
            result = parser.parse(rst_file)

        assert "Some rst content." in result.content
        assert result.file_type == "rst"
        assert result.title == "My Title"

    def test_parse_rst_without_title(self, temp_dir: Path) -> None:
        """Test that .rst files without a title heading have title=None."""
        rst_file = temp_dir / "noheading.rst"
        rst_file.write_text("Just some plain rst content without a heading.")

        parser = DocumentParser()
        result = parser.parse(rst_file)

        assert result.title is None
        assert "plain rst content" in result.content

    def test_parse_txt_directly_without_docling(self, temp_dir: Path) -> None:
        """Test that .txt files are read directly, bypassing Docling."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Plain text content.\nSecond line.")

        parser = DocumentParser()
        with patch.object(parser, "_get_converter", side_effect=RuntimeError("Docling should not be called")):
            result = parser.parse(txt_file)

        assert "Plain text content." in result.content
        assert result.file_type == "txt"
        assert result.title is None

    def test_parse_mdx_directly_without_docling(self, temp_dir: Path) -> None:
        """Test that .mdx files are read directly as markdown, bypassing Docling."""
        mdx_file = temp_dir / "test.mdx"
        mdx_file.write_text("# MDX Title\n\nContent with <Component /> tags.")

        parser = DocumentParser()
        with patch.object(parser, "_get_converter", side_effect=RuntimeError("Docling should not be called")):
            result = parser.parse(mdx_file)

        assert "Content with <Component />" in result.content
        assert result.file_type == "md"
        assert result.title == "MDX Title"


class TestScanDirectory:
    """Tests for scan_directory function."""

    def test_scan_empty_directory(self, temp_dir: Path) -> None:
        """Test scanning an empty directory."""
        result = scan_directory(temp_dir)
        assert result == []

    def test_scan_with_supported_files(self, temp_dir: Path) -> None:
        """Test scanning directory with supported files."""
        (temp_dir / "doc.md").write_text("# Test")
        (temp_dir / "doc.pdf").write_bytes(b"pdf")
        (temp_dir / "ignored.py").write_text("code")

        result = scan_directory(temp_dir)

        assert len(result) == 2
        assert any(p.suffix == ".md" for p in result)
        assert any(p.suffix == ".pdf" for p in result)

    def test_scan_includes_rst_txt_mdx(self, temp_dir: Path) -> None:
        """Test that scan_directory includes .rst, .txt, and .mdx files."""
        (temp_dir / "doc.rst").write_text("RST content")
        (temp_dir / "doc.txt").write_text("Plain text")
        (temp_dir / "doc.mdx").write_text("# MDX content")
        (temp_dir / "ignored.py").write_text("code")

        result = scan_directory(temp_dir)

        suffixes = {p.suffix for p in result}
        assert ".rst" in suffixes
        assert ".txt" in suffixes
        assert ".mdx" in suffixes
        assert ".py" not in suffixes

    def test_scan_recursive(self, temp_dir: Path) -> None:
        """Test recursive directory scanning."""
        (temp_dir / "subdir").mkdir()
        (temp_dir / "root.md").write_text("# Root")
        (temp_dir / "subdir" / "nested.md").write_text("# Nested")

        result = scan_directory(temp_dir, recursive=True)
        assert len(result) == 2

    def test_scan_non_recursive(self, temp_dir: Path) -> None:
        """Test non-recursive directory scanning."""
        (temp_dir / "subdir").mkdir()
        (temp_dir / "root.md").write_text("# Root")
        (temp_dir / "subdir" / "nested.md").write_text("# Nested")

        result = scan_directory(temp_dir, recursive=False)
        assert len(result) == 1
        assert result[0].name == "root.md"

    def test_scan_nonexistent_directory(self, temp_dir: Path) -> None:
        """Test scanning a non-existent directory."""
        result = scan_directory(temp_dir / "nonexistent")
        assert result == []
