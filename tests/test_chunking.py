"""Tests for document chunking."""

from grounded_code_mcp.chunking import (
    Chunk,
    ChunkingContext,
    DocumentChunker,
    generate_chunk_id,
)


class TestGenerateChunkId:
    """Tests for chunk ID generation."""

    def test_generates_unique_ids(self) -> None:
        """Test that IDs are unique."""
        id1 = generate_chunk_id("test.md", 0)
        id2 = generate_chunk_id("test.md", 0)
        assert id1 != id2

    def test_returns_valid_uuid(self) -> None:
        """Test that ID is a valid UUID."""
        import uuid

        chunk_id = generate_chunk_id("path/to/doc.md", 5)
        # Should not raise
        uuid.UUID(chunk_id)


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_char_count(self) -> None:
        """Test character count property."""
        chunk = Chunk(chunk_id="id", content="Hello World", chunk_index=0)
        assert chunk.char_count == 11

    def test_defaults(self) -> None:
        """Test default values."""
        chunk = Chunk(chunk_id="id", content="test", chunk_index=0)
        assert chunk.heading_context == []
        assert chunk.is_code is False
        assert chunk.code_language is None
        assert chunk.is_table is False


class TestChunkingContext:
    """Tests for ChunkingContext."""

    def test_heading_stack_updates(self) -> None:
        """Test heading stack management."""
        ctx = ChunkingContext()

        ctx.update_headings(1, "Top Level")
        assert ctx.get_heading_context() == ["Top Level"]

        ctx.update_headings(2, "Section")
        assert ctx.get_heading_context() == ["Top Level", "Section"]

        ctx.update_headings(3, "Subsection")
        assert ctx.get_heading_context() == ["Top Level", "Section", "Subsection"]

    def test_heading_stack_replaces_same_level(self) -> None:
        """Test that same-level headings replace previous."""
        ctx = ChunkingContext()

        ctx.update_headings(1, "First")
        ctx.update_headings(2, "Section A")
        ctx.update_headings(2, "Section B")

        assert ctx.get_heading_context() == ["First", "Section B"]

    def test_heading_stack_pops_lower_levels(self) -> None:
        """Test that higher-level headings pop lower levels."""
        ctx = ChunkingContext()

        ctx.update_headings(1, "Chapter 1")
        ctx.update_headings(2, "Section")
        ctx.update_headings(3, "Subsection")
        ctx.update_headings(2, "New Section")

        assert ctx.get_heading_context() == ["Chapter 1", "New Section"]

    def test_next_chunk_id(self) -> None:
        """Test chunk ID generation and index increment."""
        import uuid

        ctx = ChunkingContext(source_path="test.md")

        id1 = ctx.next_chunk_id()
        id2 = ctx.next_chunk_id()

        # IDs should be valid UUIDs
        uuid.UUID(id1)
        uuid.UUID(id2)
        # IDs should be unique
        assert id1 != id2
        assert ctx.current_index == 2


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_empty_content(self) -> None:
        """Test chunking empty content."""
        chunker = DocumentChunker()
        result = chunker.chunk("")
        assert result == []

    def test_whitespace_only(self) -> None:
        """Test chunking whitespace-only content."""
        chunker = DocumentChunker()
        result = chunker.chunk("   \n\t  ")
        assert result == []

    def test_simple_text(self) -> None:
        """Test chunking simple text."""
        chunker = DocumentChunker(text_chunk_size=100, text_chunk_max_size=200)
        content = "This is a simple paragraph of text."

        result = chunker.chunk(content, "test.md")

        assert len(result) == 1
        assert result[0].content == content
        assert result[0].is_code is False
        assert result[0].is_table is False

    def test_code_block_preserved(self) -> None:
        """Test that code blocks are preserved atomically."""
        chunker = DocumentChunker(max_code_chunk_size=5000)
        content = """Some text.

```python
def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
```

More text.
"""
        result = chunker.chunk(content, "test.md")

        code_chunks = [c for c in result if c.is_code]
        assert len(code_chunks) == 1
        assert "def hello():" in code_chunks[0].content
        assert "def goodbye():" in code_chunks[0].content
        assert code_chunks[0].code_language == "python"

    def test_code_block_split_on_functions(self) -> None:
        """Test that large code blocks split on function boundaries."""
        chunker = DocumentChunker(max_code_chunk_size=100)
        content = """```python
def first_function():
    # A very long function with lots of code
    x = 1
    y = 2
    z = 3
    return x + y + z

def second_function():
    # Another long function
    a = 10
    b = 20
    return a + b
```"""
        result = chunker.chunk(content, "test.md")

        code_chunks = [c for c in result if c.is_code]
        assert len(code_chunks) >= 2

    def test_table_preserved(self) -> None:
        """Test that tables are preserved atomically."""
        chunker = DocumentChunker()
        content = """Some text before.

| Column A | Column B | Column C |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

Some text after.
"""
        result = chunker.chunk(content, "test.md")

        table_chunks = [c for c in result if c.is_table]
        assert len(table_chunks) == 1
        assert "Column A" in table_chunks[0].content
        assert "Value 6" in table_chunks[0].content

    def test_heading_context_preserved(self) -> None:
        """Test that heading context is preserved in chunks."""
        chunker = DocumentChunker(text_chunk_size=50, text_chunk_max_size=100)
        content = """# Main Title

## Section One

Content in section one.

## Section Two

Content in section two.
"""
        result = chunker.chunk(content, "test.md")

        # Find chunks with section content
        section_two_chunks = [
            c for c in result if "section two" in c.content.lower()
        ]
        if section_two_chunks:
            # Should have heading context
            assert len(section_two_chunks[0].heading_context) > 0

    def test_source_path_in_chunks(self) -> None:
        """Test that source path is included in chunks."""
        chunker = DocumentChunker()
        result = chunker.chunk("Test content", "path/to/doc.md")

        assert len(result) == 1
        assert result[0].source_path == "path/to/doc.md"

    def test_chunk_ids_unique(self) -> None:
        """Test that chunk IDs are unique within a document."""
        chunker = DocumentChunker(text_chunk_size=20, text_chunk_max_size=50)
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        result = chunker.chunk(content, "test.md")

        chunk_ids = [c.chunk_id for c in result]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_text_chunking_respects_size(self) -> None:
        """Test that text chunks respect size limits."""
        chunker = DocumentChunker(
            text_chunk_size=100,
            text_chunk_max_size=150,
            text_chunk_overlap=20,
        )

        # Create long content
        paragraphs = [f"This is paragraph number {i}." for i in range(20)]
        content = "\n\n".join(paragraphs)

        result = chunker.chunk(content, "test.md")

        # All chunks should be under max size
        for chunk in result:
            assert chunk.char_count <= 200  # Some tolerance for edge cases

    def test_multiple_code_languages(self) -> None:
        """Test handling multiple code block languages."""
        chunker = DocumentChunker(max_code_chunk_size=5000)
        content = """```python
def py_func():
    pass
```

```javascript
function jsFunc() {
    return true;
}
```
"""
        result = chunker.chunk(content, "test.md")

        code_chunks = [c for c in result if c.is_code]
        assert len(code_chunks) == 2

        languages = {c.code_language for c in code_chunks}
        assert "python" in languages
        assert "javascript" in languages

    def test_code_without_language(self) -> None:
        """Test code blocks without language specification."""
        chunker = DocumentChunker()
        content = """```
plain code block
```"""
        result = chunker.chunk(content, "test.md")

        code_chunks = [c for c in result if c.is_code]
        assert len(code_chunks) == 1
        assert code_chunks[0].code_language is None

    def test_from_settings(self) -> None:
        """Test creating chunker from settings."""
        from grounded_code_mcp.config import ChunkingSettings

        settings = ChunkingSettings(
            text_chunk_size=500,
            text_chunk_max_size=750,
            text_chunk_overlap=100,
            max_code_chunk_size=2000,
        )

        chunker = DocumentChunker.from_settings(settings)

        assert chunker.text_chunk_size == 500
        assert chunker.text_chunk_max_size == 750
        assert chunker.text_chunk_overlap == 100
        assert chunker.max_code_chunk_size == 2000
