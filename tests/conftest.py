"""Shared pytest fixtures for grounded-code-mcp tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_markdown_content() -> str:
    """Sample markdown content for testing."""
    return """# Sample Document

This is a sample document for testing.

## Section 1

Some content in section 1.

```python
def hello():
    print("Hello, world!")
```

## Section 2

More content here.
"""


@pytest.fixture
def sample_config_toml() -> str:
    """Sample configuration TOML for testing."""
    return """
[knowledge_base]
sources_dir = "sources"
data_dir = ".grounded-code-mcp"
manifest_file = "manifest.json"

[ollama]
model = "mxbai-embed-large"
host = "http://localhost:11434"

[chunking]
text_chunk_size = 1000
text_chunk_overlap = 200
max_code_chunk_size = 3000

[vectorstore]
provider = "qdrant"
collection_prefix = "grounded_"
"""
