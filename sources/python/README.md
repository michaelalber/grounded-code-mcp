# `python` — Python Language & Libraries

Python language reference, modern idioms, testing patterns, web frameworks,
and data tooling. Focus on Python 3.10+ and the libraries most relevant to
AI-assisted software development.

## What belongs here

- Python language and standard library reference
- Modern idiomatic Python (Fluent Python, type hints, async)
- Testing with pytest — patterns, fixtures, and best practices
- Web frameworks: FastAPI, Flask
- Data validation: Pydantic v2
- Architecture patterns in Python (DDD, ports & adapters, clean architecture)
- Data analysis and ML with Python

## Sources in this collection

### Books (local)
| File | Content |
|---|---|
| `fluent-python.pdf` | Fluent Python — Ramalho (O'Reilly); idiomatic Python, essential reference |
| `Deep_Learning_with_Python.pdf` | Deep Learning with Python — Chollet (Manning) |
| `flask-web-development.pdf` | Flask Web Development — Grinberg (O'Reilly) |
| `python-for-data-analysis.pdf` | Python for Data Analysis — McKinney (O'Reilly); pandas reference |
| `core-python-applications-programming.pdf` | Core Python Applications Programming 3rd ed. — Chun (Prentice Hall) |
| `python-cookbook.pdf` | Python Cookbook 2nd ed. — Beazley & Jones (O'Reilly) |
| `mastering-python-design-patterns.epub` | Mastering Python Design Patterns |

### Free/Open Resources (downloaded)
| Directory / File | Content |
|---|---|
| `python_testing_guide.md` | Internal pytest patterns and fixture guide |
| `architecture-patterns-python/` | Architecture Patterns with Python — Percival & Gregory (cosmicpython.com, CC BY-NC-ND) |
| `fastapi-docs/` | FastAPI official docs — markdown source (tiangolo/fastapi, MIT) |
| `pydantic-v2-docs/` | Pydantic v2 official docs — markdown source (pydantic/pydantic, MIT) |
| `pytest-docs/` | pytest official docs — rst source (pytest-dev/pytest, MIT) |
| `python-3-docs/` | Python 3.13 official HTML docs — docs.python.org (PSF License) |

## Refreshing open-source content

```bash
PY_DST=sources/python

# Architecture Patterns with Python (cosmicpython.com)
curl -sL https://github.com/cosmicpython/book/archive/refs/heads/master.zip -o /tmp/cosmic.zip && \
  unzip -q /tmp/cosmic.zip -d /tmp/cosmic && \
  mkdir -p $PY_DST/architecture-patterns-python && \
  rsync -a --include="*.asciidoc" --include="*.md" --exclude="*" \
    /tmp/cosmic/book-master/ $PY_DST/architecture-patterns-python/

# FastAPI docs
curl -sL https://github.com/fastapi/fastapi/archive/refs/heads/master.zip -o /tmp/fastapi.zip && \
  unzip -q /tmp/fastapi.zip -d /tmp/fastapi && \
  mkdir -p $PY_DST/fastapi-docs && \
  rsync -a --include="*.md" --exclude="*" /tmp/fastapi/fastapi-master/docs/en/docs/ $PY_DST/fastapi-docs/

# Pydantic v2 docs
curl -sL https://github.com/pydantic/pydantic/archive/refs/heads/main.zip -o /tmp/pydantic.zip && \
  unzip -q /tmp/pydantic.zip -d /tmp/pydantic && \
  mkdir -p $PY_DST/pydantic-v2-docs && \
  rsync -a --include="*.md" --exclude="*" /tmp/pydantic/pydantic-main/docs/ $PY_DST/pydantic-v2-docs/

# pytest docs
curl -sL https://github.com/pytest-dev/pytest/archive/refs/heads/main.zip -o /tmp/pytest.zip && \
  unzip -q /tmp/pytest.zip -d /tmp/pytest && \
  mkdir -p $PY_DST/pytest-docs && \
  rsync -a --include="*.rst" --include="*.md" --exclude="*" \
    /tmp/pytest/pytest-main/doc/en/ $PY_DST/pytest-docs/

# Python 3 official docs (update version number as needed)
curl -sL https://www.python.org/ftp/python/doc/3.13.3/python-3.13.3-docs-html.tar.bz2 \
     -o /tmp/pydocs.tar.bz2 && \
  tar -xjf /tmp/pydocs.tar.bz2 -C /tmp/ && \
  mkdir -p $PY_DST/python-3-docs && \
  rsync -a /tmp/python-3.13.3-docs-html/ $PY_DST/python-3-docs/
```
