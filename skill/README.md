# grounded-code-mcp-skill

Pi.dev extension that exposes the `grounded-code-mcp` local knowledge base as five
searchable tools. Eliminates hallucination by grounding pi's responses in vetted,
local documentation — no cloud calls, no internet required.

## Prerequisites

- `grounded-code-mcp` installed via pipx and on PATH
- Qdrant and Ollama running (same setup as the MCP server)
- At least one collection ingested

## Installation

### Option A — local path (simplest)

Add the extension directory to `~/.pi/settings.json`:

```json
{
  "extensions": [
    "/path/to/grounded-code-mcp/skill/extensions"
  ]
}
```

### Option B — test before installing

```bash
pi -e /path/to/grounded-code-mcp/skill/extensions/index.ts
```

### Option C — git package

```bash
# from inside pi:
/install git:codeberg.org/michaelkalber/grounded-code-mcp?path=skill
```

## Tools

| Tool | Description |
|---|---|
| `grounded_search` | Vector search across all (or one) collection. Returns prose chunks with score and source path. |
| `grounded_search_code` | Code-block-only search with optional language filter. |
| `grounded_list_sources` | Lists every ingested document; use to discover what's available. |
| `grounded_source_info` | Metadata for a specific source: chunk count, SHA-256, ingestion date. |
| `grounded_query_graph` | Graph traversal — finds concept relationships and linked sources. |

## Example usage in pi

```
Search for FastAPI dependency injection patterns
→ grounded_search(query="dependency injection", collection="python")

Find Python async context manager examples
→ grounded_search_code(query="async context manager", language="python")

What documentation is indexed?
→ grounded_list_sources()

How does CQRS relate to clean architecture?
→ grounded_query_graph(concept="CQRS", depth=2)
```

## Collection reference

Pass the bare suffix (the server prepends `grounded_` automatically):

| Pass as `collection=` | Contents |
|---|---|
| `internal` | XP, TDD, CI/CD, OWASP, Diátaxis |
| `patterns` | GoF, CQRS, DDD, Clean Architecture |
| `python` | Python 3, FastAPI, Pydantic v2, pytest |
| `dotnet` | EF Core, DI in .NET, Telerik |
| `architecture` | DDIA, SRE, 12-Factor, C4 |
| `edge_ai` | RAG, embeddings, LangSmith, AI agents |
| `databases` | PostgreSQL, SQL, relational theory |
| `javascript` | TypeScript, Vue 3, ECMAScript 2024 |
| `gov` | NIST 800-53/171, Zero Trust, AI RMF |
| `robotics` | ROS 2, MuJoCo, LeRobot |

## How it works

Each tool runs `grounded-code-mcp <subcommand> --json` as a subprocess and
returns the parsed JSON to pi's context. No network calls — everything is local.
