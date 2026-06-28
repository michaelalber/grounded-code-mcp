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

Pass the bare suffix (the server prepends `grounded_` automatically). Use `grounded_list_sources()` for the authoritative runtime list.

| Pass as `collection=` | Contents |
|---|---|
| `internal` | XP, TDD, CI/CD, DDD, OWASP, NIST AI; technical writing |
| `patterns` | GoF, CQRS, DDD, Clean Architecture, DI |
| `architecture` | DDIA, SRE, 12-Factor, AOSA, C4, arc42 |
| `systems_thinking` | Meadows leverage points, feedback loops, chaos engineering |
| `ui_ux` | Laws of UX, Nielsen, WCAG 2.2, ARIA, USWDS, GOV.UK |
| `dotnet` | EF Core in Action, DI in .NET, Telerik UI components |
| `python` | Python 3, FastAPI, Pydantic v2, pytest, cosmicpython |
| `databases` | PostgreSQL, SQL, relational theory |
| `edge_ai` | RAG, embeddings, LLM application design, AI agents |
| `automation` | PLC, OPC UA, MODBUS, ICS security, Raspberry Pi |
| `php` | PHP manual, Laravel 5.5 / 6.x / 12.x |
| `javascript` | TypeScript, Vue 3, ECMAScript 2024 |
| `gov` | NIST 800-53/171/218, DOE, Zero Trust, AI RMF, CUI |
| `robotics` | ROS 2, MuJoCo, Isaac Lab, LeRobot, VLA models |
| `rust` | Rust ownership, async/Tokio, Cargo, Axum |
| `api_design` | Zalando guidelines, Google AIP, Microsoft REST API guidelines |

## How it works

Each tool runs `grounded-code-mcp <subcommand> --json` as a subprocess and
returns the parsed JSON to pi's context. No network calls — everything is local.
