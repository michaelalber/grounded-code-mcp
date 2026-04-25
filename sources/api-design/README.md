# API Design Sources

REST and HTTP API design standards for the `grounded_api_design` collection.

## Collection

Pass `collection="api_design"` in MCP tool calls to search this collection.

## What Lives Here

| Path | Source | Coverage |
|------|--------|----------|
| `zalando/` | opensource.zalando.com/restful-api-guidelines | ~200 numbered REST design rules |
| `google-aip/` | google.aip.dev | Google API Improvement Proposals |
| `microsoft-rest-api/` | github.com/microsoft/api-guidelines | Azure + Graph REST guidelines |

### Zalando RESTful API Guidelines

Opinionated, numbered rules covering: resource modelling, URI design, HTTP method semantics, versioning strategy, error schema (`application/problem+json`), pagination (cursor and offset), filtering, hypermedia (HAL), security headers, and asynchronous patterns. Used across all Zalando production APIs.

### Google API Improvement Proposals (AIP)

Google's API governance framework used across all Google Cloud APIs:
- Resource-oriented design principles
- Standard methods: List, Get, Create, Update, Delete
- Custom methods, long-running operations
- Pagination, filtering, field masks
- Error modelling and status codes
- Client library design guidance

### Microsoft REST API Guidelines

Microsoft's internal standards for Azure and Microsoft Graph APIs:
- Naming conventions (camelCase, plural nouns)
- Versioning (date-based, breaking vs. non-breaking)
- Error contracts (`code`/`message`/`target`/`details`)
- Collections: filtering (`$filter`), sorting, pagination
- Async request/reply pattern (202 + Location header)
- Delta queries and change tracking

## Downloading

```bash
python download_api_design_docs.py
```

Requires: `pip install requests beautifulsoup4 html2text`

## Ingesting

```bash
grounded-code-mcp ingest sources/api-design --collection api_design
```
