# grounded-code-mcp

## Philosophy
LLMs are sophisticated pattern matchers, not reasoning engines. This project mitigates known failure modes (hallucination, brittle compositionality, lack of grounding) by anchoring AI coding assistants to vetted, authoritative documentation via RAG.

**Design principles**:

* Ground responses in your curated knowledge base
* Treat AI output as "junior dev work" requiring review
* Structured outputs with validation
* Pair with TDD workflows—let tests catch AI mistakes

## Goal
Build a local MCP (Model Context Protocol) server that provides RAG capabilities over a persistent knowledge base of technical documentation, programming books, and software engineering references. The server will be queryable by Claude Code, OpenCode, and other MCP-compatible clients.
