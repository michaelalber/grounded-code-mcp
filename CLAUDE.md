# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

grounded-code-mcp is a local MCP (Model Context Protocol) server that provides RAG capabilities over a persistent knowledge base. The server enables AI coding assistants (Claude Code, OpenCode, etc.) to ground their responses in vetted, authoritative documentation.

## Philosophy

This project addresses LLM limitations (hallucination, brittle compositionality, lack of grounding) by anchoring AI responses to curated technical documentation via RAG.

**Core principles:**
- Ground responses in curated knowledge base content
- Treat AI output as requiring human review
- Use structured outputs with validation
- Pair with TDD workflows

## Tech Stack (Planned)

- **Language:** Python
- **Linting:** ruff
- **Type checking:** pyright
- **Testing:** pytest

## Project Status

This is a greenfield project. The architecture described here represents the intended design.
