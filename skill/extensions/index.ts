import { execFileSync } from "child_process";
import { Type } from "typebox";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

// Keep tool output well under pi's 50 KB / 2000-line truncation threshold.
const MAX_CHARS = 40_000;

function runCli(args: string[]): unknown {
  try {
    const stdout = execFileSync("grounded-code-mcp", [...args, "--json"], {
      encoding: "utf8",
      maxBuffer: 4 * 1024 * 1024,
      timeout: 30_000,
    });
    return JSON.parse(stdout);
  } catch (err: unknown) {
    const e = err as { stderr?: string | Buffer; message?: string };
    const stderr =
      typeof e.stderr === "string"
        ? e.stderr
        : e.stderr?.toString() ?? "";
    throw new Error(`grounded-code-mcp: ${stderr || e.message || String(err)}`);
  }
}

function toText(data: unknown): string {
  const text = JSON.stringify(data, null, 2);
  return text.length > MAX_CHARS
    ? text.slice(0, MAX_CHARS) + "\n... [truncated]"
    : text;
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "grounded_search",
    label: "Grounded Search",
    description:
      "Search the local knowledge base for authoritative documentation. " +
      "Returns vector-search results with score, source path, and content. " +
      "Prefer this over training-data guesses for any topic in the collection map.",
    promptSnippet: "search local knowledge base for vetted documentation",
    promptGuidelines: [
      "Use grounded_search before answering questions about documented topics " +
        "to get citation-backed, authoritative sources.",
      "Pass a collection name (e.g. 'python', 'dotnet', 'patterns', 'internal') " +
        "to narrow the search when you know the domain.",
    ],
    parameters: Type.Object({
      query: Type.String({
        description: "Search query — 2–6 content words work best, no filler",
      }),
      collection: Type.Optional(
        Type.String({
          description:
            "Collection suffix to search, e.g. 'python', 'patterns', 'internal'. " +
            "Omit to search all collections.",
        })
      ),
      n_results: Type.Optional(
        Type.Number({ description: "Results to return (default 5, max 20)" })
      ),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const args = ["search", params.query];
      if (params.collection) args.push("--collection", params.collection);
      if (params.n_results) args.push("-n", String(params.n_results));
      const data = runCli(args);
      return {
        content: [{ type: "text", text: toText(data) }],
        details: { data },
      };
    },
  });

  pi.registerTool({
    name: "grounded_search_code",
    label: "Grounded Code Search",
    description:
      "Search the knowledge base for code examples only. " +
      "Filters to code blocks and optionally narrows by programming language.",
    promptSnippet: "search local knowledge base for code examples",
    promptGuidelines: [
      "Use grounded_search_code when you need a concrete implementation example " +
        "rather than prose documentation.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "What kind of code to find" }),
      language: Type.Optional(
        Type.String({
          description:
            "Language filter, e.g. 'python', 'typescript', 'csharp', 'rust'",
        })
      ),
      n_results: Type.Optional(
        Type.Number({ description: "Results to return (default 5)" })
      ),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const args = ["search-code", params.query];
      if (params.language) args.push("--language", params.language);
      if (params.n_results) args.push("-n", String(params.n_results));
      const data = runCli(args);
      return {
        content: [{ type: "text", text: toText(data) }],
        details: { data },
      };
    },
  });

  pi.registerTool({
    name: "grounded_list_sources",
    label: "Grounded List Sources",
    description:
      "List all documents ingested into the knowledge base. " +
      "Use this to discover what documentation is available before searching.",
    promptSnippet: "list ingested knowledge base sources",
    promptGuidelines: [
      "Use grounded_list_sources to see which books and docs are indexed " +
        "before deciding which collection to search.",
    ],
    parameters: Type.Object({
      collection: Type.Optional(
        Type.String({
          description:
            "Filter by collection suffix, e.g. 'python'. Omit to list all.",
        })
      ),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const args = ["list-sources"];
      if (params.collection) args.push("--collection", params.collection);
      const data = runCli(args);
      return {
        content: [{ type: "text", text: toText(data) }],
        details: { data },
      };
    },
  });

  pi.registerTool({
    name: "grounded_source_info",
    label: "Grounded Source Info",
    description:
      "Get detailed metadata for a specific ingested source document: " +
      "chunk count, ingestion date, SHA-256, page count.",
    promptSnippet: "get metadata for a specific knowledge base source",
    promptGuidelines: [
      "Use grounded_source_info after grounded_list_sources to inspect a " +
        "specific document's ingestion details.",
    ],
    parameters: Type.Object({
      source_path: Type.String({
        description:
          "Relative path of the source as returned by grounded_list_sources",
      }),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const data = runCli(["source-info", params.source_path]);
      return {
        content: [{ type: "text", text: toText(data) }],
        details: { data },
      };
    },
  });

  pi.registerTool({
    name: "grounded_query_graph",
    label: "Grounded Query Graph",
    description:
      "Traverse the concept graph to find relationships around a concept. " +
      "Returns matched nodes, typed relationships, and linked source documents.",
    promptSnippet: "traverse concept graph for relationships",
    promptGuidelines: [
      "Use grounded_query_graph to explore how engineering concepts relate — " +
        "e.g. 'dependency injection', 'CQRS', 'clean architecture', 'OWASP'.",
    ],
    parameters: Type.Object({
      concept: Type.String({
        description: "Concept to look up in the graph",
      }),
      depth: Type.Optional(
        Type.Number({ description: "Traversal depth 1–3 (default 2)" })
      ),
      domain: Type.Optional(
        Type.String({
          description: "Filter by domain, e.g. 'design', 'architecture', 'security'",
        })
      ),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const args = ["query-graph", params.concept];
      if (params.depth) args.push("--depth", String(params.depth));
      if (params.domain) args.push("--domain", params.domain);
      const data = runCli(args);
      return {
        content: [{ type: "text", text: toText(data) }],
        details: { data },
      };
    },
  });
}
