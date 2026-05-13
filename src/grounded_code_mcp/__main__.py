"""CLI entry point for grounded-code-mcp."""

import json as _json
import subprocess  # nosec B404
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from grounded_code_mcp.config import Settings

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """grounded-code-mcp: RAG over a persistent knowledge base."""


@cli.command()
@click.option("--force", is_flag=True, help="Force re-ingestion of all files")
@click.option("--collection", help="Target collection name")
@click.argument("path", required=False, type=click.Path(exists=True))
def ingest(force: bool, collection: str | None, path: str | None) -> None:
    """Ingest documents into the knowledge base."""
    from grounded_code_mcp.ingest import ingest_documents

    settings = Settings.load()
    source_path = Path(path) if path else None

    console.print("[bold]Starting ingestion...[/bold]")

    def _on_file(file_path: Path) -> None:
        console.print(f"  [dim]Processing:[/dim] {file_path.name}")

    stats = ingest_documents(
        settings,
        path=source_path,
        collection=collection,
        force=force,
        progress_callback=_on_file,
    )

    if stats.success:
        console.print(f"[green]✓[/green] Ingested {stats.files_ingested} files")
        console.print(f"  Scanned: {stats.files_scanned}")
        console.print(f"  Skipped: {stats.files_skipped}")
        console.print(f"  Chunks created: {stats.chunks_created}")
    else:
        console.print("[red]✗[/red] Ingestion completed with errors")
        console.print(f"  Files failed: {stats.files_failed}")
        for error in stats.errors[:5]:  # Show first 5 errors
            console.print(f"  [red]Error:[/red] {error}")


@cli.command()
def status() -> None:
    """Show knowledge base status."""
    from grounded_code_mcp.embeddings import EmbeddingClient
    from grounded_code_mcp.manifest import Manifest
    from grounded_code_mcp.vectorstore import create_vector_store

    settings = Settings.load()

    # Show settings info
    console.print("[bold]Knowledge Base Status[/bold]\n")

    console.print(f"Sources directory: {settings.knowledge_base.sources_dir}")
    console.print(f"Data directory: {settings.knowledge_base.data_dir}")
    console.print(f"Vector store: {settings.vectorstore.provider}")
    console.print(f"Embedding model: {settings.ollama.model}")
    console.print()

    # Check Ollama
    embedder = EmbeddingClient.from_settings(settings.ollama)
    health = embedder.health_check()

    if health["healthy"]:
        console.print("[green]✓[/green] Ollama is running")
        console.print(f"  Model: {health['model']}")
    else:
        console.print("[red]✗[/red] Ollama is not ready")
        if health["error"]:
            console.print(f"  Error: {health['error']}")

    console.print()

    # Show manifest stats
    manifest_path = settings.knowledge_base.manifest_path
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        stats = manifest.stats()

        console.print("[bold]Manifest[/bold]")
        console.print(f"  Total sources: {stats['total_sources']}")
        console.print(f"  Total chunks: {stats['total_chunks']}")
        console.print(f"  Last updated: {stats['updated_at']}")

        if stats["collections"]:
            console.print("\n[bold]Collections:[/bold]")
            table = Table()
            table.add_column("Collection", style="cyan")
            table.add_column("Sources", justify="right")

            for coll_name, count in stats["collections"].items():
                table.add_row(coll_name, str(count))

            console.print(table)
    else:
        console.print("[yellow]No manifest found. Run 'ingest' to create one.[/yellow]")

    # Show vector store collections
    try:
        store = create_vector_store(settings)
        collections = store.list_collections()
        if collections:
            console.print("\n[bold]Vector Store Collections:[/bold]")
            table = Table()
            table.add_column("Collection", style="cyan")
            table.add_column("Chunks", justify="right")

            for coll in collections:
                count = store.collection_count(coll)
                table.add_row(coll, str(count))

            console.print(table)
    except Exception as e:
        console.print(f"[yellow]Could not read vector store: {e}[/yellow]")


@cli.command()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default=None,
    help="Transport protocol (default: stdio)",
)
@click.option(
    "--host", default="127.0.0.1", help="Host to bind HTTP transport (default: 127.0.0.1)"
)
@click.option("--port", default=8080, type=int, help="Port for HTTP transport (default: 8080)")
def serve(debug: bool, transport: str | None, host: str, port: int) -> None:
    """Start the MCP server."""
    from grounded_code_mcp.server import run_server

    console.print("[bold]Starting MCP server...[/bold]")
    if debug:
        console.print("[yellow]Debug mode enabled[/yellow]")
    if transport:
        console.print(f"Transport: {transport} on {host}:{port}")

    run_server(debug=debug, transport=transport, host=host, port=port)  # type: ignore[arg-type]


@cli.command()
@click.argument("query")
@click.option("--collection", help="Search specific collection")
@click.option("-n", "--n-results", default=5, help="Number of results")
@click.option("--min-score", default=0.5, help="Minimum similarity score")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON for agent/script use")
def search(
    query: str, collection: str | None, n_results: int, min_score: float, as_json: bool
) -> None:
    """Search the knowledge base."""
    from grounded_code_mcp.embeddings import EmbeddingClient, get_helpful_error_message
    from grounded_code_mcp.vectorstore import create_vector_store

    settings = Settings.load()
    embedder = EmbeddingClient.from_settings(settings.ollama)

    try:
        embedder.ensure_ready()
    except Exception as e:
        msg = get_helpful_error_message(e)
        if as_json:
            click.echo(_json.dumps([{"error": msg}], indent=2))
        else:
            console.print(f"[red]{msg}[/red]")
        return

    store = create_vector_store(settings)

    if not as_json:
        console.print(f"[dim]Searching for: {query}[/dim]\n")
    result = embedder.embed(query, is_query=True)

    if collection:
        collections = [f"{settings.vectorstore.collection_prefix}{collection}"]
    else:
        collections = store.list_collections()

    if not collections:
        if as_json:
            click.echo(_json.dumps([], indent=2))
        else:
            console.print("[yellow]No collections found. Run 'ingest' first.[/yellow]")
        return

    all_results = []
    for coll in collections:
        try:
            results = store.search(
                coll,
                result.embedding,
                n_results=n_results,
                min_score=min_score,
            )
            all_results.extend(results)
        except Exception as e:
            if not as_json:
                console.print(f"[yellow]Error searching {coll}: {e}[/yellow]")

    all_results.sort(key=lambda x: x.score, reverse=True)
    all_results = all_results[:n_results]

    if as_json:
        click.echo(
            _json.dumps(
                [
                    {
                        "content": r.content,
                        "score": round(r.score, 4),
                        "source_path": r.source_path,
                        "heading_context": r.heading_context,
                    }
                    for r in all_results
                ],
                indent=2,
            )
        )
        return

    if not all_results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, r in enumerate(all_results, 1):
        console.print(f"[bold cyan]Result {i}[/bold cyan] (score: {r.score:.4f})")
        console.print(f"[dim]Source: {r.source_path}[/dim]")
        if r.heading_context:
            console.print(f"[dim]Context: {' > '.join(r.heading_context)}[/dim]")
        content = r.content[:500] + "..." if len(r.content) > 500 else r.content
        console.print(content)
        console.print()


@cli.command()
@click.option("--collection", help="Convert a specific collection (default: all)")
@click.option("--force", is_flag=True, help="Re-convert even if a sidecar already exists")
@click.option("--dry-run", is_flag=True, help="List files without converting")
@click.option("--no-ocr", "disable_ocr", is_flag=True, help="Disable OCR (overrides config)")
@click.argument("path", required=False, type=click.Path(exists=True))
def convert(
    collection: str | None, force: bool, dry_run: bool, disable_ocr: bool, path: str | None
) -> None:
    """Pre-convert documents to Markdown sidecars using GPU-accelerated Docling."""
    from grounded_code_mcp.parser import (
        PLAINTEXT_EXTENSIONS,
        DocumentParser,
        scan_directory,
        sidecar_path,
    )

    settings = Settings.load()
    enable_ocr = False if disable_ocr else settings.docling.enable_ocr

    # Single-file path: caller is already an isolated process — parse directly.
    # Directory/collection/default: spawn one subprocess per file so a native
    # crash (heap corruption from Docling's PDF stack) only kills that file's
    # process and doesn't abort the whole batch.
    is_single_file = path is not None and Path(path).is_file()

    if path:
        p = Path(path)
        if p.is_file():
            files = [p] if p.suffix.lower() not in PLAINTEXT_EXTENSIONS else []
        else:
            root = p
            files = [
                f for f in scan_directory(root) if f.suffix.lower() not in PLAINTEXT_EXTENSIONS
            ]
    elif collection:
        for source_path, coll_name in settings.collections.items():
            if coll_name == collection:
                root = Path(source_path)
                break
        else:
            console.print(f"[red]Collection '{collection}' not found in config.[/red]")
            return
        files = [f for f in scan_directory(root) if f.suffix.lower() not in PLAINTEXT_EXTENSIONS]
    else:
        root = settings.knowledge_base.sources_dir
        files = [f for f in scan_directory(root) if f.suffix.lower() not in PLAINTEXT_EXTENSIONS]

    if not files:
        console.print("[yellow]No files to convert.[/yellow]")
        return

    # Only instantiate DocumentParser in single-file mode; subprocesses create their own.
    parser = (
        DocumentParser(enable_ocr=enable_ocr, docling_settings=settings.docling)
        if is_single_file
        else None
    )

    converted = 0
    skipped = 0
    failed = 0

    for file in files:
        sc = sidecar_path(file)
        if sc.exists() and not force:
            skipped += 1
            continue
        if dry_run:
            console.print(f"  [dim]Would convert:[/dim] {file}")
            converted += 1
            continue

        if is_single_file:
            if parser is None:
                raise RuntimeError("Parser not initialized for single-file conversion.")
            try:
                result = parser.parse(file)
                sc.write_text(result.content, encoding="utf-8")
                console.print(f"  [green]Converted:[/green] {file.name}")
                converted += 1
            except Exception as e:
                console.print(f"  [red]Failed:[/red] {file.name}: {e}")
                failed += 1
        else:
            cmd = [sys.executable, "-m", "grounded_code_mcp", "convert"]
            if disable_ocr:
                cmd.append("--no-ocr")
            if force:
                cmd.append("--force")
            cmd.append(str(file))
            proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603  # nosec B603
            if proc.returncode == 0:
                console.print(f"  [green]Converted:[/green] {file.name}")
                converted += 1
            else:
                console.print(f"  [red]Failed:[/red] {file.name}")
                if proc.stderr:
                    console.print(f"    {proc.stderr.strip()}")
                failed += 1

    console.print(
        f"\n[bold]Summary:[/bold] {converted} converted / {skipped} skipped / {failed} failed"
    )


@cli.command("build-graph")
@click.argument("path", required=False, type=click.Path(exists=True))
@click.option(
    "--dry-run", is_flag=True, help="Parse and validate triples without writing the graph."
)
def build_graph(path: str | None, dry_run: bool) -> None:
    """Build the concept graph from RELATIONSHIPS.md files."""
    from graph.graph_builder import BuildStats, build
    from graph.graph_store import GraphStore

    settings = Settings.load()
    input_path = Path(path) if path else settings.knowledge_base.sources_dir

    store = GraphStore()
    store.load()

    prefix = "[dry-run] " if dry_run else ""
    console.print(f"[bold]{prefix}Building concept graph from:[/bold] {input_path}")

    stats: BuildStats = build(input_path, store, dry_run=dry_run)

    if dry_run:
        console.print(
            f"[dim]{stats.files_processed} files, {stats.triples_parsed} triples, "
            f"{stats.triples_skipped} skipped (dry run — nothing written)[/dim]"
        )
    else:
        console.print(f"[green]✓[/green] {stats.files_processed} files processed")
        console.print(f"  Triples parsed: {stats.triples_parsed}")
        console.print(f"  Triples skipped: {stats.triples_skipped}")
        console.print(f"  Nodes added: {stats.nodes_added}")
        if stats.errors:
            for err in stats.errors[:5]:
                console.print(f"  [red]Error:[/red] {err}")


@cli.command("search-code")
@click.argument("query")
@click.option("--language", help="Filter by programming language (e.g. python, typescript)")
@click.option("-n", "--n-results", default=5, help="Number of results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON for agent/script use")
def search_code(query: str, language: str | None, n_results: int, as_json: bool) -> None:
    """Search for code examples in the knowledge base."""
    from grounded_code_mcp import server as _server

    settings = Settings.load()
    _server.initialize(settings)
    results = _server._search_code_examples_impl(query, language, n_results)

    if as_json:
        click.echo(_json.dumps(results, indent=2))
        return

    if not results:
        console.print("[yellow]No code examples found.[/yellow]")
        return

    if isinstance(results, list) and results and "error" in results[0]:
        console.print(f"[red]{results[0]['error']}[/red]")
        return

    for i, r in enumerate(results, 1):
        lang = r.get("language", "text")
        source = r.get("source_path", "")
        score = r.get("score", 0.0)
        code = r.get("code", "")
        context = r.get("heading_context") or []

        console.print(
            f"[bold cyan]Result {i}[/bold cyan] [[green]{lang}[/green]] (score: {score:.4f})"
        )
        console.print(f"[dim]Source: {source}[/dim]")
        if context:
            console.print(f"[dim]Context: {' > '.join(context)}[/dim]")
        console.print(code[:800] + "..." if len(code) > 800 else code)
        console.print()


@cli.command("list-sources")
@click.option("--collection", help="Filter by collection (bare suffix, e.g. 'python')")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON for agent/script use")
def list_sources(collection: str | None, as_json: bool) -> None:
    """List all ingested sources in the knowledge base."""
    from grounded_code_mcp import server as _server

    settings = Settings.load()
    _server.initialize(settings)
    results = _server._list_sources_impl(collection)

    if as_json:
        click.echo(_json.dumps(results, indent=2))
        return

    if not results:
        console.print("[yellow]No sources found.[/yellow]")
        return

    table = Table(title="Ingested Sources")
    table.add_column("Title", style="cyan")
    table.add_column("Collection", style="dim")
    table.add_column("Type", justify="center")
    table.add_column("Chunks", justify="right")

    for s in results:
        table.add_row(
            s.get("title") or s.get("path", ""),
            s.get("collection", ""),
            s.get("file_type", ""),
            str(s.get("chunk_count", 0)),
        )

    console.print(table)


@cli.command("source-info")
@click.argument("source_path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON for agent/script use")
def source_info(source_path: str, as_json: bool) -> None:
    """Get detailed metadata for a specific ingested source."""
    from grounded_code_mcp import server as _server

    settings = Settings.load()
    _server.initialize(settings)
    result = _server._get_source_info_impl(source_path)

    if as_json:
        click.echo(_json.dumps(result, indent=2))
        return

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    table = Table(title="Source Information")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    for key, value in result.items():
        table.add_row(key, str(value))

    console.print(table)


@cli.command("query-graph")
@click.argument("concept")
@click.option(
    "--depth",
    default=2,
    type=click.IntRange(1, 3),
    help="Graph traversal depth (1-3, default: 2)",
)
@click.option("--domain", help="Filter results by domain")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON for agent/script use")
def query_graph(concept: str, depth: int, domain: str | None, as_json: bool) -> None:
    """Query the concept graph for relationships around a concept."""
    from grounded_code_mcp import server as _server

    settings = Settings.load()
    _server.initialize(settings)
    result = _server._query_graph_impl(concept, depth, domain)

    if as_json:
        click.echo(_json.dumps(result, indent=2))
        return

    console.print(f"[bold]Graph Query:[/bold] {concept}\n")
    console.print(result.get("summary", "No summary available."))

    nodes = result.get("matched_nodes", [])
    if nodes:
        console.print(f"\n[bold]Matched Nodes ({len(nodes)}):[/bold]")
        for node in nodes[:10]:
            node_id = node.get("id", "")
            node_type = node.get("type", "")
            node_domain = node.get("domain", "")
            desc = node.get("description", "")
            console.print(
                f"  [cyan]{node_id}[/cyan] [{node_type}]"
                + (f" ({node_domain})" if node_domain else "")
            )
            if desc:
                console.print(f"    [dim]{desc[:100]}[/dim]")

    rels = result.get("relationships", [])
    if rels:
        console.print(f"\n[bold]Relationships ({len(rels)}):[/bold]")
        for rel in rels[:10]:
            console.print(
                f"  {rel.get('from', '')} [dim]→[/dim] {rel.get('rel', '')} "
                f"[dim]→[/dim] {rel.get('to', '')}"
            )


if __name__ == "__main__":
    cli()
