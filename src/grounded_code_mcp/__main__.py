"""CLI entry point for grounded-code-mcp."""

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

    stats = ingest_documents(
        settings,
        path=source_path,
        collection=collection,
        force=force,
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
@click.option("--host", default="127.0.0.1", help="Host to bind HTTP transport (default: 127.0.0.1)")
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
def search(query: str, collection: str | None, n_results: int, min_score: float) -> None:
    """Search the knowledge base."""
    from grounded_code_mcp.embeddings import EmbeddingClient, get_helpful_error_message
    from grounded_code_mcp.vectorstore import create_vector_store

    settings = Settings.load()
    embedder = EmbeddingClient.from_settings(settings.ollama)

    try:
        embedder.ensure_ready()
    except Exception as e:
        console.print(f"[red]{get_helpful_error_message(e)}[/red]")
        return

    store = create_vector_store(settings)

    # Generate query embedding
    console.print(f"[dim]Searching for: {query}[/dim]\n")
    result = embedder.embed(query, is_query=True)

    # Determine collections to search
    if collection:
        collections = [f"{settings.vectorstore.collection_prefix}{collection}"]
    else:
        collections = store.list_collections()

    if not collections:
        console.print("[yellow]No collections found. Run 'ingest' first.[/yellow]")
        return

    # Search
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
            console.print(f"[yellow]Error searching {coll}: {e}[/yellow]")

    # Sort and limit
    all_results.sort(key=lambda x: x.score, reverse=True)
    all_results = all_results[:n_results]

    if not all_results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Display results
    for i, r in enumerate(all_results, 1):
        console.print(f"[bold cyan]Result {i}[/bold cyan] (score: {r.score:.4f})")
        console.print(f"[dim]Source: {r.source_path}[/dim]")
        if r.heading_context:
            console.print(f"[dim]Context: {' > '.join(r.heading_context)}[/dim]")

        # Truncate content for display
        content = r.content[:500] + "..." if len(r.content) > 500 else r.content
        console.print(content)
        console.print()


if __name__ == "__main__":
    cli()
