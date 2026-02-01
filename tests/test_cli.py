"""Tests for CLI commands."""

from click.testing import CliRunner

from grounded_code_mcp.__main__ import cli


def test_cli_version() -> None:
    """Test that --version flag works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help() -> None:
    """Test that --help flag works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "grounded-code-mcp" in result.output


def test_ingest_command_exists() -> None:
    """Test that ingest command exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "Ingest documents" in result.output


def test_status_command_exists() -> None:
    """Test that status command exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output.lower()


def test_serve_command_exists() -> None:
    """Test that serve command exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.output


def test_search_command_exists() -> None:
    """Test that search command exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--help"])
    assert result.exit_code == 0
    assert "Search" in result.output
