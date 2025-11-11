from typer.testing import CliRunner

from rag_toolkit.cli import app


def test_cli_commands(monkeypatch):
    # Use test settings to avoid heavy downloads
    monkeypatch.setenv("RAG_SETTINGS", "config/test_settings.yaml")
    runner = CliRunner()

    # Build index
    result = runner.invoke(app, ["index", "--data", "data/raw"])
    assert result.exit_code == 0

    # Query
    result = runner.invoke(app, ["query", "--q", "what is in these docs?", "--k", "3"])
    assert result.exit_code == 0

    # Eval
    result = runner.invoke(app, [
        "eval", "--qrels", "data/qrels.tsv", "--queries", "data/queries.tsv", "--k", "5"
    ])
    assert result.exit_code == 0