from __future__ import annotations

from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli


def test_support_command_prints_feedback_links():
    result = CliRunner().invoke(cli, ["support"])

    assert result.exit_code == 0
    assert "olw does not collect telemetry" in result.output
    assert "https://github.com/kytmanov/obsidian-llm-wiki-local/issues" in result.output
    assert "https://github.com/kytmanov/obsidian-llm-wiki-local/discussions" in result.output
    assert "https://github.com/kytmanov/obsidian-llm-wiki-local" in result.output


def test_root_help_mentions_support_command():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "support" in result.output
    assert "Run `olw support`" in result.output
    assert "bug" in result.output
    assert "reports, suggestions, and feedback links" in result.output
