"""Regression tests for MCP server availability in cron jobs.

Background
==========
``cron/scheduler.py:run_job()`` constructs ``AIAgent(...)`` directly without
calling ``discover_mcp_tools()`` — the initialization that CLI and gateway
paths do at startup. Cron jobs therefore never saw any MCP tools from
``mcp_servers`` in config.yaml. See #4219.

The fix inserts ``discover_mcp_tools()`` before the ``AIAgent(...)`` call,
wrapped in try/except so a broken MCP server can't kill an otherwise
working cron job. ``discover_mcp_tools`` is idempotent — subsequent ticks
short-circuit on already-connected servers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_no_agent_cron_job_does_not_initialize_mcp():
    """Cron jobs with no_agent=True are script-only — no AIAgent, no MCP
    tools needed. We must NOT pay the MCP init cost for those."""
    from cron import scheduler

    job = {
        "id": "noagent-job",
        "name": "noagent-job",
        "no_agent": True,
        "script": "/nonexistent/script.sh",
    }

    discover_called = []

    def fake_discover():
        discover_called.append(True)
        return []

    # _run_job_script returns (ok, output); make it fail cleanly so we
    # don't need a real script file.
    with (
        patch("tools.mcp_tool.discover_mcp_tools", side_effect=fake_discover),
        patch("cron.scheduler._run_job_script", return_value=(False, "no such file")),
    ):
        scheduler.run_job(job)

    assert not discover_called, (
        "discover_mcp_tools was called for a no_agent job — wasted MCP init "
        "for a script-only cron tick"
    )


def test_no_mcp_toolset_sentinel_skips_mcp_discovery(tmp_path):
    """An explicit no_mcp opt-out must skip discovery, not merely hide tools.

    Connecting configured servers can retry for seconds before AIAgent starts;
    jobs that opted out should pay none of that startup cost.
    """
    from cron import scheduler

    (tmp_path / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    job = {
        "id": "no-mcp-job",
        "name": "no-mcp-job",
        "prompt": "hello",
        "enabled_toolsets": ["file", "no_mcp"],
    }
    runtime = {
        "provider": "openrouter",
        "api_key": "x",
        "base_url": "https://example.invalid",
        "api_mode": "chat_completions",
    }
    fake_db = MagicMock()

    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime
        ),
        patch("tools.mcp_tool.discover_mcp_tools") as mock_discover,
        patch("run_agent.AIAgent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, _, _, error = scheduler.run_job(job)

    assert success is True
    assert error is None
    mock_discover.assert_not_called()
    assert mock_agent_cls.call_args.kwargs["enabled_toolsets"] == ["file"]
