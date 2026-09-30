"""
End-to-End Pipeline Tests for Laya Desktop Agent.
"""

from core.config import load_config
from app.daemon import LayaDaemon


def test_e2e_pipeline_diagnostic_execution():
    config = load_config(profile="balanced")
    daemon = LayaDaemon(config, diagnostic_mode=True)
    daemon.model_mgr.laya.load()

    # Test Open Chrome command
    res1 = daemon.coordinator.process_text_command("Open Chrome")
    assert res1.intent == "open_application"
    assert res1.action_executed == "open_application"
    assert res1.success is True
    assert res1.verified is True
    assert "[DIAGNOSTIC]" in res1.response_text
    assert "laya_routing" in res1.metrics.durations_ms

    # Test Screenshot command
    res2 = daemon.coordinator.process_text_command("Take a screenshot")
    assert res2.action_executed == "take_screenshot"
    assert res2.success is True
    assert res2.verified is True

    # Test Scroll command
    res3 = daemon.coordinator.process_text_command("Scroll down twice")
    assert res3.intent == "scroll"
    assert res3.action_executed == "scroll"
    assert res3.success is True
    assert res3.verified is True

    daemon.model_mgr.shutdown()
