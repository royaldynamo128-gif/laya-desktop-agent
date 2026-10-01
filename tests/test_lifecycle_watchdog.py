"""
Unit Tests for CommandLifecycle and Watchdog Timer.
Verifies stage progression, timeout cancellation, and watchdog recovery.
"""

import time
import pytest
from core.lifecycle import CommandLifecycle, CommandStage, LifecycleManager


def test_lifecycle_stage_transitions():
    lifecycle = CommandLifecycle("cmd-test-001")
    assert lifecycle.current_stage == CommandStage.IDLE
    assert not lifecycle.is_finalized

    lifecycle.transition(CommandStage.PTT_START)
    assert lifecycle.current_stage == CommandStage.PTT_START

    lifecycle.transition(CommandStage.STT_START)
    lifecycle.transition(CommandStage.STT_COMPLETE, detail="Open Chrome")
    lifecycle.transcription = "Open Chrome"

    lifecycle.transition(CommandStage.LAYA_START)
    lifecycle.transition(CommandStage.LAYA_COMPLETE, detail="open_application")
    lifecycle.intent = "open_application"

    lifecycle.transition(CommandStage.ENTITY_READY, detail="Google Chrome")
    lifecycle.target = "Google Chrome"

    lifecycle.transition(CommandStage.EXECUTION_START)
    lifecycle.transition(CommandStage.EXECUTION_COMPLETE)

    lifecycle.transition(CommandStage.VERIFICATION_START)
    lifecycle.transition(CommandStage.VERIFICATION_COMPLETE)

    lifecycle.succeed(detail="Chrome opened")
    assert lifecycle.current_stage == CommandStage.SUCCESS
    assert lifecycle.is_finalized

    diag = lifecycle.to_diagnostic_dict()
    assert diag["command_id"] == "cmd-test-001"
    assert diag["transcription"] == "Open Chrome"
    assert diag["target"] == "Google Chrome"
    assert diag["final_stage"] == "SUCCESS"
    assert len(diag["stages"]) == 11


def test_lifecycle_watchdog_timeout_trigger():
    timeout_fired = False
    timed_out_cid = None

    def on_timeout(lc: CommandLifecycle):
        nonlocal timeout_fired, timed_out_cid
        timeout_fired = True
        timed_out_cid = lc.command_id

    lifecycle = CommandLifecycle("cmd-timeout-002")
    lifecycle.transition(CommandStage.STT_START)
    lifecycle.start_watchdog(timeout_sec=0.1, on_timeout=on_timeout)

    # Wait for watchdog to trigger
    time.sleep(0.2)

    assert timeout_fired is True
    assert timed_out_cid == "cmd-timeout-002"
    assert lifecycle.current_stage == CommandStage.TIMEOUT
    assert lifecycle.is_finalized
    assert "timed out after 0.1s" in (lifecycle.error_message or "")


def test_lifecycle_watchdog_cancelled_on_success():
    timeout_fired = False

    def on_timeout(lc: CommandLifecycle):
        nonlocal timeout_fired
        timeout_fired = True

    lifecycle = CommandLifecycle("cmd-cancel-003")
    lifecycle.transition(CommandStage.STT_START)
    lifecycle.start_watchdog(timeout_sec=0.2, on_timeout=on_timeout)

    # Complete before watchdog expires
    time.sleep(0.05)
    lifecycle.succeed(detail="Quick success")

    time.sleep(0.25)
    assert timeout_fired is False
    assert lifecycle.current_stage == CommandStage.SUCCESS


def test_lifecycle_manager_tracking():
    mgr = LifecycleManager(history_size=3)
    c1 = mgr.new_command()
    assert c1.command_id == "cmd-001"
    assert mgr.active_command is c1

    c1.succeed()
    mgr.clear_active(c1)
    assert mgr.active_command is None

    c2 = mgr.new_command()
    assert c2.command_id == "cmd-002"
    c3 = mgr.new_command()
    assert c3.command_id == "cmd-003"
    c4 = mgr.new_command()
    assert c4.command_id == "cmd-004"

    # History size is 3, so c1 should be evicted
    history = mgr.get_all_history()
    assert len(history) == 3
    ids = [h["command_id"] for h in history]
    assert ids == ["cmd-002", "cmd-003", "cmd-004"]
    assert mgr.get_last_command().command_id == "cmd-004"
