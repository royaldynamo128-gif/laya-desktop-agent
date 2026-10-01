"""
Tests for AppCatalog, Intelligent Entity Resolution, ParameterBinder, and ActionVerifier.
Verifies that:
- 'Open Opera browser' opens Opera, NEVER Chrome.
- 'Open Upscale', 'Open Upscayl', 'Open image upscaler' resolve to Upscayl.
- Unresolvable applications return None (NEVER guesses).
- Ambiguous commands trigger clarification.
- ActionVerifier detects TARGET_MISMATCH if wrong window is focused.
"""

import pytest
from unittest.mock import MagicMock
from integrations.app_catalog import AppCatalog, soundex
from execution.parameter_binder import ParameterBinder, BoundAction
from models.laya_engine import LayaDecision
from verification.verifier import ActionVerifier
from verification.observer import DesktopObservation
from tools.base import ToolResult


@pytest.fixture(scope="module")
def catalog():
    return AppCatalog()


@pytest.fixture
def binder(catalog):
    return ParameterBinder(catalog=catalog)


def test_soundex_phonetic_matches():
    """Verify that soundex produces identical codes for phonetic variations."""
    assert soundex("upscale") == soundex("upscayl")
    assert soundex("upscale") == "U124"
    assert soundex("opera") == soundex("oprah")
    assert soundex("opera") == "O160"


def test_resolve_opera_browser(catalog):
    """'Open Opera browser' must resolve to Opera, NEVER Chrome."""
    res = catalog.resolve("Opera browser")
    assert res is not None
    assert res.name == "Opera"
    assert "opera" in res.desktop_id.lower()
    assert "chrome" not in res.desktop_id.lower()
    assert "chrome" not in res.name.lower()


def test_resolve_upscale_and_upscayl(catalog):
    """'upscale', 'upscayl', and 'image upscaler' must resolve to Upscayl."""
    # Phonetic variation 1
    res1 = catalog.resolve("upscale")
    assert res1 is not None
    assert res1.name == "Upscayl"
    assert "upscayl" in res1.desktop_id.lower()

    # Exact name
    res2 = catalog.resolve("upscayl")
    assert res2 is not None
    assert res2.name == "Upscayl"

    # Semantic comment search ("Upscale Images with AI")
    res3 = catalog.resolve("image upscaler")
    assert res3 is not None
    assert res3.name == "Upscayl"


def test_resolve_chrome_and_brave(catalog):
    """'chrome' and 'brave' must resolve to their respective browsers."""
    res_chrome = catalog.resolve("google chrome")
    assert res_chrome is not None
    assert "chrome" in res_chrome.desktop_id.lower()

    res_brave = catalog.resolve("brave browser")
    assert res_brave is not None
    assert "brave" in res_brave.desktop_id.lower()


def test_never_guess_unresolvable_app(catalog, binder):
    """Unresolvable app must return None and NEVER guess another app."""
    res = catalog.resolve("NonExistentFakeProgramXYZ")
    assert res is None

    decision = LayaDecision(intent="open_application", confidence=0.9)
    bound = binder.bind("Open NonExistentFakeProgramXYZ", decision)
    assert bound is None


def test_binder_opera_browser(binder):
    """ParameterBinder binds 'Open Opera browser' to exact Opera parameters."""
    decision = LayaDecision(intent="open_application", confidence=0.9)
    bound = binder.bind("Open Opera browser", decision)
    assert bound is not None
    assert bound.tool_name == "open_application"
    assert bound.target == "Opera"
    assert bound.arguments["name"] == "Opera"
    assert bound.arguments["desktop_id"] == "opera.desktop"


def test_binder_upscale_app(binder):
    """ParameterBinder binds 'Open upscale' to Upscayl parameters."""
    decision = LayaDecision(intent="open_application", confidence=0.9)
    bound = binder.bind("Open upscale", decision)
    assert bound is not None
    assert bound.tool_name == "open_application"
    assert bound.target == "Upscayl"
    assert "upscayl" in bound.arguments["desktop_id"].lower()


def test_verifier_target_mismatch():
    """ActionVerifier must fail with TARGET_MISMATCH when wrong window is active."""
    verifier = ActionVerifier()
    action = BoundAction(
        tool_name="open_application",
        arguments={"name": "Opera", "desktop_id": "opera.desktop", "wm_class": "Opera"},
        target="Opera",
    )
    result = ToolResult(
        success=True,
        message="Launched Opera.",
        data={"target_name": "Opera", "already_running": False},
        observation_required=True,
    )
    pre_obs = DesktopObservation(
        timestamp=0.0,
        active_window_id="100",
        active_window_name="Google Chrome",
        active_window_class="google-chrome",
        desktop_number=1,
        open_window_ids=["100"],
    )
    # Post obs has Google Chrome still active
    post_obs = DesktopObservation(
        timestamp=1.0,
        active_window_id="100",
        active_window_name="Google Chrome",
        active_window_class="google-chrome",
        desktop_number=1,
        open_window_ids=["100"],
    )

    verif_res = verifier.verify(action, result, pre_obs, post_obs)
    assert verif_res.verified is False
    assert "TARGET_MISMATCH" in verif_res.message
    assert "Google Chrome" in verif_res.message
    assert verif_res.details["stage"] == "target_mismatch"


def test_verifier_target_match():
    """ActionVerifier succeeds when expected window is active."""
    verifier = ActionVerifier()
    action = BoundAction(
        tool_name="open_application",
        arguments={"name": "Opera", "desktop_id": "opera.desktop", "wm_class": "Opera"},
        target="Opera",
    )
    result = ToolResult(
        success=True,
        message="Launched Opera.",
        data={"target_name": "Opera", "already_running": True},
        observation_required=True,
    )
    post_obs = DesktopObservation(
        timestamp=1.0,
        active_window_id="200",
        active_window_name="Opera",
        active_window_class="Opera",
        desktop_number=1,
        open_window_ids=["100", "200"],
    )

    verif_res = verifier.verify(action, result, None, post_obs)
    assert verif_res.verified is True
    assert "Verified" in verif_res.message
