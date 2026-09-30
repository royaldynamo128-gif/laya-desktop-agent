"""
Convai Laya Decision Engine for Laya Desktop Agent.
Uses non-autoregressive ModernBERT (421M params) for sub-35ms structured decision routing,
intent classification, safety gating, and post-action observation flags.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import gc
import logging
import time
from core.config import LayaConfig

logger = logging.getLogger(__name__)

# Intent taxonomy and criteria for Laya
INTENT_CRITERIA = {
    "open_application": "User wants to launch, open, or start an application, program, directory, or folder (e.g. Chrome, Telegram, terminal, files, Projects folder, Downloads)",
    "close_application": "User wants to close, exit, or quit a window or program",
    "switch_window": "User wants to switch to, bring up, or focus an existing window or app (e.g. switch to browser, switch to Opera)",
    "window_action": "User wants to minimize, maximize, resize, or move the current window",
    "scroll": "User wants to scroll up, down, page up, or page down",
    "browser_action": "User wants to search the web, open a webpage, navigate, or browse (e.g. YouTube, Google)",
    "screen_query": "User wants to take a screenshot, capture screenshot, screen capture, read screen, or inspect screen",
    "input_action": "User wants to type text, press Enter, press Tab, or send keyboard input",
    "system_setting": "User wants to change audio volume, mute sound, or control media playback",
    "terminal_command": "User wants to run a specific command in terminal",
    "other": "Ambiguous, unrecognized, conversational, or clarification needed",
}

SAFETY_CRITERIA = {
    "safe": "Action is completely harmless, read-only, or standard UI interaction (open app, switch window, scroll, screenshot)",
    "moderate": "Action modifies files, settings, or sends messages, but is non-destructive",
    "high_risk": "Action is potentially destructive, deletes files, modifies system security, or runs arbitrary destructive commands",
}


@dataclass
class LayaDecision:
    intent: str
    confidence: float
    safety_level: str
    requires_confirmation: bool
    requires_observation: bool
    raw_decision: Dict[str, Any]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "safety_level": self.safety_level,
            "requires_confirmation": self.requires_confirmation,
            "requires_observation": self.requires_observation,
            "latency_ms": round(self.latency_ms, 2),
        }


class LayaEngine:
    def __init__(self, config: LayaConfig):
        self.config = config
        self._router = None
        self._current_device = config.device
        self._last_used_time = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._router is not None

    def load(self, device: Optional[str] = None) -> None:
        """Initialize Laya Router with device specification."""
        target_device = device or self.config.device
        if target_device == "auto":
            try:
                import torch
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                target_device = "cpu"

        if self._router is not None and self._current_device == target_device:
            return

        logger.info("Initializing Laya Router on %s...", target_device)
        self.unload()

        start = time.perf_counter()
        try:
            from laya import Router
            self._router = Router(
                max_loaded=1,
                default="english",
                device=target_device,
            )
            self._current_device = target_device
            self._last_used_time = time.time()
            logger.info("Laya Router initialized in %.2f s on %s", time.perf_counter() - start, target_device)
        except Exception as e:
            if target_device == "cuda":
                logger.warning("CUDA initialization failed for Laya (%s). Falling back to CPU...", e)
                self.load(device="cpu")
            else:
                logger.error("Failed to load Laya on CPU: %s", e)
                raise

    def decide(self, user_command: str, context: Optional[Dict[str, Any]] = None) -> LayaDecision:
        """Route and classify user voice command into structured decision in a single forward pass."""
        if not self.is_loaded:
            self.load()

        self._last_used_time = time.time()
        start = time.perf_counter()

        state = {
            "message": user_command,
            "desktop_context": context or {},
        }

        questions = {
            "intent": {
                "type": "choice",
                "instructions": "What is the primary action or intent the user wants to perform on their computer?",
                "criteria": INTENT_CRITERIA,
            },
            "safety_level": {
                "type": "choice",
                "instructions": "What is the safety risk tier of this requested computer action?",
                "criteria": SAFETY_CRITERIA,
            },
            "requires_confirmation": {
                "type": "noul",
                "instructions": "Does this action involve deletion, killing processes, or significant permanent system alteration?",
            },
            "requires_observation": {
                "type": "noul",
                "instructions": "Should the system observe and verify the desktop state after executing this action?",
            },
        }

        try:
            raw = self._router.predict(state, questions)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            answers = raw.get("answers", {})
            intent_ans = answers.get("intent", {})
            safety_ans = answers.get("safety_level", {})
            confirm_ans = answers.get("requires_confirmation", {})
            observe_ans = answers.get("requires_observation", {})

            intent = intent_ans.get("choice", "other")
            confidence = intent_ans.get("answer_confidence", intent_ans.get("confidence", 0.8))
            safety = safety_ans.get("choice", "safe")
            
            # Noul answers return calibrated probability
            confirm_prob = confirm_ans.get("noul", 0.0)
            observe_prob = observe_ans.get("noul", 0.8)

            requires_confirm = confirm_prob > 0.5 or safety == "high_risk"
            requires_observe = observe_prob > 0.4

            decision = LayaDecision(
                intent=intent,
                confidence=float(confidence),
                safety_level=safety,
                requires_confirmation=requires_confirm,
                requires_observation=requires_observe,
                raw_decision=raw,
                latency_ms=elapsed_ms,
            )

            logger.info(
                "Laya decision: intent=%s (conf: %.2f) | safety=%s | confirm=%s | observe=%s [%.1f ms]",
                intent,
                confidence,
                safety,
                requires_confirm,
                requires_observe,
                elapsed_ms,
            )
            return decision

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.error("Laya decision inference failed: %s", e)
            # Safe heuristic fallback
            return self._heuristic_fallback(user_command, elapsed_ms)

    def _heuristic_fallback(self, command: str, latency_ms: float) -> LayaDecision:
        """Deterministic regex/keyword fallback if model inference encounters an unexpected issue."""
        cmd = command.lower().strip()
        intent = "other"
        safety = "safe"

        if any(w in cmd for w in ["open", "launch", "start"]):
            intent = "open_application"
        elif any(w in cmd for w in ["close", "quit", "exit"]):
            intent = "close_application"
        elif any(w in cmd for w in ["switch to", "focus", "bring up"]):
            intent = "switch_window"
        elif any(w in cmd for w in ["scroll down", "scroll up", "scroll"]):
            intent = "scroll"
        elif any(w in cmd for w in ["minimize", "maximize"]):
            intent = "window_action"
        elif any(w in cmd for w in ["search youtube", "search google", "search", "navigate"]):
            intent = "browser_action"
        elif any(w in cmd for w in ["screenshot", "what is on my screen", "read"]):
            intent = "screen_query"
        elif any(w in cmd for w in ["type", "press enter"]):
            intent = "input_action"
        elif any(w in cmd for w in ["volume", "mute"]):
            intent = "system_setting"

        return LayaDecision(
            intent=intent,
            confidence=0.85,
            safety_level=safety,
            requires_confirmation=False,
            requires_observation=True,
            raw_decision={"fallback": True},
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Release Laya model resources."""
        if self._router is not None:
            logger.info("Unloading Laya Router from %s...", self._current_device)
            self._router = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def check_idle(self) -> bool:
        if self.is_loaded and (time.time() - self._last_used_time) > self.config.idle_timeout_sec:
            logger.info("Laya Router exceeded idle timeout (%ds). Unloading.", self.config.idle_timeout_sec)
            self.unload()
            return True
        return False
