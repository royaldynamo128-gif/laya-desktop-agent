"""
Controlled Action Executor for Laya Desktop Agent.
Executes registered tools with timeout management, safety permission gating, and emergency stop enforcement.
"""

from typing import Optional, Dict, Any
import concurrent.futures
import logging
import time
from core.config import SafetyConfig
from tools.base import Tool, ToolResult
from tools.registry import ToolRegistry
from execution.parameter_binder import BoundAction
from safety.policy import SafetyPolicy
from safety.emergency_stop import EmergencyStop

logger = logging.getLogger(__name__)


class ActionExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: SafetyPolicy,
        emergency_stop: EmergencyStop,
        config: SafetyConfig,
    ):
        self.registry = registry
        self.policy = policy
        self.emergency_stop = emergency_stop
        self.config = config
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ToolExec")

    def execute_action(self, action: BoundAction, is_confirmed: bool = False) -> ToolResult:
        """Execute a single bound action under controlled safety policies."""
        # 1. Emergency Stop Check
        if self.emergency_stop.is_stopped():
            msg = "Action rejected: Agent is in Emergency Stop state."
            logger.warning(msg)
            return ToolResult(success=False, message=msg, error="Emergency Stop Active")

        # 2. Tool Lookup
        tool = self.registry.get(action.tool_name)
        if not tool:
            msg = f"Unknown tool: '{action.tool_name}'."
            logger.error(msg)
            return ToolResult(success=False, message=msg, error="Tool Not Found")

        # 3. Safety Policy Permission Check
        permitted, reason = self.policy.check_permission(tool, is_confirmed=is_confirmed)
        if not permitted:
            return ToolResult(success=False, message=reason, error="Permission Denied")

        # 4. Timeout-enforced Execution
        start = time.perf_counter()
        logger.info("Executing tool [%s] with arguments: %s", action.tool_name, action.arguments)

        try:
            future = self._executor.submit(tool.execute, **action.arguments)
            result = future.result(timeout=self.config.execution_timeout_sec)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info("Tool [%s] finished in %.1f ms | Success: %s", action.tool_name, elapsed_ms, result.success)
            return result
        except concurrent.futures.TimeoutError:
            msg = f"Tool execution timed out after {self.config.execution_timeout_sec}s."
            logger.error(msg)
            return ToolResult(success=False, message=msg, error="Timeout")
        except Exception as e:
            msg = f"Tool execution error: {e}"
            logger.error(msg)
            return ToolResult(success=False, message=msg, error=str(e))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
