"""
Browser Executor Node for LangGraph.

Executes browser automation tasks using BrowserUseTool.
"""

import logging
from typing import Any, Dict

from agent_state import AgentStateDict
from tools.browser_use import BrowserUseTool

logger = logging.getLogger(__name__)


async def browser_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute browser automation task.

    Args:
        state: Current agent state

    Returns:
        Updated state with browser execution results
    """
    logger.info("=== Browser Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    if not action_details:
        error_msg = "Browser executor requires action_details with task description"
        logger.error(error_msg)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
        }

    try:
        # Initialize browser tool
        browser_tool = BrowserUseTool()

        # Execute browser task
        logger.info(f"Executing browser task: {action_details[:100]}")
        result = await browser_tool._arun(
            task=action_details, headless=True, max_steps=20
        )

        logger.info(f"Browser execution result: {result[:200]}...")

        return {
            "last_tool_output": result,
            "iteration_count": iteration + 1,
            "current_action": "planning",  # Return to planner
        }

    except Exception as e:
        error_msg = f"Browser execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }
