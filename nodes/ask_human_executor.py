"""
AskHuman Executor Node for LangGraph.

Requests user input using AskHumanTool.
"""

import logging
from typing import Any, Dict
import json

from agent_state import AgentStateDict
from tools.ask_human import AskHumanTool

logger = logging.getLogger(__name__)


async def ask_human_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute ask human task.

    Args:
        state: Current agent state

    Returns:
        Updated state with user response
    """
    logger.info("=== AskHuman Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    if not action_details:
        error_msg = "AskHuman executor requires action_details with question"
        logger.error(error_msg)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
        }

    try:
        # Parse action_details
        # Format: JSON {"question": "...", "options": [...]} or just text question
        try:
            params = json.loads(action_details)
            question = params.get("question", action_details)
            options = params.get("options", None)
            required = params.get("required", True)
        except json.JSONDecodeError:
            # Treat as simple question
            question = action_details
            options = None
            required = True

        # Initialize ask human tool
        ask_human_tool = AskHumanTool()

        # Ask user
        logger.info(f"Asking user: {question[:100]}")
        result = await ask_human_tool._arun(
            question=question, options=options, required=required
        )

        logger.info(f"User response: {result}")

        return {
            "last_tool_output": result,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }

    except Exception as e:
        error_msg = f"AskHuman execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }
