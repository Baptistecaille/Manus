"""
Planning Executor Node for LangGraph.

Generates detailed action plans using PlanningTool.
"""

import logging
from typing import Any, Dict

from agent_state import AgentStateDict
from tools.planning_tool import PlanningTool

logger = logging.getLogger(__name__)


async def planning_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute planning task.

    Args:
        state: Current agent state

    Returns:
        Updated state with generated plan
    """
    logger.info("=== Planning Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    if not action_details:
        error_msg = "Planning executor requires action_details with task description"
        logger.error(error_msg)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
        }

    try:
        # Parse detail level if provided
        # Format: "task|detail_level" or just "task"
        parts = action_details.split("|")
        task = parts[0].strip()
        detail_level = parts[1].strip() if len(parts) > 1 else "medium"

        # Initialize planning tool
        planning_tool = PlanningTool()

        # Generate plan
        logger.info(f"Generating plan for: {task[:100]} (detail: {detail_level})")
        result = await planning_tool._arun(task=task, detail_level=detail_level)

        logger.info(f"Plan generated successfully ({len(result)} chars)")

        return {
            "last_tool_output": result,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }

    except Exception as e:
        error_msg = f"Planning execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }
