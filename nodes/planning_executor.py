"""
Planning Executor Node for LangGraph.

Generates detailed action plans using PlanningTool.
"""

import logging
from typing import Any, Dict
import asyncio
import os

from agent_state import AgentStateDict
from tools.planning_tool import PlanningTool
from nodes.planning_manager import PlanningManager

logger = logging.getLogger(__name__)


def planning_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute planning task and persist changes.

    Args:
        state: Current agent state

    Returns:
        Updated state with generated plan
    """
    logger.info("=== Planning Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)
    workspace = state.get("workspace_dir")

    if not action_details:
        error_msg = "Planning executor requires action_details with task description"
        logger.error(error_msg)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
        }

    try:
        # Parse detail level if provided
        parts = action_details.split("|")
        task = parts[0].strip()
        detail_level = parts[1].strip() if len(parts) > 1 else "medium"

        # Initialize tools
        planning_tool = PlanningTool()
        # Initialize manager with workspace from state/env
        manager = PlanningManager(workspace_dir=workspace)

        # Generate plan content
        logger.info(f"Generating plan for: {task[:100]} (detail: {detail_level})")

        # Run sync tool
        # In a real async environment, we'd want this to be async too
        pk_plan_text = planning_tool._run(task=task, detail_level=detail_level)

        # Persist the plan using PlanningManager
        # We need to run this async function synchronously here
        async def _persist_plan():
            # Initialize (overwrite) the plan with the new goal
            # We assume 'task' is the new goal
            result = await manager.initialize_plan(goal=task)
            # Force a refresh to update the state with structured data
            plan_data = await manager.refresh_plan()
            return result, plan_data

        loop = asyncio.get_event_loop()

        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                init_result, plan_data = executor.submit(
                    asyncio.run, _persist_plan()
                ).result()
        else:
            init_result, plan_data = loop.run_until_complete(_persist_plan())

        msg = f"Plan generated and persisted to {init_result['plan_path']}"
        logger.info(msg)

        return {
            "last_tool_output": f"PLANNING PHASE COMPLETE. Plan persisted to {init_result['plan_path']}. You typically should now EXECUTE the first step of this plan.",
            "plan_file_path": init_result["plan_path"],
            "current_phase": plan_data["current_phase"],
            "messages": [
                {
                    "role": "system",
                    "content": f"[PLAN UPDATED] New plan initialized for: {task}",
                }
            ],
            "actions_since_refresh": 0,  # Reset counter so we don't immediately refresh
            "iteration_count": iteration + 1,
            "current_action": "plan",
        }

    except Exception as e:
        error_msg = f"Planning execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }
