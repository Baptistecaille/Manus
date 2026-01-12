"""
Editor Executor Node for LangGraph.

Executes file editing operations using StrReplaceEditorTool.
"""

import logging
from typing import Any, Dict
import json

from agent_state import AgentStateDict
from tools.str_replace_editor import StrReplaceEditorTool

logger = logging.getLogger(__name__)


async def editor_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute file editing task.

    Args:
        state: Current agent state

    Returns:
        Updated state with edit results
    """
    logger.info("=== Editor Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    if not action_details:
        error_msg = (
            "Editor executor requires action_details with command and parameters"
        )
        logger.error(error_msg)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
        }

    try:
        # Parse action_details as JSON
        # Expected format: {"command": "view|create|str_replace|insert|undo_edit", "path": "...", ...}
        try:
            params = json.loads(action_details)
        except json.JSONDecodeError:
            # Fallback: treat as simple "command|path" format
            parts = action_details.split("|", 1)
            params = {
                "command": parts[0].strip(),
                "path": parts[1].strip() if len(parts) > 1 else "/workspace/file.txt",
            }

        # Initialize editor tool
        editor_tool = StrReplaceEditorTool()

        # Execute editor command
        command = params.get("command", "view")
        logger.info(
            f"Executing editor command: {command} on {params.get('path', 'N/A')}"
        )

        result = editor_tool._run(**params)

        logger.info(f"Editor result (first 200 chars): {result[:200]}...")

        return {
            "last_tool_output": result,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }

    except Exception as e:
        error_msg = f"Editor execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }
