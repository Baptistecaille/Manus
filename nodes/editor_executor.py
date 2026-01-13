"""
Editor Executor Node for LangGraph.

Executes file editing operations using StrReplaceEditorTool.
Supports multiple input formats from LLM output.
"""

import logging
import re
from typing import Any, Dict
import json

from agent_state import AgentStateDict
import os
from agent_state import AgentStateDict
from tools.str_replace_editor import StrReplaceEditorTool

logger = logging.getLogger(__name__)


def _parse_action_details(action_details: str) -> Dict[str, Any]:
    """
    Parse action_details from various LLM output formats.

    Supported formats:
    1. JSON: {"command": "create", "path": "/workspace/file.py", "file_text": "..."}
    2. Pipe format: "command|path"
    3. YAML-like: "file_path: /workspace/file.py\ncontent: |..."

    Returns:
        Dict with 'command', 'path', and optional 'file_text' keys.
    """
    action_details = action_details.strip()

    # Try JSON first
    try:
        params = json.loads(action_details)
        if isinstance(params, dict):
            return params
    except json.JSONDecodeError:
        pass

    # Try YAML-like format: file_path: ... content: ...
    file_path_match = re.search(r"file_path:\s*([^\n]+)", action_details, re.IGNORECASE)
    content_match = re.search(
        r"content:\s*\|?\s*\n?([\s\S]*)", action_details, re.IGNORECASE
    )

    if file_path_match:
        file_path = file_path_match.group(1).strip()
        content = ""
        if content_match:
            content = content_match.group(1).strip()
            # Remove leading indentation from content (common in YAML)
            lines = content.split("\n")
            if lines:
                # Find minimum indentation
                min_indent = float("inf")
                for line in lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                if min_indent < float("inf") and min_indent > 0:
                    content = "\n".join(
                        line[min_indent:] if len(line) >= min_indent else line
                        for line in lines
                    )

        return {"command": "create", "path": file_path, "file_text": content}

    # Try pipe format: command|path
    if "|" in action_details:
        parts = action_details.split("|", 1)
        return {
            "command": parts[0].strip(),
            "path": parts[1].strip() if len(parts) > 1 else "/workspace/file.txt",
        }

    # Fallback: assume it's a path for viewing
    if action_details.startswith("/"):
        return {
            "command": "view",
            "path": action_details.split()[0],  # Take first word as path
        }

    # Last resort: return as-is
    return {
        "command": "view",
        "path": "/workspace",
        "error": f"Could not parse action_details: {action_details[:100]}",
    }


def editor_executor_node(state: AgentStateDict) -> Dict[str, Any]:
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
        # Parse action_details using the improved parser
        params = _parse_action_details(action_details)

        if "error" in params:
            logger.warning(f"Parse warning: {params['error']}")

        # Initialize editor tool with correct local path
        workspace_path = os.path.join(os.getcwd(), "workspace")
        editor_tool = StrReplaceEditorTool(workspace_root=workspace_path)

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
