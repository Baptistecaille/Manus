"""
File Manager Executor Node.

Wraps the FileManagerSkill to allow the agent to perform advanced file operations
such as organization, compression, and conversion.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from agent_state import AgentStateDict
from skills.file_manager import FileManagerSkill

logger = logging.getLogger(__name__)


def file_manager_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute file management tasks.

    Parses action_details to determine the specific operation:
    - organize: Organize files by type
    - compress: Create zip/tar archives
    - extract: Extract archives
    - convert: Convert file formats (csv <-> json)
    - list: List files with details

    Args:
        state: Current agent state.

    Returns:
        Updated state with execution results.
    """
    logger.info("=== File Manager Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)
    current_action = state.get("current_action", "file_manager")

    # Initialize skill
    skill = FileManagerSkill()

    output_msg = ""
    status = "success"

    try:
        # Parse action details
        params = {}
        if isinstance(action_details, str):
            try:
                params = json.loads(action_details)
            except json.JSONDecodeError:
                # If not JSON, treat as a simple command if possible, or fail gracefully
                # For now, we expect the planner to provide JSON for complex tools
                # But we can support simple "organize /path" syntax
                parts = action_details.split(maxsplit=1)
                if len(parts) >= 1:
                    params["command"] = parts[0]
                    if len(parts) > 1:
                        params["path"] = parts[1]
        elif isinstance(action_details, dict):
            params = action_details

        command = params.get("command", "").lower()

        # Execute based on command
        if command == "organize":
            directory = params.get("path") or params.get("directory", ".")
            result = await_sync(skill.organize_by_type(directory))
            output_msg = (
                f"Organized files in {directory}: {json.dumps(result, indent=2)}"
            )

        elif command == "compress":
            files = params.get("files", [])
            output_path = params.get("output_path", "archive.zip")
            # If files is a string, maybe it's a directory or single file
            if isinstance(files, str):
                files = [files]

            # If no output path, generate one
            if not output_path and files:
                output_path = f"{files[0]}_archive.zip"

            result = await_sync(skill.compress_files(files, output_path))
            output_msg = f"Compressed files to {result}"

        elif command == "extract":
            archive_path = params.get("path") or params.get("archive_path")
            destination = params.get("destination", ".")
            result = await_sync(skill.extract_archive(archive_path, destination))
            output_msg = f"Extracted to {destination}: {len(result)} files"

        elif command == "convert":
            file_path = params.get("path") or params.get("file_path")
            target_format = params.get("target_format", "json")
            result = await_sync(skill.convert_format(file_path, target_format))
            output_msg = f"Converted {file_path} to {result}"

        elif command == "list":
            directory = params.get("path") or params.get("directory", ".")
            result = await_sync(skill.list_files(directory))
            # Summarize for context if too long
            summary = [f"{f['name']} ({f['size_bytes']}b)" for f in result[:20]]
            if len(result) > 20:
                summary.append(f"... and {len(result)-20} more")
            output_msg = f"Files in {directory}:\n" + "\n".join(summary)

        else:
            # Default or fallback
            output_msg = f"Unknown or missing file manager command: {command}"
            status = "failed"

    except Exception as e:
        logger.error(f"File manager execution failed: {e}", exc_info=True)
        output_msg = f"File manager error: {str(e)}"
        status = "failed"

    return {
        "last_tool_output": output_msg,
        "iteration_count": iteration + 1,
        "executor_outputs": [
            {
                "source": "file_manager_executor",
                "status": status,
                "output": output_msg,
                "timestamp": datetime.now().isoformat(),
            }
        ],
    }


def await_sync(awaitable):
    """Helper to run async code synchronously."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(asyncio.run, awaitable).result()
    else:
        return loop.run_until_complete(awaitable)
