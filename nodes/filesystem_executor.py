"""
Filesystem Executor Node - Deep Agents FilesystemMiddleware integration.

LangGraph node that executes filesystem operations via Deep Agents
FilesystemMiddleware tools: ls, read_file, write_file, edit_file, glob, grep.

Provides a unified interface for filesystem operations that integrates
with the existing Manus architecture while leveraging Deep Agents capabilities.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agent_state import AgentStateDict

logger = logging.getLogger(__name__)


def filesystem_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute filesystem operations via Deep Agents FilesystemMiddleware.

    Handles the following operations:
    - ls: List directory contents
    - read_file: Read file content
    - write_file: Write new file
    - edit_file: Edit existing file
    - glob: Pattern matching for files
    - grep: Search within files

    Args:
        state: Current agent state with tool_name and tool_params.

    Returns:
        Updated state with filesystem operation result.

    Example:
        >>> state = AgentStateDict(
        ...     tool_name="write_file",
        ...     tool_params={"path": "test.txt", "content": "Hello World"}
        ... )
        >>> result = filesystem_executor_node(state)
        >>> print(result["last_tool_output"])
    """
    logger.info("=== Filesystem Executor Node ===")

    # Extract action details
    action = state.get("current_action", "") or state.get("tool_name", "")
    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    # Parse action details
    params = _parse_action_details(action_details)

    # Determine the specific filesystem operation
    operation = _extract_operation(action, params)

    output_msg = ""
    status = "success"

    try:
        # Try to use Deep Agents middleware first
        result = _execute_via_deepagents(operation, params)
        if result is not None:
            output_msg = result
        else:
            # Fallback to native implementation
            result = _execute_native(operation, params)
            output_msg = result

    except Exception as e:
        logger.error(f"Filesystem operation '{operation}' failed: {e}", exc_info=True)
        output_msg = f"Filesystem error: {str(e)}"
        status = "failed"

    # Update filesystem history
    history_entry = {
        "operation": operation,
        "params": params,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }

    filesystem_history = state.get("filesystem_history", []) or []
    filesystem_history = filesystem_history.copy()
    filesystem_history.append(history_entry)

    return {
        "last_tool_output": output_msg,
        "iteration_count": iteration + 1,
        "filesystem_history": filesystem_history,
        "executor_outputs": [
            {
                "source": "filesystem_executor",
                "status": status,
                "output": output_msg[:1000] if len(output_msg) > 1000 else output_msg,
                "timestamp": datetime.now().isoformat(),
            }
        ],
    }


def _parse_action_details(action_details: Any) -> Dict[str, Any]:
    """Parse action details into a params dict."""
    if isinstance(action_details, dict):
        return action_details

    if isinstance(action_details, str):
        # Try JSON parsing
        try:
            return json.loads(action_details)
        except json.JSONDecodeError:
            pass

        # Try simple "operation path" format
        parts = action_details.strip().split(maxsplit=1)
        if len(parts) >= 1:
            result = {"operation": parts[0]}
            if len(parts) > 1:
                result["path"] = parts[1]
            return result

    return {}


def _extract_operation(action: str, params: Dict[str, Any]) -> str:
    """Extract the filesystem operation from action and params."""
    # Check explicit operation in params
    if "operation" in params:
        return params["operation"].lower()

    # Check action name
    action_lower = action.lower()
    filesystem_ops = ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]

    for op in filesystem_ops:
        if op in action_lower or action_lower.startswith(op):
            return op

    # Check for common aliases
    if "list" in action_lower or "dir" in action_lower:
        return "ls"
    elif "read" in action_lower or "cat" in action_lower:
        return "read_file"
    elif "write" in action_lower or "create" in action_lower:
        return "write_file"
    elif "edit" in action_lower or "modify" in action_lower:
        return "edit_file"
    elif "find" in action_lower or "search" in action_lower:
        if "content" in action_lower:
            return "grep"
        return "glob"

    # Default to ls if path looks like directory
    path = params.get("path", "")
    if path.endswith("/") or path == "." or not path:
        return "ls"

    return "read_file"


def _execute_via_deepagents(operation: str, params: Dict[str, Any]) -> Optional[str]:
    """
    Execute operation via Deep Agents FilesystemMiddleware.

    Returns None if Deep Agents is not available.
    """
    try:
        from middleware.deepagents_setup import get_deepagents_config

        config = get_deepagents_config()
        fs_middleware = config.get_filesystem_middleware()

        if fs_middleware is None:
            return None

        # Get the appropriate tool
        tools = config.get_tools()
        tool = None
        for t in tools:
            if t.name == operation:
                tool = t
                break

        if tool is None:
            return None

        # Execute the tool
        logger.debug(f"Executing via Deep Agents: {operation}")
        result = tool.invoke(params)
        return str(result)

    except ImportError:
        logger.debug("Deep Agents not available, using native implementation")
        return None
    except Exception as e:
        logger.warning(f"Deep Agents execution failed: {e}, falling back to native")
        return None


def _execute_native(operation: str, params: Dict[str, Any]) -> str:
    """
    Execute filesystem operation using native Python.

    Fallback when Deep Agents is not available.
    """
    workspace = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
    if not workspace.exists():
        workspace = Path.cwd() / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    path = params.get("path", ".")

    # Resolve path relative to workspace
    if not path.startswith("/"):
        full_path = workspace / path
    else:
        full_path = Path(path)

    # Security check: ensure path is within workspace
    try:
        resolved = full_path.resolve()
        workspace_resolved = workspace.resolve()
        if not str(resolved).startswith(str(workspace_resolved)):
            return f"Error: Path escapes workspace: {path}"
    except Exception:
        pass

    if operation == "ls":
        return _native_ls(full_path)
    elif operation == "read_file":
        return _native_read_file(full_path, params)
    elif operation == "write_file":
        return _native_write_file(full_path, params)
    elif operation == "edit_file":
        return _native_edit_file(full_path, params)
    elif operation == "glob":
        return _native_glob(workspace, params)
    elif operation == "grep":
        return _native_grep(workspace, params)
    else:
        return f"Unknown filesystem operation: {operation}"


def _native_ls(path: Path) -> str:
    """List directory contents."""
    if not path.exists():
        return f"Error: Path not found: {path}"

    if path.is_file():
        stat = path.stat()
        return f"File: {path.name} ({stat.st_size} bytes)"

    items = []
    for item in sorted(path.iterdir()):
        item_type = "DIR" if item.is_dir() else "FILE"
        size = f" ({item.stat().st_size}b)" if item.is_file() else ""
        items.append(f"[{item_type}] {item.name}{size}")

    if not items:
        return f"Directory is empty: {path}"

    return f"Contents of {path}:\n" + "\n".join(items)


def _native_read_file(path: Path, params: Dict[str, Any]) -> str:
    """Read file content."""
    if not path.exists():
        return f"Error: File not found: {path}"

    if not path.is_file():
        return f"Error: Not a file: {path}"

    try:
        content = path.read_text(encoding="utf-8")

        # Handle line range if specified
        start_line = params.get("start_line")
        end_line = params.get("end_line")

        if start_line is not None or end_line is not None:
            lines = content.splitlines()
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            content = "\n".join(lines[start:end])

        return f"Content of {path.name}:\n\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"


def _native_write_file(path: Path, params: Dict[str, Any]) -> str:
    """Write content to file."""
    content = params.get("content", "")
    if not content:
        return "Error: No content provided for write_file"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path.name}"
    except Exception as e:
        return f"Error writing file: {e}"


def _native_edit_file(path: Path, params: Dict[str, Any]) -> str:
    """Edit existing file."""
    if not path.exists():
        return f"Error: File not found: {path}"

    try:
        content = path.read_text(encoding="utf-8")

        # Handle edits
        edits = params.get("edits", [])
        if not edits:
            # Simple search/replace
            old_text = params.get("old_text", params.get("search", ""))
            new_text = params.get("new_text", params.get("replace", ""))

            if old_text:
                if old_text in content:
                    content = content.replace(old_text, new_text, 1)
                else:
                    return f"Error: Text not found in file: {old_text[:50]}..."
        else:
            # Apply structured edits
            for edit in edits:
                old = edit.get("old", "")
                new = edit.get("new", "")
                if old and old in content:
                    content = content.replace(old, new, 1)

        path.write_text(content, encoding="utf-8")
        return f"Successfully edited {path.name}"
    except Exception as e:
        return f"Error editing file: {e}"


def _native_glob(workspace: Path, params: Dict[str, Any]) -> str:
    """Find files matching pattern."""
    pattern = params.get("pattern", "*")

    try:
        matches = list(workspace.glob(pattern))
        if not matches:
            return f"No files matching pattern: {pattern}"

        results = [str(m.relative_to(workspace)) for m in matches[:50]]
        if len(matches) > 50:
            results.append(f"... and {len(matches) - 50} more")

        return f"Files matching '{pattern}':\n" + "\n".join(results)
    except Exception as e:
        return f"Error in glob: {e}"


def _native_grep(workspace: Path, params: Dict[str, Any]) -> str:
    """Search for pattern in files."""
    pattern = params.get("pattern", params.get("query", ""))
    if not pattern:
        return "Error: No search pattern provided"

    file_pattern = params.get("file_pattern", "**/*")
    max_results = 20

    try:
        results = []
        for file_path in workspace.glob(file_pattern):
            if file_path.is_file() and file_path.suffix in [
                ".txt",
                ".py",
                ".md",
                ".json",
                ".yaml",
                ".yml",
                ".sh",
                ".html",
                ".css",
                ".js",
            ]:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern.lower() in line.lower():
                            rel_path = file_path.relative_to(workspace)
                            results.append(f"{rel_path}:{i}: {line[:100]}")
                            if len(results) >= max_results:
                                break
                except Exception:
                    pass
            if len(results) >= max_results:
                break

        if not results:
            return f"No matches for pattern: {pattern}"

        return f"Matches for '{pattern}':\n" + "\n".join(results)
    except Exception as e:
        return f"Error in grep: {e}"


# Async wrapper for compatibility
async def filesystem_executor_node_async(state: AgentStateDict) -> Dict[str, Any]:
    """Async version of filesystem_executor_node."""
    return filesystem_executor_node(state)
