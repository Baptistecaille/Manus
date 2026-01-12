"""
String Replace Editor Tool for LangChain integration.

Advanced file editing tool with view, create, str_replace, insert, and undo operations.
Adapted from OpenManus implementation for LangChain/LangGraph architecture.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Literal, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Constants
SNIPPET_LINES = 4
MAX_RESPONSE_LEN = 16000
TRUNCATED_MESSAGE = (
    "<response clipped><NOTE>To save on context only part of this file has been shown. "
    "Retry after using grep -n to find line numbers.</NOTE>"
)


class StrReplaceEditorInput(BaseModel):
    """Input schema for str_replace_editor."""

    command: Literal["view", "create", "str_replace", "insert", "undo_edit"] = Field(
        description="Command: view, create, str_replace, insert, or undo_edit"
    )
    path: str = Field(description="Absolute path to file or directory")
    file_text: Optional[str] = Field(
        default=None, description="Content for create command"
    )
    old_str: Optional[str] = Field(
        default=None, description="String to replace (str_replace command)"
    )
    new_str: Optional[str] = Field(
        default=None, description="Replacement string (str_replace/insert commands)"
    )
    insert_line: Optional[int] = Field(
        default=None, description="Line number to insert after (insert command)"
    )
    view_range: Optional[List[int]] = Field(
        default=None,
        description="Line range for view [start, end], e.g. [10, 20] or [10, -1]",
    )


class StrReplaceEditorTool(BaseTool):
    """
    Advanced file editing tool with multiple commands.

    Commands:
    - view: Display file or directory contents
    - create: Create a new file
    - str_replace: Replace unique string in file
    - insert: Insert text at specific line
    - undo_edit: Revert last edit
    """

    name: str = "str_replace_editor"
    description: str = (
        "Advanced file editing tool. Commands: view (display file/dir), "
        "create (new file), str_replace (replace unique string), "
        "insert (add text at line), undo_edit (revert last change). "
        "Use absolute paths from /workspace."
    )
    args_schema: type[BaseModel] = StrReplaceEditorInput

    workspace_root: str = "/workspace"
    file_history: DefaultDict[str, List[str]] = defaultdict(list)

    def _get_safe_path(self, path: str) -> Path:
        """Get safe path within workspace."""
        full_path = (
            Path(path)
            if path.startswith("/workspace")
            else Path(self.workspace_root) / path
        )
        resolved = full_path.resolve()
        workspace = Path(self.workspace_root).resolve()

        if not str(resolved).startswith(str(workspace)):
            raise ValueError(f"Path escapes workspace: {path}")

        return resolved

    def _maybe_truncate(self, content: str) -> str:
        """Truncate content if too long."""
        if len(content) <= MAX_RESPONSE_LEN:
            return content
        return content[:MAX_RESPONSE_LEN] + TRUNCATED_MESSAGE

    def _make_output(self, content: str, descriptor: str, init_line: int = 1) -> str:
        """Format content with line numbers."""
        content = self._maybe_truncate(content).expandtabs()
        numbered = "\n".join(
            [f"{i + init_line:6}\t{line}" for i, line in enumerate(content.split("\n"))]
        )
        return f"Here's the result of `cat -n` on {descriptor}:\n{numbered}\n"

    def _view_file(self, path: Path, view_range: Optional[List[int]]) -> str:
        """View file contents with optional line range."""
        content = path.read_text(encoding="utf-8")
        init_line = 1

        if view_range:
            if len(view_range) != 2:
                raise ValueError("view_range must be [start, end]")

            lines = content.split("\n")
            n_lines = len(lines)
            start, end = view_range

            if start < 1 or start > n_lines:
                raise ValueError(f"start line {start} out of range [1, {n_lines}]")
            if end != -1 and (end > n_lines or end < start):
                raise ValueError(f"end line {end} invalid")

            init_line = start
            if end == -1:
                content = "\n".join(lines[start - 1 :])
            else:
                content = "\n".join(lines[start - 1 : end])

        return self._make_output(content, str(path), init_line)

    def _view_directory(self, path: Path) -> str:
        """View directory contents."""
        items = []
        for item in sorted(path.glob("**/*")):
            if item.name.startswith("."):
                continue
            depth = len(item.relative_to(path).parts)
            if depth > 2:
                continue

            indent = "  " * (depth - 1)
            item_type = "DIR" if item.is_dir() else "FILE"
            items.append(f"{indent}[{item_type}] {item.name}")

        if not items:
            return f"Directory {path} is empty"

        return f"Contents of {path} (max depth 2):\n" + "\n".join(items)

    def _run(
        self,
        command: Literal["view", "create", "str_replace", "insert", "undo_edit"],
        path: str,
        file_text: Optional[str] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
        view_range: Optional[List[int]] = None,
    ) -> str:
        """Execute editor command."""
        try:
            safe_path = self._get_safe_path(path)

            # VIEW command
            if command == "view":
                if not safe_path.exists():
                    return f"Error: Path not found: {path}"

                if safe_path.is_dir():
                    if view_range:
                        return "Error: view_range not allowed for directories"
                    return self._view_directory(safe_path)
                else:
                    return self._view_file(safe_path, view_range)

            # CREATE command
            elif command == "create":
                if safe_path.exists():
                    return f"Error: File already exists: {path}"
                if file_text is None:
                    return "Error: file_text required for create command"

                safe_path.parent.mkdir(parents=True, exist_ok=True)
                safe_path.write_text(file_text, encoding="utf-8")
                self.file_history[str(safe_path)].append(file_text)
                return f"File created successfully at: {path}"

            # STR_REPLACE command
            elif command == "str_replace":
                if not safe_path.exists() or safe_path.is_dir():
                    return f"Error: Invalid file path: {path}"
                if old_str is None:
                    return "Error: old_str required for str_replace"

                content = safe_path.read_text(encoding="utf-8").expandtabs()
                old_str_exp = old_str.expandtabs()
                new_str_exp = (new_str or "").expandtabs()

                # Check uniqueness
                occurrences = content.count(old_str_exp)
                if occurrences == 0:
                    return f"Error: old_str not found in {path}"
                if occurrences > 1:
                    lines = [
                        idx + 1
                        for idx, line in enumerate(content.split("\n"))
                        if old_str_exp in line
                    ]
                    return f"Error: old_str appears {occurrences} times (lines {lines}). Must be unique."

                # Perform replacement
                self.file_history[str(safe_path)].append(content)
                new_content = content.replace(old_str_exp, new_str_exp)
                safe_path.write_text(new_content, encoding="utf-8")

                # Create snippet
                replacement_line = content.split(old_str_exp)[0].count("\n")
                start = max(0, replacement_line - SNIPPET_LINES)
                end = replacement_line + SNIPPET_LINES + new_str_exp.count("\n")
                snippet = "\n".join(new_content.split("\n")[start : end + 1])

                return (
                    f"File {path} edited successfully. "
                    + self._make_output(snippet, f"snippet of {path}", start + 1)
                    + "Review changes and edit again if needed."
                )

            # INSERT command
            elif command == "insert":
                if not safe_path.exists() or safe_path.is_dir():
                    return f"Error: Invalid file path: {path}"
                if insert_line is None or new_str is None:
                    return "Error: insert_line and new_str required for insert"

                content = safe_path.read_text(encoding="utf-8").expandtabs()
                new_str_exp = new_str.expandtabs()
                lines = content.split("\n")
                n_lines = len(lines)

                if insert_line < 0 or insert_line > n_lines:
                    return (
                        f"Error: insert_line {insert_line} out of range [0, {n_lines}]"
                    )

                # Perform insertion
                self.file_history[str(safe_path)].append(content)
                new_lines = new_str_exp.split("\n")
                result_lines = lines[:insert_line] + new_lines + lines[insert_line:]

                new_content = "\n".join(result_lines)
                safe_path.write_text(new_content, encoding="utf-8")

                # Create snippet
                snippet_lines = (
                    lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
                    + new_lines
                    + lines[insert_line : insert_line + SNIPPET_LINES]
                )
                snippet = "\n".join(snippet_lines)

                return (
                    f"File {path} edited successfully. "
                    + self._make_output(
                        snippet, "snippet", max(1, insert_line - SNIPPET_LINES + 1)
                    )
                    + "Review indentation and check for duplicates."
                )

            # UNDO_EDIT command
            elif command == "undo_edit":
                if (
                    str(safe_path) not in self.file_history
                    or not self.file_history[str(safe_path)]
                ):
                    return f"Error: No edit history for {path}"

                old_content = self.file_history[str(safe_path)].pop()
                safe_path.write_text(old_content, encoding="utf-8")
                return f"Last edit to {path} undone successfully."

            else:
                return f"Error: Unknown command: {command}"

        except ValueError as e:
            return f"Security/Validation error: {str(e)}"
        except Exception as e:
            logger.error(f"Editor error: {e}")
            return f"Error: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        """Async version - calls sync version."""
        return self._run(**kwargs)
