"""
File Operations Tool for LangChain integration.

Provides structured file operations: read, write, append, delete, and list.
"""

import logging
import os
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FileOperatorsInput(BaseModel):
    """Input schema for file operations."""

    operation: Literal["read", "write", "append", "delete", "list"] = Field(
        description="File operation to perform: read, write, append, delete, or list"
    )
    path: str = Field(description="File or directory path (relative to workspace)")
    content: Optional[str] = Field(
        default=None,
        description="Content to write/append (required for write and append operations)",
    )


class FileOperatorsTool(BaseTool):
    """
    File operations tool for workspace file management.

    Supports read, write, append, delete, and list operations.
    All paths are relative to the workspace directory for security.
    """

    name: str = "file_operators"
    description: str = (
        "Perform file operations in the workspace: read, write, append, delete files, "
        "or list directory contents. Use this to manage files safely within the workspace."
    )
    args_schema: type[BaseModel] = FileOperatorsInput

    workspace_root: str = "/workspace"  # Default workspace

    def _get_safe_path(self, path: str) -> Path:
        """
        Get safe absolute path within workspace.

        Args:
            path: Relative or absolute path

        Returns:
            Absolute Path object within workspace

        Raises:
            ValueError: If path escapes workspace
        """
        # Convert to Path object
        if path.startswith("/workspace"):
            full_path = Path(path)
        else:
            full_path = Path(self.workspace_root) / path

        # Resolve to absolute path and check it's within workspace
        resolved = full_path.resolve()
        workspace = Path(self.workspace_root).resolve()

        if not str(resolved).startswith(str(workspace)):
            raise ValueError(f"Path escapes workspace: {path}")

        return resolved

    def _run(
        self,
        operation: Literal["read", "write", "append", "delete", "list"],
        path: str,
        content: Optional[str] = None,
    ) -> str:
        """
        Execute file operation.

        Args:
            operation: Operation to perform
            path: File or directory path
            content: Content for write/append operations

        Returns:
            Result message or file content
        """
        try:
            safe_path = self._get_safe_path(path)

            if operation == "read":
                if not safe_path.exists():
                    return f"Error: File not found: {path}"
                if not safe_path.is_file():
                    return f"Error: Not a file: {path}"

                content = safe_path.read_text(encoding="utf-8")
                return f"File content of {path}:\n\n{content}"

            elif operation == "write":
                if content is None:
                    return "Error: content parameter required for write operation"

                # Create parent directories if needed
                safe_path.parent.mkdir(parents=True, exist_ok=True)

                safe_path.write_text(content, encoding="utf-8")
                return f"Successfully wrote {len(content)} characters to {path}"

            elif operation == "append":
                if content is None:
                    return "Error: content parameter required for append operation"

                # Create file if it doesn't exist
                if not safe_path.exists():
                    safe_path.parent.mkdir(parents=True, exist_ok=True)
                    safe_path.write_text(content, encoding="utf-8")
                    return f"Created new file and wrote {len(content)} characters to {path}"
                else:
                    with safe_path.open("a", encoding="utf-8") as f:
                        f.write(content)
                    return f"Successfully appended {len(content)} characters to {path}"

            elif operation == "delete":
                if not safe_path.exists():
                    return f"Error: File not found: {path}"

                if safe_path.is_file():
                    safe_path.unlink()
                    return f"Successfully deleted file: {path}"
                else:
                    return f"Error: Not a file: {path}. Use 'list' to see directory contents."

            elif operation == "list":
                if not safe_path.exists():
                    return f"Error: Directory not found: {path}"
                if not safe_path.is_dir():
                    return f"Error: Not a directory: {path}"

                items = []
                for item in sorted(safe_path.iterdir()):
                    item_type = "DIR" if item.is_dir() else "FILE"
                    size = f"({item.stat().st_size} bytes)" if item.is_file() else ""
                    items.append(f"  [{item_type}] {item.name} {size}")

                if not items:
                    return f"Directory is empty: {path}"

                return f"Contents of {path}:\n" + "\n".join(items)

            else:
                return f"Error: Unknown operation: {operation}"

        except ValueError as e:
            logger.error(f"Security error: {e}")
            return f"Security error: {str(e)}"
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return f"Error: {str(e)}"

    async def _arun(
        self,
        operation: Literal["read", "write", "append", "delete", "list"],
        path: str,
        content: Optional[str] = None,
    ) -> str:
        """Async version - calls sync version."""
        return self._run(operation, path, content)
