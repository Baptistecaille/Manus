"""
Deep Agents Middleware Configuration.

Centralized configuration for Deep Agents middlewares:
- FilesystemMiddleware: ls, read_file, write_file, edit_file, glob, grep
- MemoryMiddleware: For context management (replaces TodoListMiddleware)
- SubAgentMiddleware: task for spawning sub-agents

This module provides tools that can be injected into the existing
Manus LangGraph architecture without replacing the custom StateGraph.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeepAgentsConfig:
    """
    Configuration hub for Deep Agents middleware components.

    Provides centralized setup for FilesystemMiddleware, TodoListMiddleware,
    and SubAgentMiddleware. Tools from these middlewares can be extracted
    and used within the existing Manus LangGraph nodes.

    Attributes:
        workspace: Path to the workspace directory.
        middlewares: Dict of configured middleware instances.

    Example:
        >>> config = DeepAgentsConfig("workspace")
        >>> tools = config.get_tools()
        >>> print([t.name for t in tools])
        ['ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'write_todos', 'task']
    """

    def __init__(self, workspace_path: Optional[str] = None):
        """
        Initialize Deep Agents configuration.

        Args:
            workspace_path: Path to workspace directory. If None, uses
                           WORKSPACE_DIR env var or defaults to ./workspace.
        """
        if workspace_path:
            self.workspace = Path(workspace_path)
        else:
            env_path = os.getenv("WORKSPACE_DIR")
            if env_path:
                self.workspace = Path(env_path)
            elif Path("/workspace").exists():
                self.workspace = Path("/workspace")
            else:
                self.workspace = Path.cwd() / "workspace"

        # Ensure workspace exists
        self.workspace.mkdir(parents=True, exist_ok=True)

        self._middlewares: Dict[str, Any] = {}
        self._tools: List[Any] = []
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize middlewares on first access."""
        if self._initialized:
            return

        try:
            self._setup_middlewares()
            self._initialized = True
            logger.info(
                f"DeepAgentsConfig initialized with workspace: {self.workspace}"
            )
        except ImportError as e:
            logger.warning(f"Deep Agents package not installed: {e}")
            logger.info("Install with: uv add deepagents")
            self._initialized = True  # Mark as initialized to avoid repeated attempts

    def _setup_middlewares(self) -> None:
        """
        Configure all Deep Agents middlewares.

        Sets up:
        - FilesystemMiddleware with local backend
        - MemoryMiddleware for context management
        - SubAgentMiddleware for task delegation
        """
        try:
            from deepagents import (
                FilesystemMiddleware,
                SubAgentMiddleware,
                MemoryMiddleware,
            )
            from deepagents.backends import FilesystemBackend

            # Configure FilesystemMiddleware with local backend
            fs_backend = FilesystemBackend(root_dir=str(self.workspace))
            self._middlewares["filesystem"] = FilesystemMiddleware(
                backend=fs_backend,
            )
            logger.debug("FilesystemMiddleware configured")

            # Configure MemoryMiddleware (replaces TodoListMiddleware)
            # Using default backend - MemoryMiddleware handles its own storage
            try:
                self._middlewares["memory"] = MemoryMiddleware(
                    backend=fs_backend,
                    sources=[],
                )
                logger.debug("MemoryMiddleware configured")
            except Exception as mem_err:
                logger.warning(f"MemoryMiddleware setup failed: {mem_err}")

            # Configure SubAgentMiddleware (requires LLM API key)
            try:
                self._middlewares["subagent"] = SubAgentMiddleware(
                    default_model=os.getenv("LLM_MODEL", "gpt-4o"),
                )
                logger.debug("SubAgentMiddleware configured")
            except Exception as subagent_err:
                logger.warning(
                    f"SubAgentMiddleware setup failed (API key may be missing): {subagent_err}"
                )

            # Collect tools from all middlewares
            self._collect_tools()

        except ImportError as e:
            logger.warning(f"Could not import Deep Agents modules: {e}")
            raise

    def _collect_tools(self) -> None:
        """Collect tools from all configured middlewares."""
        self._tools = []

        for name, middleware in self._middlewares.items():
            if hasattr(middleware, "get_tools"):
                tools = middleware.get_tools()
                self._tools.extend(tools)
                logger.debug(f"Collected {len(tools)} tools from {name}")
            elif hasattr(middleware, "tools"):
                self._tools.extend(middleware.tools)
                logger.debug(f"Collected tools from {name}.tools")

    @property
    def middlewares(self) -> Dict[str, Any]:
        """Get configured middlewares (lazy initialization)."""
        self._lazy_init()
        return self._middlewares

    def get_tools(self) -> List[Any]:
        """
        Get all tools from configured middlewares.

        Returns:
            List of LangChain tool instances.
        """
        self._lazy_init()
        return self._tools

    def get_filesystem_middleware(self) -> Optional[Any]:
        """Get the FilesystemMiddleware instance."""
        self._lazy_init()
        return self._middlewares.get("filesystem")

    def get_memory_middleware(self) -> Optional[Any]:
        """Get the MemoryMiddleware instance."""
        self._lazy_init()
        return self._middlewares.get("memory")

    def get_subagent_middleware(self) -> Optional[Any]:
        """Get the SubAgentMiddleware instance."""
        self._lazy_init()
        return self._middlewares.get("subagent")

    def get_tool_names(self) -> List[str]:
        """
        Get names of all available tools.

        Returns:
            List of tool name strings.
        """
        return [tool.name for tool in self.get_tools()]


# Singleton instance for convenience
_config_instance: Optional[DeepAgentsConfig] = None


def get_deepagents_config(workspace_path: Optional[str] = None) -> DeepAgentsConfig:
    """
    Get or create the global DeepAgentsConfig instance.

    Args:
        workspace_path: Optional workspace path (only used on first call).

    Returns:
        DeepAgentsConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = DeepAgentsConfig(workspace_path)
    return _config_instance


# Tool name sets for router detection
FILESYSTEM_TOOLS = {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}
MEMORY_TOOLS = {"memory_store", "memory_retrieve"}  # Updated from TodoList
SUBAGENT_TOOLS = {"task"}
ALL_DEEPAGENTS_TOOLS = FILESYSTEM_TOOLS | MEMORY_TOOLS | SUBAGENT_TOOLS


def is_deepagents_tool(tool_name: str) -> bool:
    """Check if a tool name is a Deep Agents tool."""
    return tool_name.lower() in ALL_DEEPAGENTS_TOOLS


def get_tool_category(tool_name: str) -> Optional[str]:
    """
    Get the category of a Deep Agents tool.

    Args:
        tool_name: Name of the tool.

    Returns:
        Category string ('filesystem', 'memory', 'subagent') or None.
    """
    name = tool_name.lower()
    if name in FILESYSTEM_TOOLS:
        return "filesystem"
    elif name in MEMORY_TOOLS:
        return "memory"
    elif name in SUBAGENT_TOOLS:
        return "subagent"
    return None
