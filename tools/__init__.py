"""
Tools package for New_manus agent.

This package contains LangChain-based tools for various operations including
browser automation, web search, crawling, file operations, and code editing.
"""

from tools.file_operators import FileOperatorsTool
from tools.str_replace_editor import StrReplaceEditorTool

__all__ = [
    "FileOperatorsTool",
    "StrReplaceEditorTool",
]
