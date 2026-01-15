"""
Eviction Handler for Large Results.

Automatically detects and evicts large tool results (>5000 tokens)
to workspace files to save LLM context window space. Replaces
in-context results with file references and summaries.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default token threshold for eviction
DEFAULT_TOKEN_THRESHOLD = 5000

# Approximate characters per token (conservative estimate)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses a simple character-based approximation.
    For more accurate counting, use tiktoken.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def count_tokens_accurate(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens accurately using tiktoken.

    Falls back to estimation if tiktoken is not available.

    Args:
        text: Text to count tokens for.
        model: Model name for encoding selection.

    Returns:
        Token count.
    """
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        logger.debug("tiktoken not available, using estimation")
        return estimate_tokens(text)
    except Exception as e:
        logger.warning(f"tiktoken error: {e}, using estimation")
        return estimate_tokens(text)


class EvictionHandler:
    """
    Handler for evicting large results to filesystem.

    When tool results exceed the token threshold, they are:
    1. Written to a file in the workspace
    2. Replaced with a reference dict containing file path and summary

    Attributes:
        workspace: Path to workspace for storing evicted files.
        threshold_tokens: Token count above which to evict.
        eviction_dir: Subdirectory for evicted files.

    Example:
        >>> handler = EvictionHandler("workspace", threshold_tokens=5000)
        >>> result = "x" * 50000  # ~12500 tokens
        >>> evicted = handler.maybe_evict(result, "search_result_0")
        >>> print(evicted["type"])  # "evicted"
        >>> print(evicted["file"])  # "workspace/.evicted/search_result_0_abc123.txt"
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        threshold_tokens: int = DEFAULT_TOKEN_THRESHOLD,
        use_accurate_counting: bool = False,
    ):
        """
        Initialize the eviction handler.

        Args:
            workspace_path: Path to workspace directory.
            threshold_tokens: Token threshold for eviction.
            use_accurate_counting: If True, use tiktoken for counting.
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

        self.threshold_tokens = threshold_tokens
        self.eviction_dir = self.workspace / ".evicted"
        self.use_accurate_counting = use_accurate_counting

        # Ensure eviction directory exists
        self.eviction_dir.mkdir(parents=True, exist_ok=True)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using configured method."""
        if self.use_accurate_counting:
            return count_tokens_accurate(text)
        return estimate_tokens(text)

    def _generate_filename(self, identifier: str, content: str) -> str:
        """Generate a unique filename for evicted content."""
        # Use hash of content for uniqueness
        content_hash = hashlib.md5(content[:1000].encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%H%M%S")
        safe_id = identifier.replace("/", "_").replace("\\", "_")[:50]
        return f"{safe_id}_{timestamp}_{content_hash}.txt"

    def should_evict(self, content: Any) -> bool:
        """
        Check if content should be evicted.

        Args:
            content: Content to check (will be stringified).

        Returns:
            True if content exceeds token threshold.
        """
        text = str(content)
        tokens = self._count_tokens(text)
        return tokens > self.threshold_tokens

    def evict(self, content: Any, identifier: str) -> Dict[str, Any]:
        """
        Evict content to a file.

        Args:
            content: Content to evict.
            identifier: Identifier for the content (used in filename).

        Returns:
            Dict with eviction metadata:
            - type: "evicted"
            - file: Path to evicted file
            - original_tokens: Token count of original
            - summary: First 200 chars of content
        """
        text = str(content)
        tokens = self._count_tokens(text)

        # Generate filename and write
        filename = self._generate_filename(identifier, text)
        file_path = self.eviction_dir / filename
        file_path.write_text(text, encoding="utf-8")

        logger.info(f"Evicted {tokens} tokens to {file_path}")

        return {
            "type": "evicted",
            "file": str(file_path),
            "original_tokens": tokens,
            "summary": text[:200] + ("..." if len(text) > 200 else ""),
            "timestamp": datetime.now().isoformat(),
        }

    def maybe_evict(self, content: Any, identifier: str) -> Any:
        """
        Evict content if it exceeds threshold, otherwise return as-is.

        Args:
            content: Content to potentially evict.
            identifier: Identifier for the content.

        Returns:
            Original content or eviction reference dict.
        """
        if self.should_evict(content):
            return self.evict(content, identifier)
        return content

    def process_results(self, results: List[Any]) -> List[Any]:
        """
        Process a list of results, evicting large ones.

        Args:
            results: List of tool results.

        Returns:
            List with large results replaced by eviction references.
        """
        processed = []
        evicted_count = 0

        for i, result in enumerate(results):
            processed_result = self.maybe_evict(result, f"result_{i}")
            if (
                isinstance(processed_result, dict)
                and processed_result.get("type") == "evicted"
            ):
                evicted_count += 1
            processed.append(processed_result)

        if evicted_count > 0:
            logger.info(f"Evicted {evicted_count}/{len(results)} results to files")

        return processed

    def read_evicted(self, eviction_ref: Dict[str, Any]) -> str:
        """
        Read content from an evicted file.

        Args:
            eviction_ref: Eviction reference dict with 'file' key.

        Returns:
            File content.

        Raises:
            FileNotFoundError: If evicted file doesn't exist.
        """
        if not isinstance(eviction_ref, dict) or eviction_ref.get("type") != "evicted":
            raise ValueError("Not an eviction reference")

        file_path = Path(eviction_ref["file"])
        if not file_path.exists():
            raise FileNotFoundError(f"Evicted file not found: {file_path}")

        return file_path.read_text(encoding="utf-8")

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old evicted files.

        Args:
            max_age_hours: Maximum age in hours before deletion.

        Returns:
            Number of files deleted.
        """
        import time

        deleted = 0
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()

        for file_path in self.eviction_dir.glob("*.txt"):
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                deleted += 1

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old evicted files")

        return deleted


# Global handler instance
_eviction_handler: Optional[EvictionHandler] = None


def get_eviction_handler(workspace_path: Optional[str] = None) -> EvictionHandler:
    """Get or create the global EvictionHandler instance."""
    global _eviction_handler
    if _eviction_handler is None:
        _eviction_handler = EvictionHandler(workspace_path)
    return _eviction_handler


def eviction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for evicting large results.

    Processes tool_results in state and evicts large ones to files.

    Args:
        state: Agent state with tool_results key.

    Returns:
        Updated state with processed tool_results.
    """
    handler = get_eviction_handler()

    tool_results = state.get("tool_results", [])
    if not tool_results:
        return {}

    processed = handler.process_results(tool_results)

    # Track evicted results
    evicted = [
        r for r in processed if isinstance(r, dict) and r.get("type") == "evicted"
    ]

    return {
        "tool_results": processed,
        "evicted_results": evicted,
    }
