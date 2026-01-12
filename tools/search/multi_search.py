"""
Multi-Search Tool with automatic fallback.

Orchestrates multiple search engines with retry logic and automatic fallback
when a search engine fails or is rate-limited.
"""

import logging
import time
from typing import Optional, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.search.google_search import GoogleSearchTool
from tools.search.bing_search import BingSearchTool
from tools.search.baidu_search import BaiduSearchTool
from tools.search.duckduckgo_search import DuckDuckGoSearchTool

logger = logging.getLogger(__name__)


class MultiSearchInput(BaseModel):
    """Input schema for multi-search."""

    query: str = Field(description="Search query to execute")
    num_results: int = Field(
        default=5, description="Number of results to return", ge=1, le=20
    )
    preferred_engine: Optional[str] = Field(
        default="google",
        description="Preferred search engine: google, bing, baidu, or duckduckgo",
    )


class MultiSearchTool(BaseTool):
    """
    Multi-engine search tool with automatic fallback.

    Tries search engines in order of preference, automatically falling back
    to the next engine if one fails. Implements retry logic for rate limiting.
    """

    name: str = "multi_search"
    description: str = (
        "Search the web using multiple search engines with automatic fallback. "
        "Tries Google first, then falls back to DuckDuckGo, Bing, and Baidu if needed. "
        "Use this for reliable web search with redundancy."
    )
    args_schema: type[BaseModel] = MultiSearchInput

    # Search engine instances
    google: GoogleSearchTool = GoogleSearchTool()
    bing: BingSearchTool = BingSearchTool()
    baidu: BaiduSearchTool = BaiduSearchTool()
    duckduckgo: DuckDuckGoSearchTool = DuckDuckGoSearchTool()

    # Configuration
    retry_delay: int = 60  # Seconds to wait before retrying all engines
    max_retries: int = 3  # Maximum number of retry cycles

    def get_engine_order(self, preferred: str) -> List[BaseTool]:
        """
        Get search engines in fallback order based on preference.

        Args:
            preferred: Preferred search engine name

        Returns:
            List of search tool instances in priority order
        """
        engine_map = {
            "google": self.google,
            "bing": self.bing,
            "baidu": self.baidu,
            "duckduckgo": self.duckduckgo,
        }

        # Start with preferred engine
        engines = []
        if preferred in engine_map:
            engines.append(engine_map[preferred])

        # Add remaining engines as fallback
        fallback_order = ["google", "duckduckgo", "bing", "baidu"]
        for engine_name in fallback_order:
            engine = engine_map[engine_name]
            if engine not in engines:
                engines.append(engine)

        return engines

    def _run(
        self,
        query: str,
        num_results: int = 5,
        preferred_engine: Optional[str] = "google",
    ) -> str:
        """
        Execute multi-search with fallback.

        Args:
            query: Search query
            num_results: Number of results to return
            preferred_engine: Preferred search engine

        Returns:
            Formatted search results from the first successful engine
        """
        engines = self.get_engine_order(preferred_engine or "google")
        retry_count = 0

        while retry_count < self.max_retries:
            for engine in engines:
                try:
                    logger.info(f"Trying {engine.name} for query: {query}")
                    result = engine._run(query, num_results)

                    # Check if result is an error message
                    if not result.startswith("Error:") and not result.startswith(
                        f"{engine.name.replace('_', ' ').title()} search error:"
                    ):
                        logger.info(
                            f"Successfully retrieved results from {engine.name}"
                        )
                        return f"[Source: {engine.name}]\n\n{result}"

                    logger.warning(f"{engine.name} returned error: {result}")

                except Exception as e:
                    logger.warning(f"{engine.name} failed with exception: {e}")
                    continue

            # All engines failed, increment retry counter
            retry_count += 1
            if retry_count < self.max_retries:
                logger.warning(
                    f"All search engines failed. Retrying in {self.retry_delay}s "
                    f"(attempt {retry_count}/{self.max_retries})"
                )
                time.sleep(self.retry_delay)

        # All retries exhausted
        return (
            f"All search engines failed after {self.max_retries} retry attempts. "
            f"Query: {query}. Please try again later or rephrase your query."
        )

    async def _arun(
        self,
        query: str,
        num_results: int = 5,
        preferred_engine: Optional[str] = "google",
    ) -> str:
        """Async version - calls sync version."""
        # TODO: Implement proper async version with asyncio.gather for parallel attempts
        return self._run(query, num_results, preferred_engine)
