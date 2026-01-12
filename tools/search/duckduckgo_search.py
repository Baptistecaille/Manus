"""
DuckDuckGo Search Tool for LangChain integration.

Provides web search functionality using the duckduckgo-search library.
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DuckDuckGoSearchInput(BaseModel):
    """Input schema for DuckDuckGo search."""

    query: str = Field(description="Search query to execute")
    num_results: int = Field(
        default=5, description="Number of results to return", ge=1, le=20
    )


class DuckDuckGoSearchTool(BaseTool):
    """
    DuckDuckGo Search tool.

    This tool performs web searches via DuckDuckGo and returns URLs and snippets.
    Privacy-focused search engine.
    """

    name: str = "duckduckgo_search"
    description: str = (
        "Search the web using DuckDuckGo. Returns URLs and snippets for the top results. "
        "Privacy-focused search engine, good for general queries."
    )
    args_schema: type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """
        Execute DuckDuckGo search.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Formatted search results
        """
        try:
            from duckduckgo_search import DDGS

            results = []

            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=num_results)

                for result in search_results:
                    title = result.get("title", "No title")
                    url = result.get("href", "")
                    body = result.get("body", "")

                    results.append(f"• {title}\n  URL: {url}\n  {body}")

            if not results:
                return f"No results found for query: {query}"

            return f"DuckDuckGo Search Results for '{query}':\n\n" + "\n\n".join(
                results
            )

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return f"DuckDuckGo search error: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Async version - calls sync version."""
        return self._run(query, num_results)
