"""
Baidu Search Tool for LangChain integration.

Provides web search functionality using the baidusearch library.
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BaiduSearchInput(BaseModel):
    """Input schema for Baidu search."""

    query: str = Field(description="Search query to execute")
    num_results: int = Field(
        default=5, description="Number of results to return", ge=1, le=20
    )


class BaiduSearchTool(BaseTool):
    """
    Baidu Search tool using baidusearch library.

    This tool performs web searches via Baidu and returns URLs and snippets.
    Useful for Chinese language queries.
    """

    name: str = "baidu_search"
    description: str = (
        "Search the web using Baidu (Chinese search engine). "
        "Returns URLs and snippets for the top results. "
        "Use this for Chinese language queries or when other search engines fail."
    )
    args_schema: type[BaseModel] = BaiduSearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """
        Execute Baidu search.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Formatted search results
        """
        try:
            from baidusearch import search

            results = []
            search_results = search(query, num_results=num_results)

            for result in search_results:
                # baidusearch returns dict with 'title', 'url', 'abstract'
                title = result.get("title", "No title")
                url = result.get("url", "")
                abstract = result.get("abstract", "")

                results.append(f"• {title}\n  URL: {url}\n  {abstract}")

            if not results:
                return f"No results found for query: {query}"

            return f"Baidu Search Results for '{query}':\n\n" + "\n\n".join(results)

        except ImportError:
            logger.error("baidusearch not installed")
            return "Error: baidusearch library not installed"
        except Exception as e:
            logger.error(f"Baidu search failed: {e}")
            return f"Baidu search error: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Async version - calls sync version."""
        return self._run(query, num_results)
