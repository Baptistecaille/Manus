"""
Google Search Tool for LangChain integration.

Provides web search functionality using the googlesearch-python library.
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GoogleSearchInput(BaseModel):
    """Input schema for Google search."""

    query: str = Field(description="Search query to execute")
    num_results: int = Field(
        default=5, description="Number of results to return", ge=1, le=20
    )
    lang: str = Field(default="en", description="Language code (e.g., 'en', 'fr')")


class GoogleSearchTool(BaseTool):
    """
    Google Search tool using googlesearch-python library.

    This tool performs web searches and returns URLs and snippets.
    """

    name: str = "google_search"
    description: str = (
        "Search the web using Google. Returns URLs and snippets for the top results. "
        "Use this when you need to find current information from the internet."
    )
    args_schema: type[BaseModel] = GoogleSearchInput

    def _run(self, query: str, num_results: int = 5, lang: str = "en") -> str:
        """
        Execute Google search.

        Args:
            query: Search query
            num_results: Number of results to return
            lang: Language code

        Returns:
            Formatted search results
        """
        try:
            from googlesearch import search

            results = []
            for url in search(query, num_results=num_results, lang=lang, advanced=True):
                # googlesearch returns SearchResult objects with .url, .title, .description
                results.append(f"• {url.title}\n  URL: {url.url}\n  {url.description}")

            if not results:
                return f"No results found for query: {query}"

            return f"Google Search Results for '{query}':\n\n" + "\n\n".join(results)

        except ImportError:
            logger.error("googlesearch-python not installed")
            return "Error: googlesearch-python library not installed"
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return f"Google search error: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5, lang: str = "en") -> str:
        """Async version - calls sync version."""
        return self._run(query, num_results, lang)
