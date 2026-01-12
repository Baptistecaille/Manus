"""
Bing Search Tool for LangChain integration.

Provides web search functionality using a simple HTTP-based approach to Bing.
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BingSearchInput(BaseModel):
    """Input schema for Bing search."""

    query: str = Field(description="Search query to execute")
    num_results: int = Field(
        default=5, description="Number of results to return", ge=1, le=20
    )


class BingSearchTool(BaseTool):
    """
    Bing Search tool using web scraping.

    This tool performs web searches via Bing and returns URLs and snippets.
    Note: This is a basic implementation. For production, use Bing Search API.
    """

    name: str = "bing_search"
    description: str = (
        "Search the web using Bing. Returns URLs and snippets for the top results. "
        "Use this as fallback when Google search fails."
    )
    args_schema: type[BaseModel] = BingSearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """
        Execute Bing search.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Formatted search results
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import quote_plus

            # Bing search URL
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Parse Bing search results
            for result in soup.find_all("li", class_="b_algo", limit=num_results):
                title_elem = result.find("h2")
                link_elem = title_elem.find("a") if title_elem else None
                desc_elem = result.find("p")

                if link_elem:
                    title = title_elem.get_text(strip=True)
                    url = link_elem.get("href", "")
                    description = desc_elem.get_text(strip=True) if desc_elem else ""

                    results.append(f"• {title}\n  URL: {url}\n  {description}")

            if not results:
                return f"No results found for query: {query}"

            return f"Bing Search Results for '{query}':\n\n" + "\n\n".join(results)

        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return f"Bing search error: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Async version - calls sync version."""
        return self._run(query, num_results)
