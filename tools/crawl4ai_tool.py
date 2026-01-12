"""
Crawl4AI Web Crawling Tool for LangChain integration.

Intelligent web crawler for extracting structured content from websites.
"""

import logging
from typing import Optional, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Crawl4AIInput(BaseModel):
    """Input schema for Crawl4AI."""

    url: str = Field(description="URL to crawl")
    extraction_strategy: Literal["markdown", "html", "text"] = Field(
        default="markdown",
        description="Content extraction strategy: markdown, html, or text",
    )
    max_depth: int = Field(
        default=1, description="Maximum crawl depth for following links", ge=1, le=3
    )
    include_links: bool = Field(
        default=False, description="Include extracted links in output"
    )


class Crawl4AITool(BaseTool):
    """
    Intelligent web crawler using Crawl4AI library.

    Extracts structured content from web pages with various strategies.
    Supports markdown extraction, HTML parsing, and plain text extraction.
    """

    name: str = "crawl4ai"
    description: str = (
        "Crawl and extract structured content from websites using Crawl4AI. "
        "Returns markdown, HTML, or plain text. Useful for web scraping, "
        "content analysis, and data extraction from web pages."
    )
    args_schema: type[BaseModel] = Crawl4AIInput

    def _run(
        self,
        url: str,
        extraction_strategy: Literal["markdown", "html", "text"] = "markdown",
        max_depth: int = 1,
        include_links: bool = False,
    ) -> str:
        """
        Execute web crawling.

        Args:
            url: URL to crawl
            extraction_strategy: How to extract content
            max_depth: Crawl depth
            include_links: Whether to include links

        Returns:
            Extracted content
        """
        try:
            from crawl4ai import AsyncWebCrawler
            import asyncio

            async def crawl():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(url=url)

                    if not result.success:
                        return f"Crawl failed for {url}: {result.error_message}"

                    # Extract content based on strategy
                    if extraction_strategy == "markdown":
                        content = result.markdown
                    elif extraction_strategy == "html":
                        content = result.html
                    else:  # text
                        content = result.cleaned_html or result.html

                    output = f"Crawled content from {url}:\n\n{content}"

                    # Include links if requested
                    if include_links and result.links:
                        links_section = "\n\nExtracted Links:\n"
                        for link in result.links.get("internal", [])[:20]:
                            links_section += f"  • {link}\n"
                        output += links_section

                    return output

            # Run async function
            return asyncio.run(crawl())

        except ImportError:
            logger.error("crawl4ai not installed")
            return "Error: crawl4ai library not installed. Run: uv pip install crawl4ai"
        except Exception as e:
            logger.error(f"Crawl4AI error: {e}")
            return f"Crawl error: {str(e)}"

    async def _arun(
        self,
        url: str,
        extraction_strategy: Literal["markdown", "html", "text"] = "markdown",
        max_depth: int = 1,
        include_links: bool = False,
    ) -> str:
        """Async version - proper async implementation."""
        try:
            from crawl4ai import AsyncWebCrawler

            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)

                if not result.success:
                    return f"Crawl failed for {url}: {result.error_message}"

                # Extract content based on strategy
                if extraction_strategy == "markdown":
                    content = result.markdown
                elif extraction_strategy == "html":
                    content = result.html
                else:  # text
                    content = result.cleaned_html or result.html

                output = f"Crawled content from {url}:\n\n{content}"

                # Include links if requested
                if include_links and result.links:
                    links_section = "\n\nExtracted Links:\n"
                    for link in result.links.get("internal", [])[:20]:
                        links_section += f"  • {link}\n"
                    output += links_section

                return output

        except ImportError:
            logger.error("crawl4ai not installed")
            return "Error: crawl4ai library not installed"
        except Exception as e:
            logger.error(f"Crawl4AI error: {e}")
            return f"Crawl error: {str(e)}"
