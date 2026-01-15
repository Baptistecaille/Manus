"""
Search & Summarize Node - Web search and result summarization.

Executes DuckDuckGo searches for generated queries and summarizes
each result using the LLM for relevance to the research topic.
"""

import logging
import time
from typing import List, Dict, Any

from deep_research_state import DeepResearchStateDict
from deep_research_config import DeepResearchConfig, DEFAULT_CONFIG
from llm_factory import create_llm
from nodes.schema import SearchSummaryOutput

logger = logging.getLogger(__name__)

# Try to import duckduckgo-search
try:
    from ddgs import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("ddgs package not installed. Search will not work. Run: uv add ddgs")


# Summarization prompt
SUMMARIZE_PROMPT = """Summarize this web search result in relation to the research topic.

RESEARCH TOPIC: {topic}
SEARCH QUERY: {query}

SOURCE:
Title: {title}
URL: {url}
Content: {content}

Provide a concise summary (max 300 words) that:
1. Highlights key information relevant to the research topic
2. Preserves important facts, statistics, dates, and quotes
3. Notes the source type and credibility (academic, news, blog, marketing, official)
4. Indicates if information appears current or potentially outdated

RESPOND IN THIS FORMAT:
SUMMARY: [Your summary here]
SOURCE_TYPE: [academic|news|blog|marketing|official|unknown]
RELEVANCE: [high|medium|low]"""


def _search_duckduckgo(
    query: str, max_results: int = 10, max_retries: int = 3, retry_delay: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Execute DuckDuckGo search with retry logic.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        max_retries: Number of retry attempts.
        retry_delay: Delay between retries in seconds.

    Returns:
        List of search result dicts with title, href, body.
    """
    if not DDGS_AVAILABLE:
        logger.error("DuckDuckGo search not available")
        return []

    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                        region="wt-wt",  # Worldwide
                        safesearch="moderate",
                    )
                )
            logger.debug(f"Search '{query[:30]}...' returned {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"Search attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Search failed after {max_retries} attempts: {e}")
                return []

    return []


def _summarize_result(
    result: Dict[str, Any], topic: str, query: str, llm, config: DeepResearchConfig
) -> Dict[str, Any]:
    """
    Summarize a single search result using LLM.

    Args:
        result: Raw search result dict.
        topic: Research topic.
        query: Query that produced this result.
        llm: LLM instance.
        config: Research configuration.

    Returns:
        Enhanced result dict with summary.
    """
    title = result.get("title", "")
    url = result.get("href", result.get("url", ""))
    content = result.get("body", result.get("snippet", ""))

    prompt = SUMMARIZE_PROMPT.format(
        topic=topic,
        query=query,
        title=title,
        url=url,
        content=content[:2000],  # Limit content length
    )

    try:
        # Use structured output if the LLM isn't passed with it already,
        # but here we are passed a generic LLM. We should probably wrap it here or expect it wrapped.
        # Since the `llm` arg is passed in `search_and_summarize_node`, we'll wrap it there.
        # However, `_summarize_result` is called inside a loop.
        # Better to instantiate the structured version once in the main node function.
        # But for now let's just use it here locally or check if it supports it.
        # Simpler: Just use with_structured_output here. It returns a Runnable.
        structured_llm = llm.with_structured_output(SearchSummaryOutput)

        parsed: SearchSummaryOutput = structured_llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a research assistant. Be concise and factual.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        if parsed is None:
            # Fallback to using content directly if structured output fails
            return {
                "query": query,
                "title": title,
                "url": url,
                "snippet": content[:500],
                "summary": content[:300],
                "source_quality": "unknown",
                "relevance": "unknown",
            }

        return {
            "query": query,
            "title": title,
            "url": url,
            "snippet": content[:500],
            "summary": parsed.summary[: config.max_summary_tokens * 4],
            "source_quality": parsed.source_type,
            "relevance": parsed.relevance,
        }

    except Exception as e:
        logger.warning(f"Summarization error for {url}: {e}")
        return {
            "query": query,
            "title": title,
            "url": url,
            "snippet": content[:500],
            "summary": content[:300],  # Use snippet as fallback
            "source_quality": "unknown",
            "relevance": "unknown",
        }


def search_and_summarize_node(
    state: DeepResearchStateDict, config: DeepResearchConfig = DEFAULT_CONFIG
) -> dict:
    """
    LangGraph node that searches and summarizes results.

    Args:
        state: Current deep research state with queries.
        config: Research configuration.

    Returns:
        State updates with search_results and findings.
    """
    queries = state.get("research_queries", [])
    topic = state.get("research_topic", "")
    current_depth = state.get("research_depth", 0)
    existing_findings = state.get("findings", [])

    logger.info(f"Search & Summarize - Processing {len(queries)} queries")

    if not queries:
        logger.warning("No queries to search")
        return {
            "messages": [{"role": "system", "content": "[SEARCH] No queries provided"}],
        }

    # Create summarization LLM
    try:
        llm = create_llm(
            provider=config.summarization_provider,
            model=config.summarization_model,
            temperature=config.summarization_temperature,
        )
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"[SEARCH ERROR] LLM creation failed: {e}",
                }
            ],
        }

    all_results = []
    new_findings = []
    source_index = len(existing_findings) + 1  # Continue numbering

    for query in queries:
        logger.info(f"Searching: {query[:50]}...")

        # Execute search
        raw_results = _search_duckduckgo(
            query,
            max_results=config.max_search_results,
            max_retries=config.search_max_retries,
            retry_delay=config.search_retry_delay,
        )

        if not raw_results:
            logger.warning(f"No results for query: {query}")
            continue

        # Summarize each result
        for result in raw_results:
            summarized = _summarize_result(result, topic, query, llm, config)
            all_results.append(summarized)

            # Add to findings if relevant
            if summarized.get("relevance") != "low":
                new_findings.append(
                    {
                        "content": summarized["summary"],
                        "source_title": summarized["title"],
                        "source_url": summarized["url"],
                        "source_index": source_index,
                    }
                )
                source_index += 1

    # Combine with existing findings
    combined_findings = existing_findings + new_findings

    # Build summary message
    message_content = (
        f"[SEARCH COMPLETE] Cycle {current_depth + 1}\n"
        f"  Queries executed: {len(queries)}\n"
        f"  Results found: {len(all_results)}\n"
        f"  New findings added: {len(new_findings)}\n"
        f"  Total findings: {len(combined_findings)}"
    )

    logger.info(message_content.replace("[SEARCH COMPLETE] ", ""))

    return {
        "search_results": all_results,
        "findings": combined_findings,
        "messages": [{"role": "assistant", "content": message_content}],
        "last_tool_output": f"Searched {len(queries)} queries, found {len(all_results)} results",
    }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    from deep_research_state import create_deep_research_state

    state = create_deep_research_state("LangGraph agent frameworks")
    state["research_queries"] = [
        "LangGraph tutorial 2024",
        "LangGraph vs LangChain comparison",
    ]

    # Use a minimal config for testing
    test_config = DeepResearchConfig(max_search_results=3)

    result = search_and_summarize_node(state, test_config)

    print(f"\nSearch results: {len(result.get('search_results', []))}")
    print(f"Findings: {len(result.get('findings', []))}")
