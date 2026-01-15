"""
Search Executor Node.

Wraps the MultiSearchTool to perform web searches with automatic fallback
between Google, DuckDuckGo, Bing, and Baidu.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict

from agent_state import AgentStateDict
from tools.search.duckduckgo_search import DuckDuckGoSearchTool

logger = logging.getLogger(__name__)


def search_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute web search using DuckDuckGo.

    Parses action_details to get the search query and executes
    using DuckDuckGo search with fallback.

    Args:
        state: Current agent state with action_details containing query.

    Returns:
        Updated state with search results.
    """
    logger.info("=== Search Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    # Parse query from action_details
    query = ""
    num_results = 5

    if isinstance(action_details, dict):
        query = action_details.get("query", action_details.get("search", ""))
        num_results = action_details.get("num_results", 5)
    elif isinstance(action_details, str):
        # Try JSON first
        try:
            params = json.loads(action_details)
            query = params.get("query", params.get("search", action_details))
            num_results = params.get("num_results", 5)
        except json.JSONDecodeError:
            # Use the string as the query directly
            query = action_details.strip()

    if not query:
        return {
            "last_tool_output": "No search query provided.",
            "iteration_count": iteration + 1,
            "executor_outputs": [
                {
                    "source": "search_executor",
                    "status": "failed",
                    "output": "No query",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    output_msg = ""
    status = "success"

    try:
        # Use DuckDuckGo directly (it's reliable and privacy-focused)
        search_tool = DuckDuckGoSearchTool()
        result = search_tool._run(query, num_results)
        output_msg = result
        logger.info(f"Search completed for query: {query}")

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        output_msg = f"Search error: {str(e)}"
        status = "failed"

    # Update search history
    search_history = state.get("search_history", []) or []
    search_history = search_history.copy()
    search_history.append(
        {
            "query": query,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
    )

    return {
        "last_tool_output": output_msg,
        "iteration_count": iteration + 1,
        "search_history": search_history,
        "executor_outputs": [
            {
                "source": "search_executor",
                "status": status,
                "query": query,
                "output": output_msg[:2000] if len(output_msg) > 2000 else output_msg,
                "timestamp": datetime.now().isoformat(),
            }
        ],
    }


async def search_executor_node_async(state: AgentStateDict) -> Dict[str, Any]:
    """Async version of search_executor_node."""
    return search_executor_node(state)
