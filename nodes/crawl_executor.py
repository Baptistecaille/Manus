"""
Crawl Executor Node for LangGraph.

Executes web crawling tasks using Crawl4AITool.
"""

import logging
from typing import Any, Dict

from agent_state import AgentStateDict
from tools.crawl4ai_tool import Crawl4AITool

logger = logging.getLogger(__name__)


async def crawl_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute web crawling task.

    Args:
        state: Current agent state

    Returns:
        Updated state with crawl results
    """
    logger.info("=== Crawl Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    if not action_details:
        error_msg = "Crawl executor requires action_details with URL"
        logger.error(error_msg)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
        }

    try:
        # Extract URL from action_details
        # Format expected: "URL|extraction_strategy" or just "URL"
        parts = action_details.split("|")
        url = parts[0].strip()
        extraction_strategy = parts[1].strip() if len(parts) > 1 else "markdown"

        # Initialize crawl tool
        crawl_tool = Crawl4AITool()

        # Execute crawl
        logger.info(f"Crawling URL: {url} with strategy: {extraction_strategy}")
        result = await crawl_tool._arun(
            url=url,
            extraction_strategy=extraction_strategy,
            max_depth=1,
            include_links=True,
        )

        logger.info(f"Crawl result (first 200 chars): {result[:200]}...")

        return {
            "last_tool_output": result,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }

    except Exception as e:
        error_msg = f"Crawl execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "current_action": "planning",
        }
