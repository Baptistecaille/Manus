"""
Data Analysis Executor Node.

Wraps the DataAnalyzerSkill to allow the agent to analyze datasets.
"""

import logging
import json
from typing import Any, Dict

from agent_state import AgentStateDict
from skills.data_analyzer import DataAnalyzerSkill

logger = logging.getLogger(__name__)


def data_analysis_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute data analysis tasks.

    Args:
        state: Current agent state with action_details.

    Returns:
        Updated state with analysis results.
    """
    logger.info("=== Data Analysis Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    # Initialize skill
    skill = DataAnalyzerSkill()

    try:
        # Parse details
        # Format expectation: "file.csv | query" or JSON
        file_path = ""
        query = ""

        if "|" in action_details:
            parts = action_details.split("|")
            file_path = parts[0].strip()
            query = parts[1].strip() if len(parts) > 1 else "Summarize this data"
        else:
            # Assume it's just a file path and default query
            file_path = action_details.strip()
            query = "Analyze this data"

        logger.info(f"Analyzing file: {file_path} with query: {query}")

        # Execute analysis
        import asyncio
        import os

        # Determine analysis method
        ext = os.path.splitext(file_path)[1].lower()

        async def _run_analysis():
            if ext == ".csv":
                return await skill.analyze_csv(file_path)
            elif ext == ".json":
                return await skill.analyze_json(file_path)
            else:
                return f"Unsupported file extension: {ext}"

        loop = asyncio.get_event_loop()

        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = executor.submit(asyncio.run, _run_analysis()).result()
        else:
            result = loop.run_until_complete(_run_analysis())

        logger.info("Analysis complete")

        return {
            "last_tool_output": f"Analysis Result:\n{str(result)[:2000]}",  # Truncate for safety
            "iteration_count": iteration + 1,
            "current_action": "data_analysis",
        }

    except Exception as e:
        error_msg = f"Data analysis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "current_action": "data_analysis",
        }
