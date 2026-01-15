"""
Browser Executor Node - Orchestrates web interaction tasks.

Uses BrowserAutomationSkill to perform research, navigation, and extraction
based on the agent's state and current action.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from skills.browser_automation import BrowserAutomationSkill

# Import agent state definition to ensure type compatibility
try:
    from agent_state import AgentStateDict
except ImportError:
    # Relaxed type for circular imports or testing
    AgentStateDict = Dict[str, Any]

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


async def browser_executor_node(state: AgentStateDict) -> dict[str, Any]:
    """
    Execute browser-based tasks using Playwright.

    Parses the `action_details` or `current_action` from the state to determine
    navigation targets and extraction needs.

    Args:
        state: Current agent state.

    Returns:
        State updates with browser execution results.
    """
    logger.info("Starting Browser Executor")

    # Extract action details
    action_details = state.get("action_details")
    current_action = state.get("current_action", "")

    # Parse input if it's a string (e.g. from LLM output)
    target_url = None
    instruction = ""

    if isinstance(action_details, str):
        try:
            # Try parsing as JSON first
            details = json.loads(action_details)
            target_url = details.get("url")
            instruction = details.get("instruction", "")
        except json.JSONDecodeError:
            # Treat as simple string instruction
            instruction = action_details
            # Try to extract URL from string if present
            words = action_details.split()
            for word in words:
                if word.startswith("http"):
                    target_url = word
                    break
    elif isinstance(action_details, dict):
        target_url = action_details.get("url")
        instruction = action_details.get("instruction", "")

    # Fallback: if no URL found in details, maybe the previous tool output has it
    if not target_url:
        logger.warning("No URL found in action details. Checking recent history...")
        # (Optional: Logic to find URL in convo history could go here)

    if not target_url and "search" not in instruction.lower():
        return {
            "executor_outputs": [
                {
                    "source": "browser_executor",
                    "status": "failed",
                    "output": "No URL provided for browser execution.",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "last_tool_output": "Error: No URL provided.",
        }

    # Initialize Skill
    headless = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
    browser_skill = BrowserAutomationSkill(headless=headless)

    output_data = {}

    try:
        async with browser_skill:
            # 1. Navigate
            if target_url:
                result = await browser_skill.navigate(target_url)
                if not result["success"]:
                    raise RuntimeError(f"Navigation failed: {result.get('error')}")

                output_data["url"] = result["url"]
                output_data["title"] = result["title"]

            # 2. Extract Content (Default action)
            # In a real scenario, we might want specific selectors or interactions
            # For general research, getting page text is usually the goal.
            markdown_content = await browser_skill.get_page_text()

            # Truncate if too long (safe context limit)
            MAX_LEN = 15000
            if len(markdown_content) > MAX_LEN:
                markdown_content = markdown_content[:MAX_LEN] + "\n...[truncated]..."

            output_data["content"] = markdown_content

            # 3. Screenshot (Optional - good for verification)
            # screenshot_path = await browser_skill.screenshot(f"/workspace/screenshot_{int(datetime.now().timestamp())}.png")
            # output_data["screenshot"] = screenshot_path

            success_msg = f"Successfully browsed {target_url or 'page'}"
            status = "success"

    except Exception as e:
        logger.error(f"Browser execution error: {e}")
        success_msg = f"Browser error: {str(e)}"
        status = "failed"
        output_data["error"] = str(e)

    # Format output for state
    return {
        "executor_outputs": [
            {
                "source": "browser_executor",
                "status": status,
                "output": f"Browsed {target_url}\nTitle: {output_data.get('title')}\nContent Preview: {output_data.get('content', '')[:200]}...",
                "data": output_data,
                "timestamp": datetime.now().isoformat(),
            }
        ],
        "last_tool_output": json.dumps(output_data)[
            :2000
        ],  # Provide raw data dump to LLM (truncated)
    }
