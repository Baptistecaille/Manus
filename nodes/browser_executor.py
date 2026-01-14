"""
Browser Executor Node - Enhanced with Planning Manager integration.

Executes browser automation tasks using Playwright-based BrowserAutomationSkill
with action logging and retry logic.
"""

import logging
import os
from typing import Any, Optional

from agent_state import AgentStateDict

logger = logging.getLogger(__name__)

# Maximum retry attempts for browser operations
MAX_RETRIES = int(os.getenv("BROWSER_MAX_RETRIES", "3"))


async def browser_executor_node(state: AgentStateDict) -> dict[str, Any]:
    """
    Execute browser automation task with planning integration.

    Workflow:
    1. Extract action from state (navigate, click, fill, screenshot, extract)
    2. Parse action_details for URL/selector/form_data
    3. Execute action using BrowserAutomationSkill
    4. Log result via PlanningManager
    5. Retry on failure (max 3 attempts)

    Args:
        state: Current agent state with action_details.

    Returns:
        Updated state with browser execution results.

    Example action_details formats:
        - "navigate: https://example.com"
        - "click: #submit-button"
        - "fill: #username=john, #password=secret"
        - "screenshot: /workspace/page.png"
        - "extract: .article-content"
    """
    logger.info("=== Browser Executor Node (Enhanced) ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)
    workspace = state.get("workspace_dir", os.getenv("WORKSPACE_DIR", "/workspace"))

    if not action_details:
        error_msg = "Browser executor requires action_details with command"
        logger.error(error_msg)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "messages": [{"role": "system", "content": f"[BROWSER ERROR] {error_msg}"}],
        }

    # Parse action and parameters
    action, params = _parse_action_details(action_details)

    # Execute with retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await _execute_browser_action(action, params, workspace)

            # Log action via PlanningManager (if available)
            await _log_browser_action(
                workspace,
                f"Browser {action}: {params.get('target', 'N/A')[:50]}",
                str(result)[:200],
                success=True,
            )

            logger.info(f"Browser {action} succeeded on attempt {attempt}")

            return {
                "last_tool_output": result.get("content", str(result)),
                "iteration_count": iteration + 1,
                "actions_since_refresh": state.get("actions_since_refresh", 0) + 1,
                "actions_since_save": state.get("actions_since_save", 0) + 1,
                "browser_session": result.get("session_info"),
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"[BROWSER] {action.upper()} completed:\n{result.get('summary', result.get('content', '')[:500])}",
                    }
                ],
            }

        except Exception as e:
            logger.warning(f"Browser attempt {attempt}/{MAX_RETRIES} failed: {e}")

            if attempt == MAX_RETRIES:
                error_msg = f"Browser {action} failed after {MAX_RETRIES} attempts: {e}"

                # Log failure
                await _log_browser_action(
                    workspace,
                    f"Browser {action}: {params.get('target', 'N/A')[:50]}",
                    error_msg,
                    success=False,
                )

                return {
                    "last_tool_output": error_msg,
                    "iteration_count": iteration + 1,
                    "error_count": state.get("error_count", 0) + 1,
                    "messages": [
                        {"role": "system", "content": f"[BROWSER ERROR] {error_msg}"}
                    ],
                }

    # Fallback (shouldn't reach here)
    return {
        "last_tool_output": "Browser execution finished unexpectedly",
        "iteration_count": iteration + 1,
    }


def _parse_action_details(details: str) -> tuple[str, dict[str, Any]]:
    """
    Parse action details string into action type and parameters.

    Formats supported:
        - "navigate: https://example.com"
        - "click: #selector"
        - "fill: #field1=value1, #field2=value2"
        - "screenshot: /path/to/file.png?full_page=true"
        - "extract: .selector"
        - Plain URL (defaults to navigate)

    Args:
        details: Action details string.

    Returns:
        Tuple of (action_type, params_dict).
    """
    details = details.strip()

    # Check for action prefix
    if ":" in details:
        parts = details.split(":", 1)
        action = parts[0].strip().lower()
        remainder = parts[1].strip() if len(parts) > 1 else ""

        if action == "navigate":
            return "navigate", {"url": remainder}

        elif action == "click":
            return "click", {"selector": remainder}

        elif action == "fill":
            # Parse "selector=value, selector=value" format
            form_data = {}
            for pair in remainder.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    form_data[key.strip()] = value.strip()
            return "fill", {"form_data": form_data}

        elif action == "screenshot":
            # Parse path and optional params
            if "?" in remainder:
                path, query = remainder.split("?", 1)
                full_page = "full_page=true" in query.lower()
            else:
                path = remainder
                full_page = False
            return "screenshot", {"path": path.strip(), "full_page": full_page}

        elif action == "extract":
            return "extract", {"selector": remainder}

        else:
            # Unknown action, treat as task description
            return "task", {"description": details, "target": details[:50]}

    # Default: if looks like URL, navigate; else treat as task
    if details.startswith(("http://", "https://")):
        return "navigate", {"url": details}

    return "task", {"description": details, "target": details[:50]}


async def _execute_browser_action(
    action: str, params: dict[str, Any], workspace: str
) -> dict[str, Any]:
    """
    Execute a browser action using BrowserAutomationSkill.

    Args:
        action: Action type (navigate, click, fill, screenshot, extract, task).
        params: Action parameters.
        workspace: Workspace directory for screenshots.

    Returns:
        Result dict with content and metadata.
    """
    # Import here to avoid circular imports
    from skills.browser_automation import BrowserAutomationSkill

    async with BrowserAutomationSkill(headless=True) as browser:
        if action == "navigate":
            result = await browser.navigate(params["url"])
            return {
                "content": f"Navigated to {params['url']}",
                "summary": f"Page title: {result.get('title', 'N/A')}",
                "session_info": {"url": params["url"], "title": result.get("title")},
            }

        elif action == "click":
            result = await browser.click(params["selector"])
            return {
                "content": f"Clicked element: {params['selector']}",
                "summary": f"Click successful: {result.get('success', False)}",
            }

        elif action == "fill":
            result = await browser.fill_form(params["form_data"])
            return {
                "content": f"Filled form with {len(params['form_data'])} fields",
                "summary": f"Fields: {', '.join(params['form_data'].keys())}",
            }

        elif action == "screenshot":
            path = params["path"]
            # Ensure path is within workspace
            if not path.startswith("/"):
                path = f"{workspace}/{path}"
            result = await browser.screenshot(
                path, full_page=params.get("full_page", False)
            )
            return {
                "content": f"Screenshot saved to {result}",
                "summary": f"Screenshot: {result}",
            }

        elif action == "extract":
            text = await browser.extract_content(params["selector"])
            return {
                "content": text or "No content found",
                "summary": f"Extracted {len(text or '')} characters",
            }

        elif action == "task":
            # Complex task - use BrowserUseTool for agent-style execution
            from tools.browser_use import BrowserUseTool

            tool = BrowserUseTool()
            result = await tool._arun(
                task=params["description"], headless=True, max_steps=20
            )
            return {
                "content": result,
                "summary": result[:200] if result else "Task completed",
            }

        else:
            raise ValueError(f"Unknown browser action: {action}")


async def _log_browser_action(
    workspace: str, action: str, result: str, success: bool
) -> None:
    """
    Log browser action to progress file via PlanningManager.

    Args:
        workspace: Workspace directory.
        action: Action description.
        result: Result or error message.
        success: Whether action succeeded.
    """
    try:
        from nodes.planning_manager import PlanningManager

        manager = PlanningManager(workspace)
        await manager.log_action(action, result, success)
    except Exception as e:
        # Don't fail the main action if logging fails
        logger.warning(f"Failed to log browser action: {e}")
