"""
SubAgent Executor Node - Deep Agents SubAgentMiddleware integration.

LangGraph node that spawns and manages sub-agents for delegated tasks.
Sub-agents provide context isolation for complex subtasks while keeping
the main agent's context clean.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from agent_state import AgentStateDict

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_DEPTH = 3  # Maximum sub-agent nesting depth
DEFAULT_TIMEOUT = 300  # 5 minutes per sub-agent
DEFAULT_MAX_STEPS = 10  # Maximum steps per sub-agent


def subagent_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Spawn and execute a sub-agent via Deep Agents SubAgentMiddleware.

    Used for delegating complex subtasks that benefit from context isolation.
    Sub-agents run independently and return their results to the parent.

    Args:
        state: Current agent state with task description and context.

    Returns:
        Updated state with sub-agent execution result.

    Example:
        >>> state = AgentStateDict(
        ...     action_details={
        ...         "task_description": "Research Python history",
        ...         "max_steps": 5
        ...     }
        ... )
        >>> result = subagent_executor_node(state)
    """
    logger.info("=== SubAgent Executor Node ===")

    # Extract task details
    action_details = state.get("action_details", {})
    if isinstance(action_details, str):
        import json

        try:
            action_details = json.loads(action_details)
        except json.JSONDecodeError:
            action_details = {"task_description": action_details}

    task_description = action_details.get(
        "task_description",
        action_details.get("description", action_details.get("task", "")),
    )
    task_context = action_details.get("context", {})
    max_steps = action_details.get("max_steps", DEFAULT_MAX_STEPS)
    timeout = action_details.get("timeout", DEFAULT_TIMEOUT)

    # Check sub-agent depth limit
    current_depth = state.get("subagent_depth", 0)
    if current_depth >= DEFAULT_MAX_DEPTH:
        return {
            "last_tool_output": f"Error: Maximum sub-agent depth ({DEFAULT_MAX_DEPTH}) reached",
            "executor_outputs": [
                {
                    "source": "subagent_executor",
                    "status": "failed",
                    "output": "Max depth exceeded",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    if not task_description:
        return {
            "last_tool_output": "Error: No task description provided for sub-agent",
            "executor_outputs": [
                {
                    "source": "subagent_executor",
                    "status": "failed",
                    "output": "Missing task description",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    output_msg = ""
    status = "success"
    subagent_id = f"subagent_{current_depth}_{datetime.now().strftime('%H%M%S')}"

    try:
        # Try Deep Agents SubAgentMiddleware
        result = _execute_via_deepagents(
            task_description, task_context, max_steps, timeout
        )

        if result is not None:
            output_msg = result
        else:
            # Fallback to simulated execution
            output_msg = _execute_simulated(task_description, task_context)

    except Exception as e:
        logger.error(f"Sub-agent execution failed: {e}", exc_info=True)
        output_msg = f"Sub-agent error: {str(e)}"
        status = "failed"

    # Update active sub-agents list
    active_subagents = state.get("active_subagents", []) or []
    active_subagents = active_subagents.copy()
    active_subagents.append(subagent_id)

    return {
        "last_tool_output": output_msg,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "active_subagents": active_subagents,
        "executor_outputs": [
            {
                "source": "subagent_executor",
                "status": status,
                "subagent_id": subagent_id,
                "task": task_description[:100],
                "output": output_msg[:1000] if len(output_msg) > 1000 else output_msg,
                "timestamp": datetime.now().isoformat(),
            }
        ],
    }


def _execute_via_deepagents(
    description: str, context: Dict[str, Any], max_steps: int, timeout: int
) -> Optional[str]:
    """
    Execute sub-agent via Deep Agents SubAgentMiddleware.

    Returns None if Deep Agents is not available.
    """
    try:
        from middleware.deepagents_setup import get_deepagents_config

        config = get_deepagents_config()
        subagent_middleware = config.get_subagent_middleware()

        if subagent_middleware is None:
            return None

        # Find the 'task' tool
        tools = config.get_tools()
        task_tool = None
        for t in tools:
            if t.name == "task":
                task_tool = t
                break

        if task_tool is None:
            return None

        # Execute the task tool
        logger.debug(f"Executing sub-agent via Deep Agents: {description[:50]}...")
        result = task_tool.invoke(
            {
                "description": description,
                "context": context,
            }
        )

        return str(result)

    except ImportError:
        logger.debug("Deep Agents not available for sub-agent execution")
        return None
    except Exception as e:
        logger.warning(f"Deep Agents sub-agent execution failed: {e}")
        return None


def _execute_simulated(description: str, context: Dict[str, Any]) -> str:
    """
    Simulated sub-agent execution for when Deep Agents is not available.

    Returns a placeholder result indicating the sub-agent would handle the task.
    """
    logger.info(f"Simulating sub-agent for: {description[:50]}...")

    # In a real implementation, this could:
    # 1. Create a new agent graph with reduced capabilities
    # 2. Run the sub-graph with isolated context
    # 3. Return the results

    return (
        f"[Sub-Agent Simulation]\n"
        f"Task: {description}\n"
        f"Status: Would delegate to sub-agent with isolated context.\n"
        f"Note: Install 'deepagents' package for full sub-agent functionality.\n"
        f"Context keys: {list(context.keys()) if context else 'None'}"
    )


async def subagent_executor_node_async(state: AgentStateDict) -> Dict[str, Any]:
    """Async version of subagent_executor_node."""
    return subagent_executor_node(state)
