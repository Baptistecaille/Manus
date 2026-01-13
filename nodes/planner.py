"""
Planner Node - The brain of the Manus agent.

This node calls the LLM to analyze the current state and decide
the next action. It implements the "Internal Monologue" pattern
where the agent explicitly reasons before acting.
"""

import logging
import re
from typing import Optional

from agent_state import AgentStateDict, estimate_tokens, calculate_context_size
from llm_factory import create_llm
from nodes.schema import PlannerOutput

from prompts import get_planner_system_prompt

logger = logging.getLogger(__name__)

# System prompt is now dynamic via centralized module
SYSTEM_PROMPT = get_planner_system_prompt()

# User message template for each planning step
# User message template for each planning step
USER_TEMPLATE = """CURRENT CONTEXT:
{context}

CURRENT STATE:
- Todo List: {todo_list}
- Last Reflection: {internal_monologue}
- Seedbox Files: {seedbox_manifest}
- Last Tool Result: {last_tool_output}

Iteration: {iteration_count}
Context Size: ~{context_size} tokens

CRITICAL INSTRUCTION: STAGING AREA PROTOCOL
1. You MUST assume all file operations (create/edit) are initially done in a temporary directory (e.g., /workspace/temp_staging).
2. Create this directory if it doesn't exist.
3. Only when a file is finalized and verified should you move it to the root /workspace/ or its final destination.
4. Clean up the temporary directory after moving the final artifact.

Analyze the situation and provide your next action by populating the PlannerOutput structure."""


def _build_context(state: AgentStateDict) -> str:
    """Build the context string from messages and consolidated history."""
    context_size = state.get("context_size", 0)
    consolidated = state.get("consolidated_history", "")
    messages = state.get("messages", [])

    # If context is large, use consolidated history + recent messages
    if context_size > 50000 and consolidated:
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        context_parts = [
            "=== CONSOLIDATED HISTORY ===",
            consolidated,
            "",
            "=== RECENT MESSAGES ===",
        ]
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:2000]
            context_parts.append(f"[{role.upper()}]: {content}")
        return "\n".join(context_parts)

    # Otherwise, use full message history
    context_parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        context_parts.append(f"[{role.upper()}]: {content}")

    return "\n".join(context_parts)


# regex parsing function `_parse_response` is removed as we use structured output
# def _parse_response(response_text: str) -> dict: ...


def planner_node(state: AgentStateDict) -> dict:
    """
    LangGraph node that plans the next action.

    This is the "brain" of the agent. It:
    1. Builds context from message history
    2. Calls the LLM with a structured prompt
    3. Parses the response to extract the next action
    4. Updates the state with new values

    Args:
        state: Current agent state.

    Returns:
        Dict of state updates (LangGraph will merge these).
    """
    logger.info(f"Planner node - Iteration {state.get('iteration_count', 0)}")

    # Build the prompt
    context = _build_context(state)

    user_message = USER_TEMPLATE.format(
        context=context[:10000],  # Limit context in prompt
        todo_list=state.get("todo_list", "No tasks defined"),
        internal_monologue=state.get("internal_monologue", "Starting fresh"),
        seedbox_manifest=", ".join(state.get("seedbox_manifest", [])[:20]) or "Empty",
        last_tool_output=state.get("last_tool_output", "No previous output")[:1000],
        iteration_count=state.get("iteration_count", 0),
        context_size=state.get("context_size", 0),
    )

    try:
        # Create LLM and invoke
        llm = create_llm()
        structured_llm = llm.with_structured_output(PlannerOutput)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        parsed: PlannerOutput = structured_llm.invoke(messages)

        if parsed is None:
            raise ValueError("LLM returned None for structured output")

        # Log the decision
        logger.debug(
            f"Planner decision: Action={parsed.next_action}, Thought={parsed.internal_monologue[:50]}..."
        )

        # Build the assistant message for history
        # We perform a rough reconstruction of the response for the message history
        # since we don't have the raw text anymore.
        assistant_content_str = (
            f"INTERNAL_MONOLOGUE: {parsed.internal_monologue}\n"
            f"TODO_LIST: {parsed.todo_list}\n"
            f"NEXT_ACTION: {parsed.next_action}\n"
            f"ACTION_DETAILS: {parsed.action_details}\n"
            f"REASONING: {parsed.reasoning}"
        )
        assistant_message = {"role": "assistant", "content": assistant_content_str}

        # Calculate new context size
        new_context_size = calculate_context_size(state) + estimate_tokens(
            assistant_content_str
        )

        # Return state updates
        return {
            "messages": [assistant_message],  # Will be appended due to operator.add
            "internal_monologue": parsed.internal_monologue,
            "todo_list": parsed.todo_list,
            "current_action": parsed.next_action,
            "action_details": parsed.action_details,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "context_size": new_context_size,
        }

    except Exception as e:
        logger.error(f"Planner error: {e}")

        # Return error state - will trigger completion
        error_message = {
            "role": "assistant",
            "content": f"INTERNAL_MONOLOGUE: Error occurred during planning: {str(e)}\n"
            f"TODO_LIST: ❌ Error\n"
            f"NEXT_ACTION: complete\n"
            f"ACTION_DETAILS: N/A\n"
            f"REASONING: Cannot continue due to error",
        }

        return {
            "messages": [error_message],
            "internal_monologue": f"Error: {str(e)}",
            "current_action": "complete",
            "action_details": "",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)

    from agent_state import create_initial_state

    state = create_initial_state("List the files in /workspace")
    result = planner_node(state)

    print(f"\nPlanner result:")
    print(f"  Action: {result.get('current_action')}")
    print(f"  Details: {result.get('action_details')}")
    print(f"  Monologue: {result.get('internal_monologue')[:200]}...")
