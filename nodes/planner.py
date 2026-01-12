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

logger = logging.getLogger(__name__)

# System prompt template
SYSTEM_PROMPT = """You are an autonomous AI agent with access to a persistent Seedbox environment (a Docker container with bash, Python, and common tools).

Your goal is to help the user accomplish their task by breaking it down into steps and executing them one at a time.

AVAILABLE ACTIONS:
- bash: Execute a shell command in the Seedbox
- deep_research: Comprehensive multi-source web research on a topic (generates a detailed report)
- search: Quick web search for information (via DuckDuckGo)
- playwright: Navigate to a URL and extract content
- consolidate: Compress context when running low on memory
- complete: Task is finished, provide final summary

STRICT RESPONSE FORMAT (follow exactly):
INTERNAL_MONOLOGUE: [Your step-by-step reasoning about what to do next and why. Think through the problem carefully.]
TODO_LIST: [Format: ✅ Done: completed items | 🔲 Next: pending items]
NEXT_ACTION: [One of: bash|deep_research|search|playwright|consolidate|complete]
ACTION_DETAILS: [The specific command, query, or URL for the action. Be precise.]
REASONING: [Brief explanation of why this action moves toward the goal]

IMPORTANT RULES:
1. Always update the TODO_LIST to track progress
2. Be specific in ACTION_DETAILS - include exact commands/queries
3. If a command fails, analyze the error and try a different approach
4. Use the Seedbox filesystem (/workspace) for persistent storage
5. When task is complete, use NEXT_ACTION: complete and put the FINAL ANSWER or summary in ACTION_DETAILS

WHEN TO USE DEEP_RESEARCH:
- User asks to "research", "investigate", or "analyze" a topic in depth
- User wants a comprehensive report or analysis with multiple sources
- User says "deep research" or wants thorough multi-source analysis
- Topic requires gathering and synthesizing information from many sources

For deep_research, set ACTION_DETAILS to the research topic (e.g., "Latest advances in autonomous AI agents")"""

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

Analyze the situation and provide your next action following the STRICT RESPONSE FORMAT."""


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


def _parse_response(response_text: str) -> dict:
    """
    Parse the LLM response to extract structured fields.

    Args:
        response_text: Raw LLM response text.

    Returns:
        Dict with internal_monologue, todo_list, next_action, action_details, reasoning.
    """
    result = {
        "internal_monologue": "",
        "todo_list": "",
        "next_action": "complete",  # Default to complete if parsing fails
        "action_details": "",
        "reasoning": "",
    }

    # Define patterns for each section
    patterns = {
        "internal_monologue": r"INTERNAL_MONOLOGUE:\s*(.+?)(?=TODO_LIST:|$)",
        "todo_list": r"TODO_LIST:\s*(.+?)(?=NEXT_ACTION:|$)",
        "next_action": r"NEXT_ACTION:\s*(\w+)",
        "action_details": r"ACTION_DETAILS:\s*(.+?)(?=REASONING:|$)",
        "reasoning": r"REASONING:\s*(.+?)$",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Clean up the value
            value = re.sub(r"\s+", " ", value)  # Normalize whitespace

            # Remove surrounding quotes for action_details
            if field == "action_details":
                # Remove Markdown code blocks if present
                value = re.sub(r"^`+|`+$", "", value).strip()
                # Remove surrounding quotes
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1].strip()

            result[field] = value

    # Normalize next_action to lowercase
    result["next_action"] = result["next_action"].lower().strip()

    # Validate next_action
    valid_actions = {
        "bash",
        "deep_research",
        "search",
        "playwright",
        "consolidate",
        "complete",
    }
    if result["next_action"] not in valid_actions:
        logger.warning(
            f"Invalid action '{result['next_action']}', defaulting to 'complete'"
        )
        result["next_action"] = "complete"

    return result


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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = llm.invoke(messages)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        logger.debug(f"LLM Response:\n{response_text[:500]}...")

        # Parse the response
        parsed = _parse_response(response_text)

        # Build the assistant message for history
        assistant_message = {"role": "assistant", "content": response_text}

        # Calculate new context size
        new_context_size = calculate_context_size(state) + estimate_tokens(
            response_text
        )

        # Return state updates
        return {
            "messages": [assistant_message],  # Will be appended due to operator.add
            "internal_monologue": parsed["internal_monologue"],
            "todo_list": parsed["todo_list"],
            "current_action": parsed["next_action"],
            "action_details": parsed["action_details"],
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
