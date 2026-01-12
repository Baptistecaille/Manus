"""
Consolidator Node - Context compression for long-running tasks.

When context size exceeds the threshold, this node summarizes
old messages to reduce token usage while preserving critical information.
"""

import logging

from agent_state import AgentStateDict, estimate_tokens, calculate_context_size
from llm_factory import create_llm

logger = logging.getLogger(__name__)

# Consolidation prompt
CONSOLIDATE_PROMPT = """You are summarizing an agent's execution history to reduce context size while preserving critical information.

EXECUTION HISTORY TO SUMMARIZE:
{history}

Create a structured summary in this exact format:

COMPLETED_ACTIONS:
[List each successful action taken, with brief details]

KEY_DECISIONS:
[Important strategic choices made during execution]

ERRORS_ENCOUNTERED:
[Problems faced and how they were resolved]

CURRENT_PROGRESS:
[Where we are in relation to the original goal]

IMPORTANT_CONTEXT:
[Any critical information needed to continue the task]

Be concise but preserve all information needed to continue the task effectively."""


def consolidator_node(state: AgentStateDict) -> dict:
    """
    LangGraph node that consolidates/compresses the context.

    This node:
    1. Takes all messages except the most recent 3
    2. Summarizes them using the LLM
    3. Stores the summary in consolidated_history
    4. Removes old messages to reduce context size

    Args:
        state: Current agent state with potentially large message history.

    Returns:
        Dict of state updates with compressed context.
    """
    messages = state.get("messages", [])
    current_size = state.get("context_size", calculate_context_size(state))

    logger.info(f"Consolidator node - Current context size: {current_size} tokens")

    # Keep the last 3 messages
    keep_count = 3
    if len(messages) <= keep_count:
        logger.info("Not enough messages to consolidate")
        return {}

    # Split messages
    messages_to_summarize = messages[:-keep_count]
    messages_to_keep = messages[-keep_count:]

    # Build history string
    history_parts = []
    for i, msg in enumerate(messages_to_summarize):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:2000]  # Limit each message
        history_parts.append(f"[{i+1}] [{role.upper()}]: {content}")

    history_text = "\n\n".join(history_parts)

    # Combine with existing consolidated history if any
    existing_consolidated = state.get("consolidated_history", "")
    if existing_consolidated:
        history_text = f"=== PREVIOUS CONSOLIDATION ===\n{existing_consolidated}\n\n=== NEW MESSAGES TO CONSOLIDATE ===\n{history_text}"

    try:
        # Create LLM and invoke
        llm = create_llm()

        prompt = CONSOLIDATE_PROMPT.format(
            history=history_text[:15000]
        )  # Limit history size

        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a precise summarizer. Preserve critical details.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        summary = response.content if hasattr(response, "content") else str(response)

        # Calculate new context size
        new_context_size = (
            estimate_tokens(summary)
            + sum(estimate_tokens(str(m.get("content", ""))) for m in messages_to_keep)
            + estimate_tokens(state.get("todo_list", ""))
            + estimate_tokens(state.get("internal_monologue", ""))
        )

        logger.info(
            f"Consolidation complete. New context size: {new_context_size} tokens "
            f"(reduced from {current_size})"
        )

        # Create a system message noting the consolidation
        consolidation_message = {
            "role": "system",
            "content": "[CONTEXT CONSOLIDATED] Previous messages have been summarized to reduce context size.",
        }

        return {
            "consolidated_history": summary,
            "messages": [
                consolidation_message
            ],  # Note: This replaces via special handling
            "context_size": new_context_size,
        }

    except Exception as e:
        logger.error(f"Consolidation error: {e}")

        # On error, just note it and continue
        return {
            "messages": [
                {"role": "system", "content": f"[CONSOLIDATION FAILED] {str(e)}"}
            ]
        }


def should_consolidate(state: AgentStateDict, threshold: int = 80000) -> bool:
    """
    Check if consolidation is needed based on context size.

    Args:
        state: Current agent state.
        threshold: Token threshold for triggering consolidation.

    Returns:
        True if context size exceeds threshold.
    """
    current_size = state.get("context_size", 0)
    if current_size == 0:
        current_size = calculate_context_size(state)
    return current_size > threshold


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    from agent_state import create_initial_state

    # Create a state with many messages
    state = create_initial_state("Test task")

    # Add some fake messages
    for i in range(10):
        state["messages"].append(
            {
                "role": "assistant",
                "content": f"This is message {i} with some content to consolidate. "
                * 50,
            }
        )

    state["context_size"] = calculate_context_size(state)
    print(
        f"Before consolidation: {state['context_size']} tokens, {len(state['messages'])} messages"
    )

    result = consolidator_node(state)
    print(f"\nConsolidation result:")
    print(f"  New context size: {result.get('context_size', 'N/A')}")
    print(f"  Summary preview: {result.get('consolidated_history', '')[:200]}...")
