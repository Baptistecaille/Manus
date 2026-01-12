"""
Router - Navigation logic for the Manus agent graph.

Determines which node to execute next based on the current state,
including safety checks for iteration limits, context size, and HITL handling.
"""

import logging
import os

from agent_state import AgentStateDict

logger = logging.getLogger(__name__)

# Default limits (can be overridden via environment)
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))
CONSOLIDATION_THRESHOLD = int(os.getenv("CONSOLIDATION_THRESHOLD", "80000"))

# Sensitive keywords that trigger bash validation
BASH_SENSITIVE_KEYWORDS = [
    "rm",
    "rmdir",
    "dd",
    "mkfs",
    "sudo",
    "chmod",
    "chown",
    "curl | bash",
    "wget | bash",
    "eval",
    "exec",
    "/etc/",
    "/sys/",
    "/proc/",
    "kill",
    "pkill",
]


def contains_sensitive_keyword(command: str) -> bool:
    """Check if a command contains sensitive keywords."""
    command_lower = command.lower()
    for keyword in BASH_SENSITIVE_KEYWORDS:
        if keyword.lower() in command_lower:
            return True
    return False


def router(state: AgentStateDict) -> str:
    """
    Determine the next node to execute based on current state.

    Decision logic:
    1. Safety: If iteration_count > MAX_ITERATIONS → "end"
    2. Completion: If current_action == "complete" → "end"
    3. Context management: If needs consolidation → "consolidator"
    4. Action routing: Route to appropriate executor node
    5. Default: Return to planner

    Args:
        state: Current agent state.

    Returns:
        Name of the next node to execute:
        "planner", "bash_executor", "consolidator", "search", "playwright", or "end"
    """
    iteration_count = state.get("iteration_count", 0)
    current_action = state.get("current_action", "").lower().strip()
    context_size = state.get("context_size", 0)

    logger.debug(
        f"Router - iteration: {iteration_count}, action: {current_action}, context: {context_size}"
    )

    # Safety check: prevent infinite loops
    if iteration_count >= MAX_ITERATIONS:
        logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached, forcing end")
        return "end"

    # Check for completion
    if current_action == "complete":
        logger.info("Task marked complete by planner")
        return "end"

    # Check for explicit consolidation request or automatic trigger
    if current_action == "consolidate" or context_size > CONSOLIDATION_THRESHOLD:
        logger.info(
            f"Consolidation triggered (action={current_action}, context={context_size})"
        )
        return "consolidator"

    # Route to appropriate executor
    if current_action == "bash":
        return "bash_executor"

    if current_action == "search":
        return "search"

    if current_action == "playwright":
        return "playwright"

    if current_action == "deep_research":
        return "deep_research"

    # Phase 1 actions
    if current_action == "browser":
        return "browser_executor"

    if current_action == "crawl":
        return "crawl_executor"

    if current_action == "edit":
        return "editor_executor"

    # Phase 2 actions
    if current_action == "plan":
        return "planning_executor"

    if current_action == "ask":
        return "ask_human_executor"

    # Default: return to planner for next decision
    # This handles cases where action is empty or unrecognized
    if current_action:
        logger.warning(f"Unknown action '{current_action}', returning to planner")

    return "planner"


def hitl_router(state: AgentStateDict) -> str:
    """
    Route from HITL handler based on the breakpoint result.

    Decision logic:
    1. If execution_status == "failed" → "end"
    2. If plan_validation_status == "rejected" → "end" or re-plan
    3. If bash_validation_status == "rejected" → back to planner
    4. If bash_validation_status == "approved" → execute bash
    5. Default: proceed to planner

    Args:
        state: Current agent state after HITL handling.

    Returns:
        Name of the next node.
    """
    execution_status = state.get("execution_status", "running")
    current_breakpoint = state.get("current_breakpoint", "")

    # Check if user quit
    if execution_status == "failed":
        logger.info("HITL Router - Execution failed/stopped by user")
        return "end"

    # Route based on which breakpoint just completed
    if not current_breakpoint:
        # Breakpoint was cleared - check validation statuses

        # Check bash validation
        bash_status = state.get("bash_validation_status", "pending")
        if bash_status == "approved":
            logger.info("HITL Router - Bash command approved, executing")
            return "bash_executor"
        elif bash_status == "rejected":
            logger.info("HITL Router - Bash command rejected, returning to planner")
            return "planner"

        # Check plan validation
        plan_status = state.get("plan_validation_status", "pending")
        if plan_status == "rejected":
            logger.info("HITL Router - Plan rejected, ending")
            return "end"

        # Default: proceed to planner
        return "planner"

    # If still at a breakpoint (shouldn't happen normally)
    logger.warning(f"HITL Router - Still at breakpoint {current_breakpoint}")
    return "planner"


def should_validate_bash(state: AgentStateDict) -> bool:
    """
    Determine if a bash command needs HITL validation.

    Args:
        state: Current agent state.

    Returns:
        True if bash validation breakpoint should trigger.
    """
    hitl_mode = state.get("hitl_mode", "moderate")
    risk_level = state.get("global_risk_level", "low")
    command = state.get("action_details", "")

    # In strict mode, always validate
    if hitl_mode == "strict":
        return True

    # In minimal mode, only validate critical risks
    if hitl_mode == "minimal":
        return risk_level == "critical" or contains_sensitive_keyword(command)

    # Moderate mode: validate medium+ risk or sensitive commands
    if risk_level in ["medium", "high", "critical"]:
        return True

    if contains_sensitive_keyword(command):
        return True

    return False


def get_next_node_description(node_name: str) -> str:
    """
    Get a human-readable description of what a node does.

    Args:
        node_name: Name of the node.

    Returns:
        Description string.
    """
    descriptions = {
        "prompt_enhancer": "Analyzing and optimizing query...",
        "hitl_handler": "Waiting for human validation...",
        "planner": "Planning next action...",
        "bash_executor": "Executing bash command...",
        "consolidator": "Compressing context...",
        "search": "Searching the web...",
        "playwright": "Browsing webpage...",
        "deep_research": "Conducting deep research...",
        "pre_bash_validator": "Validating bash command...",
        "post_bash_validator": "Processing validation result...",
        # Phase 1 & 2 nodes
        "browser_executor": "Automating browser task...",
        "crawl_executor": "Crawling website...",
        "editor_executor": "Editing file...",
        "planning_executor": "Generating action plan...",
        "ask_human_executor": "Requesting user input...",
        "end": "Task completed",
    }
    return descriptions.get(node_name, f"Executing {node_name}...")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)

    from agent_state import create_initial_state

    # Test various scenarios
    test_cases = [
        {"current_action": "bash", "expected": "bash_executor"},
        {"current_action": "complete", "expected": "end"},
        {"current_action": "consolidate", "expected": "consolidator"},
        {"current_action": "", "expected": "planner"},
        {"iteration_count": 31, "expected": "end"},
        {"context_size": 100000, "expected": "consolidator"},
    ]

    for case in test_cases:
        state = create_initial_state("Test")
        state.update(case)
        expected = case.pop("expected")
        result = router(state)
        status = "✓" if result == expected else "✗"
        print(f"{status} {case} → {result} (expected: {expected})")

    print("\n--- HITL Router Tests ---")

    hitl_test_cases = [
        {"execution_status": "failed", "expected": "end"},
        {"bash_validation_status": "approved", "expected": "bash_executor"},
        {"bash_validation_status": "rejected", "expected": "planner"},
        {"plan_validation_status": "rejected", "expected": "end"},
    ]

    for case in hitl_test_cases:
        state = create_initial_state("Test")
        state["current_breakpoint"] = ""  # Breakpoint cleared
        state.update(case)
        expected = case.pop("expected")
        result = hitl_router(state)
        status = "✓" if result == expected else "✗"
        print(f"{status} {case} → {result} (expected: {expected})")
