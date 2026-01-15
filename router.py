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


# ═══════════════════════════════════════════════════════════════════════════════
# 2-ACTION RULE (NEW - $2B Pattern)
# ═══════════════════════════════════════════════════════════════════════════════

# Actions that count toward the 2-Action Rule for findings save
RESEARCH_ACTIONS = {"search", "view", "browser", "crawl", "deep_research"}


class EnhancedRouter:
    """
    Router with 2-Action Rule integration.

    After every 2 research actions (search, view, browser, crawl),
    forces a findings save to disk to prevent information loss.

    This is part of the "$2B pattern" - persisting discoveries
    before they get lost in context.

    Example:
        >>> router = EnhancedRouter()
        >>> next_node, should_save = await router.route("search", state)
        >>> if should_save:
        ...     await planning_manager.save_findings(...)
    """

    def __init__(self):
        """Initialize the enhanced router."""
        self._action_counter = 0
        self._pending_discoveries: list[str] = []

    def should_save_findings(self, action: str, state: dict) -> bool:
        """
        Check if findings should be saved (2-Action Rule).

        Args:
            action: Current action type.
            state: Agent state with actions_since_save.

        Returns:
            True if findings should be saved.
        """
        if action.lower() not in RESEARCH_ACTIONS:
            return False

        actions_since_save = state.get("actions_since_save", 0)
        return actions_since_save >= 2

    def get_state_updates_after_action(self, action: str) -> dict:
        """
        Get state updates to apply after executing an action.

        Increments counters for planning manager integration.

        Args:
            action: The action that was just executed.

        Returns:
            Dict of state updates.
        """
        updates = {
            "actions_since_refresh": (lambda s: s.get("actions_since_refresh", 0) + 1),
        }

        if action.lower() in RESEARCH_ACTIONS:
            updates["actions_since_save"] = lambda s: s.get("actions_since_save", 0) + 1

        return updates

    def reset_save_counter(self) -> dict:
        """
        Reset the save counter after findings are saved.

        Returns:
            State update to reset counter.
        """
        return {"actions_since_save": 0}


# Global instance for convenience
_enhanced_router = EnhancedRouter()


# ═══════════════════════════════════════════════════════════════════════════════
# BROWSER DETECTION KEYWORDS (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

BROWSER_KEYWORDS = [
    "navigate",
    "browse",
    "visit",
    "go to",
    "open page",
    "open url",
    "website",
    "webpage",
    "web page",
    "click",
    "button",
    "form",
    "fill",
    "submit",
    "login",
    "sign in",
    "screenshot",
    "scrape",
    "extract from page",
    "web",
    "download from",
    "fetch page",
    "load page",
]

DEEP_RESEARCH_KEYWORDS = [
    "research",
    "analyze",
    "summarize multiple",
    "comprehensive",
    "in-depth",
    "detailed analysis",
    "compare sources",
    "literature review",
    "state of the art",
    "survey",
]


def should_use_browser(query: str, context: dict = None) -> bool:
    """
    Determine if browser automation is needed for the query.

    Args:
        query: User query or task description.
        context: Optional context dict with additional info.

    Returns:
        True if browser automation is recommended.
    """
    query_lower = query.lower()

    # Check for browser keywords
    for keyword in BROWSER_KEYWORDS:
        if keyword in query_lower:
            return True

    # Check for URL patterns
    if "http://" in query_lower or "https://" in query_lower:
        return True

    # Check context if provided
    if context:
        detected_intent = context.get("detected_intent", "")
        if detected_intent in ("web_interaction", "web_scraping"):
            return True

    return False


def should_use_deep_research(query: str, context: dict = None) -> bool:
    """
    Determine if deep research is appropriate for the query.

    Args:
        query: User query or task description.
        context: Optional context dict.

    Returns:
        True if deep research is recommended.
    """
    query_lower = query.lower()

    for keyword in DEEP_RESEARCH_KEYWORDS:
        if keyword in query_lower:
            return True

    # Check context
    if context:
        detected_intent = context.get("detected_intent", "")
        if detected_intent == "web_research":
            return True

    return False


def select_optimal_executor(
    task: str,
    available_executors: list[str],
    context: dict = None,
) -> str:
    """
    Select the best executor for a given task.

    Uses keyword matching and context to score executors.

    Args:
        task: Task description.
        available_executors: List of available executor names.
        context: Optional context dict.

    Returns:
        Name of the recommended executor.
    """
    if not available_executors:
        return "planner"

    task_lower = task.lower()
    scores: dict[str, int] = {exe: 0 for exe in available_executors}

    # Browser executor scoring
    if "browser_executor" in scores:
        if should_use_browser(task, context):
            scores["browser_executor"] += 10

    # Deep research scoring
    if "deep_research" in scores:
        if should_use_deep_research(task, context):
            scores["deep_research"] += 10

    # Bash executor scoring
    if "bash_executor" in scores:
        bash_keywords = [
            "run",
            "execute",
            "script",
            "command",
            "install",
            "create file",
        ]
        for kw in bash_keywords:
            if kw in task_lower:
                scores["bash_executor"] += 3

    # Editor executor scoring
    if "editor_executor" in scores:
        edit_keywords = ["edit", "modify", "update file", "change", "fix"]
        for kw in edit_keywords:
            if kw in task_lower:
                scores["editor_executor"] += 3

    # Crawl executor scoring
    if "crawl_executor" in scores:
        crawl_keywords = ["crawl", "scrape", "extract data"]
        for kw in crawl_keywords:
            if kw in task_lower:
                scores["crawl_executor"] += 5

    # Document executor scoring
    if "document_executor" in scores:
        doc_keywords = ["write report", "create document", "docx", "word document"]
        for kw in doc_keywords:
            if kw in task_lower:
                scores["document_executor"] += 10

    # Data analysis executor scoring
    if "data_analysis_executor" in scores:
        data_keywords = ["analyze data", "pandas", "csv", "summary statistics", "plot"]
        for kw in data_keywords:
            if kw in task_lower:
                scores["data_analysis_executor"] += 10

    # Return highest scoring executor
    best_executor = max(scores, key=lambda k: scores[k])

    # Default to planner if no good match
    if scores[best_executor] == 0:
        return "planner"

    logger.debug(
        f"Executor selection: {task[:50]} → {best_executor} (scores: {scores})"
    )
    return best_executor


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

    if current_action == "document":
        return "document_executor"

    if current_action == "data_analysis":
        return "data_analysis_executor"

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
        "document_executor": "Creating document...",
        "data_analysis_executor": "Analyzing data...",
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
