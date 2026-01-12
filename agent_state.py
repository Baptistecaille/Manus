"""
AgentState - LangGraph state definition for the Manus agent.

Implements append-only message history and structured state tracking
for autonomous task execution with context management.
"""

import operator
from typing import Annotated, Optional


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses a rough approximation of 1 token ≈ 4 characters for
    mixed French/English text. This is conservative and may
    overestimate for code or underestimate for pure French.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    if not text:
        return 0
    return len(text) // 4


class AgentState:
    """
    TypedDict-style state for the Manus agent.

    This class uses LangGraph's Annotated pattern for append-only
    message history while maintaining mutable state for other fields.

    Attributes:
        messages: Append-only list of message dicts (role, content).
        todo_list: Current task status in format "✅ Done: X | 🔲 Next: Y".
        internal_monologue: Agent's last reasoning/reflection.
        seedbox_manifest: List of files in the sandbox /workspace.
        last_tool_output: Raw output from the last tool execution.
        iteration_count: Counter for anti-infinite-loop protection.
        context_size: Estimated tokens used in context.
        consolidated_history: Summary of past steps (for context compression).
        current_action: The action type extracted from planner response.
        action_details: The action parameters extracted from planner response.
    """

    pass


# LangGraph TypedDict definition
from typing import TypedDict


class AgentStateDict(TypedDict):
    """LangGraph-compatible state dictionary with extended HITL support."""

    # ═══════════════════════════════════════
    # SECTION 1: CORE MESSAGE HANDLING
    # ═══════════════════════════════════════

    # Append-only message history using operator.add reducer
    messages: Annotated[list[dict], operator.add]

    # ═══════════════════════════════════════
    # SECTION 2: QUERY & INTENT
    # ═══════════════════════════════════════

    original_query: str  # Original user request
    enhanced_query: str  # Optimized query from Prompt Enhancer
    detected_intent: str  # code_generation|web_research|data_analysis|file_manipulation|mixed_workflow
    confidence_score: float  # 0.0 - 1.0 intent confidence
    intent_reasoning: str  # LLM's explanation of intent detection

    # ═══════════════════════════════════════
    # SECTION 3: WORKSPACE CONTEXT
    # ═══════════════════════════════════════

    workspace_context: (
        dict  # {current_directory, file_tree, git_status, available_tools}
    )
    technical_constraints: (
        dict  # {max_bash_execution_time, allowed_file_extensions, forbidden_commands}
    )

    # ═══════════════════════════════════════
    # SECTION 4: TASK TRACKING (existing)
    # ═══════════════════════════════════════

    todo_list: str
    internal_monologue: str

    # ═══════════════════════════════════════
    # SECTION 5: SEEDBOX STATE (existing)
    # ═══════════════════════════════════════

    seedbox_manifest: list[str]
    last_tool_output: str

    # ═══════════════════════════════════════
    # SECTION 6: EXECUTION PLANNING
    # ═══════════════════════════════════════

    execution_plan: dict  # {steps: list, total_estimated_duration, dependencies}
    plan_validation_status: str  # pending|approved|rejected|modified
    plan_modification_requests: list  # User feedback on plan

    # ═══════════════════════════════════════
    # SECTION 7: CURRENT EXECUTION
    # ═══════════════════════════════════════

    current_step_id: str
    executor_outputs: list  # Historique of all executions
    current_action: str  # Action type from planner
    action_details: str  # Action parameters from planner

    # ═══════════════════════════════════════
    # SECTION 8: BASH VALIDATION
    # ═══════════════════════════════════════

    pending_bash_commands: list  # [{command, risk_level, justification}]
    bash_validation_status: str  # pending|approved|rejected|skipped

    # ═══════════════════════════════════════
    # SECTION 9: HUMAN-IN-THE-LOOP
    # ═══════════════════════════════════════

    hitl_mode: str  # strict|moderate|minimal
    current_breakpoint: str  # Current breakpoint name or empty
    human_interventions: list  # [{timestamp, stage, action, feedback}]
    awaiting_human_input: bool  # True when blocked on user

    # ═══════════════════════════════════════
    # SECTION 10: RISK MANAGEMENT
    # ═══════════════════════════════════════

    global_risk_level: str  # low|medium|high|critical
    risk_factors: list  # [{type, severity, description}]
    safety_checks_passed: bool

    # ═══════════════════════════════════════
    # SECTION 11: CONTROL FLOW (existing + extended)
    # ═══════════════════════════════════════

    iteration_count: int
    max_iterations: int
    context_size: int
    consolidated_history: str
    error_log: list  # [{timestamp, error, context}]
    execution_status: str  # running|paused|completed|failed

    # ═══════════════════════════════════════
    # SECTION 12: DEEP RESEARCH OUTPUT
    # ═══════════════════════════════════════

    final_report: str


def create_initial_state(
    user_query: str, hitl_mode: str = "moderate"
) -> AgentStateDict:
    """
    Create the initial state for a new agent run.

    Args:
        user_query: The user's initial request/task.
        hitl_mode: Human-in-the-loop mode ('strict', 'moderate', 'minimal').

    Returns:
        Initialized AgentStateDict ready for graph execution.
    """
    import os

    initial_message = {"role": "user", "content": user_query}

    initial_todo = (
        f"🔲 Pending: {user_query[:100]}{'...' if len(user_query) > 100 else ''}"
    )

    # Default technical constraints
    default_constraints = {
        "max_bash_execution_time": int(os.getenv("MAX_BASH_TIMEOUT", "300")),
        "allowed_file_extensions": ["*"],
        "forbidden_commands": ["rm -rf /", "dd if=/dev/zero", "mkfs", ":(){ :|:& };:"],
    }

    return AgentStateDict(
        # Core messages
        messages=[initial_message],
        # Query & Intent
        original_query=user_query,
        enhanced_query="",  # Set by prompt_enhancer
        detected_intent="",  # Set by prompt_enhancer
        confidence_score=0.0,
        intent_reasoning="",
        # Workspace context
        workspace_context={},  # Set by prompt_enhancer
        technical_constraints=default_constraints,
        # Task tracking
        todo_list=initial_todo,
        internal_monologue="",
        # Seedbox state
        seedbox_manifest=[],
        last_tool_output="",
        # Execution planning
        execution_plan={},
        plan_validation_status="pending",
        plan_modification_requests=[],
        # Current execution
        current_step_id="",
        executor_outputs=[],
        current_action="",
        action_details="",
        # Bash validation
        pending_bash_commands=[],
        bash_validation_status="pending",
        # Human-in-the-loop
        hitl_mode=hitl_mode,
        current_breakpoint="",
        human_interventions=[],
        awaiting_human_input=False,
        # Risk management
        global_risk_level="low",
        risk_factors=[],
        safety_checks_passed=True,
        # Control flow
        iteration_count=0,
        max_iterations=int(os.getenv("MAX_ITERATIONS", "30")),
        context_size=estimate_tokens(user_query),
        consolidated_history="",
        error_log=[],
        execution_status="running",
        # Deep research output
        final_report="",
    )


def calculate_context_size(state: AgentStateDict) -> int:
    """
    Calculate total estimated context size for the current state.

    Args:
        state: The current agent state.

    Returns:
        Total estimated tokens across all state fields.
    """
    total = 0

    # Messages
    for msg in state.get("messages", []):
        total += estimate_tokens(str(msg.get("content", "")))

    # Other fields
    total += estimate_tokens(state.get("todo_list", ""))
    total += estimate_tokens(state.get("internal_monologue", ""))
    total += estimate_tokens(state.get("last_tool_output", ""))
    total += estimate_tokens(state.get("consolidated_history", ""))
    total += estimate_tokens("\n".join(state.get("seedbox_manifest", [])))

    return total


if __name__ == "__main__":
    # Quick test
    state = create_initial_state("Create a Python script that prints Hello World")
    print(f"Initial state created:")
    print(f"  Messages: {len(state['messages'])}")
    print(f"  Todo: {state['todo_list']}")
    print(f"  Context size: {state['context_size']} tokens")
