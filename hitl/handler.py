"""
HITL Handler - LangGraph node for human-in-the-loop interactions.

This node handles breakpoints by displaying the appropriate interface
and updating state based on user decisions.
"""

import logging
from datetime import datetime
from typing import Optional

from agent_state import AgentStateDict
from hitl.breakpoints import (
    BreakpointType,
    BreakpointConfig,
    BreakpointResult,
    BREAKPOINT_CONFIGS,
    should_trigger_breakpoint,
)
from hitl.cli_interface import display_breakpoint, get_user_decision

logger = logging.getLogger(__name__)


def hitl_handler_node(state: AgentStateDict) -> dict:
    """
    LangGraph node that handles human-in-the-loop breakpoints.

    This node:
    1. Checks if there's a pending breakpoint
    2. Displays the appropriate validation interface
    3. Captures user decision
    4. Updates state based on the decision

    Args:
        state: Current agent state with current_breakpoint set.

    Returns:
        Dict of state updates based on user decision.
    """
    breakpoint_name = state.get("current_breakpoint", "")

    if not breakpoint_name:
        logger.debug("No breakpoint pending, skipping HITL handler")
        return {"awaiting_human_input": False}

    logger.info(f"HITL Handler - Processing breakpoint: {breakpoint_name}")

    # Map breakpoint name to type
    breakpoint_type = None
    for bp_type in BreakpointType:
        if bp_type.value == breakpoint_name:
            breakpoint_type = bp_type
            break

    if not breakpoint_type:
        logger.warning(f"Unknown breakpoint type: {breakpoint_name}")
        return {
            "current_breakpoint": "",
            "awaiting_human_input": False,
        }

    # Get breakpoint config
    config = BREAKPOINT_CONFIGS.get(breakpoint_type)
    if not config:
        logger.warning(f"No config found for breakpoint: {breakpoint_type}")
        return {
            "current_breakpoint": "",
            "awaiting_human_input": False,
        }

    # Check if we should actually trigger based on mode/risk
    hitl_mode = state.get("hitl_mode", "moderate")
    risk_level = state.get("global_risk_level", "low")

    # For bash commands, get the command from pending
    command = ""
    justification = ""
    if breakpoint_type == BreakpointType.BASH_COMMAND:
        pending_commands = state.get("pending_bash_commands", [])
        if pending_commands:
            cmd_info = pending_commands[0]
            command = cmd_info.get("command", "")
            justification = cmd_info.get("justification", "")

    # Check if breakpoint should trigger
    if not should_trigger_breakpoint(breakpoint_type, hitl_mode, risk_level, command):
        logger.info(f"Breakpoint {breakpoint_name} skipped due to mode/risk settings")
        return {
            "current_breakpoint": "",
            "awaiting_human_input": False,
            "bash_validation_status": (
                "approved"
                if breakpoint_type == BreakpointType.BASH_COMMAND
                else state.get("bash_validation_status", "pending")
            ),
            "plan_validation_status": (
                "approved"
                if breakpoint_type == BreakpointType.PLAN_VALIDATION
                else state.get("plan_validation_status", "pending")
            ),
        }

    # Display the breakpoint interface
    display_breakpoint(
        breakpoint_type,
        state,
        command=command,
        risk_level=risk_level,
        justification=justification,
    )

    # Get user decision
    result = get_user_decision(config, timeout_enabled=True)

    # Log the intervention
    intervention = {
        "timestamp": datetime.now().isoformat(),
        "stage": breakpoint_name,
        "action": result.action,
        "feedback": result.feedback,
        "response_time": result.response_time_seconds,
        "timed_out": result.timed_out,
    }

    current_interventions = state.get("human_interventions", [])
    new_interventions = current_interventions + [intervention]

    # Process the decision
    return _process_decision(breakpoint_type, result, state, new_interventions)


def _process_decision(
    breakpoint_type: BreakpointType,
    result: BreakpointResult,
    state: AgentStateDict,
    interventions: list,
) -> dict:
    """
    Process the user's decision and return state updates.

    Args:
        breakpoint_type: Type of breakpoint.
        result: User's decision result.
        state: Current agent state.
        interventions: Updated list of interventions.

    Returns:
        Dict of state updates.
    """
    base_updates = {
        "human_interventions": interventions,
        "awaiting_human_input": False,
        "current_breakpoint": "",
    }

    # Handle quit
    if result.action == "quit":
        logger.info("User chose to quit")
        return {
            **base_updates,
            "execution_status": "failed",
            "current_action": "complete",
            "messages": [
                {
                    "role": "system",
                    "content": "[HITL] Exécution arrêtée par l'utilisateur",
                }
            ],
        }

    # Handle rejection
    if result.action == "reject":
        logger.info(f"User rejected at {breakpoint_type.value}")

        if breakpoint_type == BreakpointType.ENHANCED_PROMPT:
            return {
                **base_updates,
                "enhanced_query": state.get("original_query", ""),  # Reset to original
                "plan_validation_status": "rejected",
                "messages": [
                    {
                        "role": "system",
                        "content": f"[HITL] Requête optimisée rejetée. Feedback: {result.feedback}",
                    }
                ],
            }
        elif breakpoint_type == BreakpointType.PLAN_VALIDATION:
            return {
                **base_updates,
                "plan_validation_status": "rejected",
                "messages": [
                    {
                        "role": "system",
                        "content": f"[HITL] Plan d'exécution rejeté. Feedback: {result.feedback}",
                    }
                ],
            }
        elif breakpoint_type == BreakpointType.BASH_COMMAND:
            return {
                **base_updates,
                "bash_validation_status": "rejected",
                "pending_bash_commands": [],  # Clear pending commands
                "messages": [
                    {
                        "role": "system",
                        "content": f"[HITL] Commande bash rejetée. Feedback: {result.feedback}",
                    }
                ],
            }

    # Handle modification
    if result.action == "modify":
        logger.info(f"User modified at {breakpoint_type.value}")

        if breakpoint_type == BreakpointType.ENHANCED_PROMPT:
            # User provided new query
            new_query = result.modifications.get(
                "user_feedback", state.get("enhanced_query", "")
            )
            return {
                **base_updates,
                "enhanced_query": new_query,
                "plan_validation_status": "modified",
                "plan_modification_requests": state.get(
                    "plan_modification_requests", []
                )
                + [result.feedback],
                "messages": [
                    {
                        "role": "system",
                        "content": f"[HITL] Requête modifiée par l'utilisateur: {new_query[:100]}...",
                    }
                ],
            }
        elif breakpoint_type == BreakpointType.BASH_COMMAND:
            # User modified command
            new_command = result.modifications.get("user_feedback", "")
            pending = state.get("pending_bash_commands", [])
            if pending and new_command:
                pending[0]["command"] = new_command
            return {
                **base_updates,
                "pending_bash_commands": pending,
                "bash_validation_status": "approved",  # User modified and approved
                "messages": [
                    {
                        "role": "system",
                        "content": f"[HITL] Commande modifiée: {new_command[:100]}",
                    }
                ],
            }

    # Handle approval (default)
    logger.info(f"User approved at {breakpoint_type.value}")

    if breakpoint_type == BreakpointType.ENHANCED_PROMPT:
        return {
            **base_updates,
            "plan_validation_status": "approved",
            "messages": [
                {
                    "role": "system",
                    "content": "[HITL] Requête optimisée approuvée",
                }
            ],
        }
    elif breakpoint_type == BreakpointType.PLAN_VALIDATION:
        return {
            **base_updates,
            "plan_validation_status": "approved",
            "messages": [
                {
                    "role": "system",
                    "content": "[HITL] Plan d'exécution approuvé",
                }
            ],
        }
    elif breakpoint_type == BreakpointType.BASH_COMMAND:
        return {
            **base_updates,
            "bash_validation_status": "approved",
            "messages": [
                {
                    "role": "system",
                    "content": "[HITL] Commande bash approuvée",
                }
            ],
        }

    return base_updates


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from agent_state import create_initial_state

    # Create test state with a breakpoint
    state = create_initial_state("Supprime les fichiers temporaires")
    state["enhanced_query"] = (
        "Nettoyer les fichiers temporaires de plus de 7 jours dans /tmp"
    )
    state["detected_intent"] = "file_manipulation"
    state["confidence_score"] = 0.85
    state["global_risk_level"] = "medium"
    state["current_breakpoint"] = "enhanced_prompt_validation"
    state["awaiting_human_input"] = True

    print("\n=== Testing HITL Handler ===\n")

    result = hitl_handler_node(state)

    print("\n=== Result ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
