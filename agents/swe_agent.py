import logging
from typing import TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable

from agent_state import AgentStateDict
from nodes.swe.planner import swe_planner_node
from nodes.bash_executor import bash_executor_node
from nodes.editor_executor import editor_executor_node

logger = logging.getLogger(__name__)


def route_swe_actions(state: AgentStateDict) -> Literal["bash", "edit", "__end__"]:
    """Determine next step based on planner output."""
    action = state.get("current_action", "complete").lower()

    if action == "bash":
        return "bash"
    elif action == "edit":
        return "edit"
    else:
        return "__end__"


def create_swe_graph() -> Runnable:
    """Build the SWE Agent Subgraph."""
    workflow = StateGraph(AgentStateDict)

    # Add Nodes
    workflow.add_node("swe_planner", swe_planner_node)
    workflow.add_node("bash_executor", bash_executor_node)
    workflow.add_node("editor_executor", editor_executor_node)

    # Set Entry Point
    workflow.set_entry_point("swe_planner")

    # Conditional Edges from Planner
    workflow.add_conditional_edges(
        "swe_planner",
        route_swe_actions,
        {"bash": "bash_executor", "edit": "editor_executor", "__end__": END},
    )

    # Executors return to Planner
    workflow.add_edge("bash_executor", "swe_planner")
    workflow.add_edge("editor_executor", "swe_planner")

    return workflow.compile()
