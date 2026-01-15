"""
Agent Graph - LangGraph state graph assembly for the Manus agent.

Assembles all nodes and edges into a compiled graph ready for execution.
Includes Prompt Enhancer, HITL handler, and checkpointing support.
"""

import logging
import os
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent_state import AgentStateDict
from router import router, hitl_router
from nodes.planner import planner_node
from nodes.bash_executor import bash_executor_node
from nodes.consolidator import consolidator_node
from nodes.prompt_enhancer import prompt_enhancer_node
from hitl.handler import hitl_handler_node
from deep_research_config import DeepResearchConfig

# Phase 1 & 2 nodes
from nodes.browser_executor import browser_executor_node
from nodes.crawl_executor import crawl_executor_node
from nodes.editor_executor import editor_executor_node
from nodes.planning_executor import planning_executor_node
from nodes.ask_human_executor import ask_human_executor_node
from nodes.document_executor import document_executor_node
from nodes.data_analysis_executor import data_analysis_executor_node

# Planning Manager ($2B Pattern)
from nodes.planning_manager import (
    planning_manager_node,
    refresh_plan_node,
    should_refresh_plan,
)

# Memory Manager
from nodes.memory_manager import memory_node, memory_node_sync

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER NODES
# ═══════════════════════════════════════════════════════════════════════════════


def _placeholder_search_node(state: AgentStateDict) -> dict:
    """
    Placeholder for search node (Phase 3).

    TODO: Implement DuckDuckGo search integration.
    """
    logger.info("Search node called (placeholder)")
    return {
        "last_tool_output": "Search functionality not yet implemented. Use bash with curl instead.",
        "messages": [
            {
                "role": "system",
                "content": "[SEARCH] Not implemented yet. Try using 'curl' in bash instead.",
            }
        ],
    }


def _placeholder_playwright_node(state: AgentStateDict) -> dict:
    """
    Placeholder for playwright node (Phase 3).

    TODO: Implement Playwright web scraping.
    """
    logger.info("Playwright node called (placeholder)")
    return {
        "last_tool_output": "Playwright functionality not yet implemented. Use bash with curl instead.",
        "messages": [
            {
                "role": "system",
                "content": "[PLAYWRIGHT] Not implemented yet. Try using 'curl' in bash instead.",
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-EXECUTION VALIDATION NODE
# ═══════════════════════════════════════════════════════════════════════════════


def pre_bash_validator(state: AgentStateDict) -> dict:
    """
    Pre-execution validator for bash commands.

    Sets up the pending bash command for HITL validation if needed.
    """
    command = state.get("action_details", "").strip()
    risk_level = state.get("global_risk_level", "low")

    # Create pending command entry
    pending_command = {
        "command": command,
        "risk_level": risk_level,
        "justification": state.get("internal_monologue", ""),
    }

    return {
        "pending_bash_commands": [pending_command],
        "current_breakpoint": "bash_command_validation",
        "awaiting_human_input": True,
    }


def post_bash_validator(state: AgentStateDict) -> dict:
    """
    Post-validation handler for bash commands.

    Checks if command was approved and proceeds or skips accordingly.
    """
    validation_status = state.get("bash_validation_status", "pending")

    if validation_status == "rejected":
        return {
            "last_tool_output": "Command rejected by user",
            "pending_bash_commands": [],
            "messages": [
                {
                    "role": "system",
                    "content": "[BASH] Command rejected by user. Returning to planner.",
                }
            ],
        }

    # Approved or skipped - clear pending and proceed
    return {
        "pending_bash_commands": [],
        "bash_validation_status": "pending",  # Reset for next command
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GRAPH CREATION
# ═══════════════════════════════════════════════════════════════════════════════


def create_agent_graph(
    enable_deep_research: bool = True,
    enable_hitl: bool = True,
    enable_prompt_enhancer: bool = True,
) -> StateGraph:
    """
    Create and compile the Manus agent state graph.

    Graph structure (with HITL enabled):
    - Entry: prompt_enhancer → hitl_handler (validation) → planner
    - Conditional edges from planner based on router decision
    - Bash commands: pre_validator → hitl_handler → bash_executor
    - All executor nodes return to planner
    - END when router returns "end"

    Args:
        enable_deep_research: If True, include deep research sub-graph.
        enable_hitl: If True, include HITL breakpoints.
        enable_prompt_enhancer: If True, include prompt enhancement.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    logger.info("Creating agent graph...")

    # Create the state graph
    workflow = StateGraph(AgentStateDict)

    # ═══════════════════════════════════════════════════════════════════
    # Add core nodes
    # ═══════════════════════════════════════════════════════════════════

    if enable_prompt_enhancer:
        workflow.add_node("prompt_enhancer", prompt_enhancer_node)

    if enable_hitl:
        workflow.add_node("hitl_handler", hitl_handler_node)

    # Planning Manager nodes ($2B Pattern - anti goal-drift)
    workflow.add_node("initialize_plan", planning_manager_node)
    workflow.add_node("refresh_plan", refresh_plan_node)

    # Memory Node
    workflow.add_node("memory", memory_node_sync)

    workflow.add_node("planner", planner_node)
    workflow.add_node("bash_executor", bash_executor_node)
    workflow.add_node("consolidator", consolidator_node)
    workflow.add_node("search", _placeholder_search_node)
    workflow.add_node("playwright", _placeholder_playwright_node)

    # Phase 1 & 2 nodes
    workflow.add_node("browser_executor", browser_executor_node)
    workflow.add_node("crawl_executor", crawl_executor_node)
    workflow.add_node("editor_executor", editor_executor_node)
    workflow.add_node("planning_executor", planning_executor_node)
    workflow.add_node("ask_human_executor", ask_human_executor_node)

    # New executors
    workflow.add_node("document_executor", document_executor_node)
    workflow.add_node("data_analysis_executor", data_analysis_executor_node)

    # Pre/post bash validation nodes
    if enable_hitl:
        workflow.add_node("pre_bash_validator", pre_bash_validator)
        workflow.add_node("post_bash_validator", post_bash_validator)

    # Add deep research wrapper node if enabled
    if enable_deep_research:
        from deep_research_state import create_deep_research_state
        from deep_research_graph import run_deep_research

        def deep_research_node(state: AgentStateDict) -> dict:
            """
            Wrapper node that runs deep research with topic from action_details.
            """
            # Extract topic from action_details
            topic = state.get("action_details", "").strip()
            if not topic:
                # Fallback: try to extract from the original user query
                messages = state.get("messages", [])
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        # Extract topic after "User Query:" if present
                        if "User Query:" in content:
                            topic = (
                                content.split("User Query:")[-1].split("\n")[0].strip()
                            )
                        else:
                            topic = content[:200]
                        break

            if not topic:
                topic = "General research topic"

            logger.info(f"Deep Research Node - Topic: {topic[:50]}...")

            # Get config
            dr_config = DeepResearchConfig.from_env()

            # Run deep research
            try:
                result_state = run_deep_research(
                    topic=topic, config=dr_config, verbose=True
                )

                # Return updates for main agent state
                return {
                    "last_tool_output": f"Deep research completed. Report generated with {len(result_state.get('findings', []))} sources.",
                    "final_report": result_state.get("final_report", ""),
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"[DEEP RESEARCH COMPLETE]\n"
                            f"Topic: {topic}\n"
                            f"Research cycles: {result_state.get('research_depth', 0)}\n"
                            f"Sources analyzed: {len(result_state.get('findings', []))}\n"
                            f"Report saved to /workspace/",
                        }
                    ],
                }
            except Exception as e:
                logger.error(f"Deep research failed: {e}")
                return {
                    "last_tool_output": f"Deep research error: {str(e)}",
                    "messages": [
                        {"role": "system", "content": f"[DEEP RESEARCH ERROR] {str(e)}"}
                    ],
                }

        workflow.add_node("deep_research", deep_research_node)

    # ═══════════════════════════════════════════════════════════════════
    # Set entry point
    # ═══════════════════════════════════════════════════════════════════

    if enable_prompt_enhancer:
        workflow.set_entry_point("prompt_enhancer")
    else:
        workflow.set_entry_point("planner")

    # ═══════════════════════════════════════════════════════════════════
    # Add edges
    # ═══════════════════════════════════════════════════════════════════

    # Adding Specialized Agents
    from agents.swe_agent import create_swe_graph

    swe_graph = create_swe_graph()
    workflow.add_node("swe_agent", swe_graph)

    # Defined intent routing logic
    def intent_router(state: AgentStateDict) -> str:
        """Route to specialized agents based on detected intent."""
        intent = state.get("detected_intent", "mixed_workflow")
        logger.info(f"Routing based on intent: {intent}")

        if intent in ["code_generation", "file_manipulation"]:
            return "swe_agent"

        # Default to general planner
        return "planner"

    if enable_prompt_enhancer and enable_hitl:
        # Prompt enhancer -> Memory -> Initialize plan -> HITL handler (for validation)
        workflow.add_edge("prompt_enhancer", "memory")
        workflow.add_edge("memory", "initialize_plan")
        workflow.add_edge("initialize_plan", "hitl_handler")

        # HITL handler routes based on state
        # We intercept the "planner" route to check for specialized intents first
        def hitl_next_node(state: AgentStateDict) -> str:
            route = hitl_router(state)
            if route == "planner":
                # If going to planner, check if we can specialize
                return intent_router(state)
            return route

        workflow.add_conditional_edges(
            "hitl_handler",
            hitl_next_node,
            {
                "swe_agent": "swe_agent",
                "planner": "planner",
                "bash_executor": "bash_executor",
                "pre_bash_validator": "pre_bash_validator",
                "end": END,
            },
        )
    elif enable_prompt_enhancer:
        # No HITL - prompt enhancer -> memory -> initialize plan -> router
        workflow.add_edge("prompt_enhancer", "memory")
        workflow.add_edge("memory", "initialize_plan")
        workflow.add_conditional_edges(
            "initialize_plan",
            intent_router,
            {"swe_agent": "swe_agent", "planner": "planner"},
        )

    # Refresh plan returns to planner
    workflow.add_edge("refresh_plan", "planner")

    # SWE Agent returns to END (task complete) or Planner (if fallback needed)
    # For now, let's assume if SWE agent finishes, the task is done or needs check.
    # We route successful SWE execution to END.
    workflow.add_edge("swe_agent", END)

    # Build conditional edge mapping from planner
    edge_map = {
        "consolidator": "consolidator",
        "search": "search",
        "playwright": "playwright",
        "planner": "planner",  # For re-planning
        # Phase 1 & 2 actions
        "browser_executor": "browser_executor",
        "crawl_executor": "crawl_executor",
        "editor_executor": "editor_executor",
        "planning_executor": "planning_executor",
        "ask_human_executor": "ask_human_executor",
        "document_executor": "document_executor",
        "data_analysis_executor": "data_analysis_executor",
        "end": END,
    }

    # Bash routing depends on HITL mode
    if enable_hitl:
        edge_map["bash_executor"] = "pre_bash_validator"  # Go through validation first
    else:
        edge_map["bash_executor"] = "bash_executor"

    if enable_deep_research:
        edge_map["deep_research"] = "deep_research"

    # Add conditional routing: planner → (refresh_plan or direct execution)
    # The refresh_plan decision is made BEFORE routing to executors
    def planner_router_with_refresh(state: AgentStateDict) -> str:
        """Route from planner, optionally through refresh_plan first."""
        # Check if we need to refresh the plan first
        if should_refresh_plan(state) == "refresh":
            return "refresh_plan"
        # Otherwise, use normal router
        return router(state)

    # Add refresh_plan to edge map
    edge_map["refresh_plan"] = "refresh_plan"

    # Add conditional edges from planner
    workflow.add_conditional_edges(
        "planner",
        planner_router_with_refresh,
        edge_map,
    )

    # HITL validation flow for bash
    if enable_hitl:
        workflow.add_edge("pre_bash_validator", "hitl_handler")

        # After bash validation, decide what to do
        workflow.add_conditional_edges(
            "post_bash_validator",
            lambda s: (
                "planner"
                if s.get("bash_validation_status") == "rejected"
                else "bash_executor"
            ),
            {
                "planner": "planner",
                "bash_executor": "bash_executor",
            },
        )

    # All executor nodes return to planner
    workflow.add_edge("bash_executor", "planner")
    workflow.add_edge("consolidator", "planner")
    workflow.add_edge("search", "planner")
    workflow.add_edge("playwright", "planner")

    # Phase 1 & 2 executor nodes return to planner
    workflow.add_edge("browser_executor", "planner")
    workflow.add_edge("crawl_executor", "planner")
    workflow.add_edge("editor_executor", "planner")
    workflow.add_edge("planning_executor", "planner")
    workflow.add_edge("ask_human_executor", "planner")
    workflow.add_edge("document_executor", "planner")
    workflow.add_edge("data_analysis_executor", "planner")

    if enable_deep_research:
        workflow.add_edge("deep_research", "planner")

    logger.info("Agent graph created successfully")

    return workflow


def compile_graph(
    enable_checkpointing: bool = True,
    enable_hitl: bool = True,
    enable_prompt_enhancer: bool = True,
):
    """
    Create and compile the agent graph with optional checkpointing.

    Args:
        enable_checkpointing: If True, enable state persistence.
        enable_hitl: If True, enable HITL breakpoints.
        enable_prompt_enhancer: If True, enable prompt enhancement.

    Returns:
        Compiled graph ready for execution.
    """
    workflow = create_agent_graph(
        enable_hitl=enable_hitl,
        enable_prompt_enhancer=enable_prompt_enhancer,
    )

    if enable_checkpointing:
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


def compile_graph_simple():
    """
    Create and compile a simple graph without HITL or prompt enhancer.

    For backwards compatibility and testing.

    Returns:
        Compiled graph ready for execution.
    """
    return compile_graph(
        enable_checkpointing=False,
        enable_hitl=False,
        enable_prompt_enhancer=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick test - just compile the graph
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing Graph Compilation ===\n")

    # Test full graph
    try:
        graph = compile_graph(enable_checkpointing=True, enable_hitl=True)
        print("✓ Full graph compiled successfully!")
        print(
            f"  Nodes: {list(graph.nodes.keys()) if hasattr(graph, 'nodes') else 'N/A'}"
        )
    except Exception as e:
        print(f"✗ Full graph compilation failed: {e}")

    # Test simple graph
    try:
        graph = compile_graph_simple()
        print("✓ Simple graph compiled successfully!")
        print(
            f"  Nodes: {list(graph.nodes.keys()) if hasattr(graph, 'nodes') else 'N/A'}"
        )
    except Exception as e:
        print(f"✗ Simple graph compilation failed: {e}")
