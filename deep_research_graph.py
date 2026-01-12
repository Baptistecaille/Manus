"""
Deep Research Graph - LangGraph sub-graph for comprehensive research.

Implements a cyclic workflow:
research_planner → search_summarize → reflection → [loop or report_writer]
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from deep_research_state import DeepResearchStateDict, create_deep_research_state
from deep_research_config import DeepResearchConfig, DEFAULT_CONFIG
from nodes.research_planner import research_planner_node
from nodes.search_summarize import search_and_summarize_node
from nodes.reflection import reflection_node
from nodes.report_writer import report_writer_node

logger = logging.getLogger(__name__)


def should_continue_research(
    state: DeepResearchStateDict,
) -> Literal["plan_next_cycle", "write_report"]:
    """
    Router function to decide next step after reflection.

    Args:
        state: Current deep research state.

    Returns:
        "plan_next_cycle" to continue research, "write_report" to finish.
    """
    current_depth = state.get("research_depth", 0)
    max_depth = state.get("max_research_depth", 3)
    should_continue = state.get("should_continue", False)
    knowledge_gaps = state.get("knowledge_gaps", [])

    logger.debug(
        f"Router decision: depth={current_depth}/{max_depth}, "
        f"continue={should_continue}, gaps={len(knowledge_gaps)}"
    )

    # Stop conditions
    if current_depth >= max_depth:
        logger.info(f"Max depth {max_depth} reached, proceeding to report")
        return "write_report"

    if not should_continue:
        logger.info("Reflection recommends stopping, proceeding to report")
        return "write_report"

    if not knowledge_gaps:
        logger.info("No knowledge gaps identified, proceeding to report")
        return "write_report"

    # Continue researching
    logger.info(f"Continuing research: {len(knowledge_gaps)} gaps to fill")
    return "plan_next_cycle"


def create_deep_research_graph(
    config: DeepResearchConfig = DEFAULT_CONFIG,
) -> StateGraph:
    """
    Create the deep research sub-graph.

    Args:
        config: Research configuration.

    Returns:
        Compiled LangGraph StateGraph.
    """
    logger.info("Creating deep research graph...")

    workflow = StateGraph(DeepResearchStateDict)

    # Add nodes with config binding
    workflow.add_node("research_planner", lambda s: research_planner_node(s, config))
    workflow.add_node(
        "search_summarize", lambda s: search_and_summarize_node(s, config)
    )
    workflow.add_node("reflection", lambda s: reflection_node(s, config))
    workflow.add_node("report_writer", lambda s: report_writer_node(s, config))

    # Set entry point
    workflow.set_entry_point("research_planner")

    # Linear edges
    workflow.add_edge("research_planner", "search_summarize")
    workflow.add_edge("search_summarize", "reflection")

    # Conditional edge after reflection
    workflow.add_conditional_edges(
        "reflection",
        should_continue_research,
        {
            "plan_next_cycle": "research_planner",
            "write_report": "report_writer",
        },
    )

    # Report writer ends the sub-graph
    workflow.add_edge("report_writer", END)

    logger.info("Deep research graph created successfully")

    return workflow


def compile_deep_research_graph(config: DeepResearchConfig = DEFAULT_CONFIG):
    """
    Create and compile the deep research graph.

    Args:
        config: Research configuration.

    Returns:
        Compiled graph ready for execution.
    """
    workflow = create_deep_research_graph(config)
    return workflow.compile()


def run_deep_research(
    topic: str, config: DeepResearchConfig = DEFAULT_CONFIG, verbose: bool = True
) -> DeepResearchStateDict:
    """
    Run a complete deep research session.

    Args:
        topic: Research topic.
        config: Research configuration.
        verbose: Print progress updates.

    Returns:
        Final state with report.
    """
    if verbose:
        print(f"\n🔬 Starting Deep Research: {topic}")
        print(f"   Max cycles: {config.max_research_depth}")
        print(f"   Queries per cycle: {config.initial_queries_count}")
        print()

    # Create initial state
    state = create_deep_research_state(topic, config.max_research_depth)

    # Compile and run graph
    graph = compile_deep_research_graph(config)

    final_state = state
    step = 0

    for event in graph.stream(state):
        step += 1
        if isinstance(event, dict):
            for node_name, updates in event.items():
                if isinstance(updates, dict):
                    final_state.update(updates)

                if verbose:
                    depth = final_state.get("research_depth", 0)
                    print(f"  [{step}] {node_name} (cycle {depth})")

    if verbose and final_state.get("final_report"):
        report = final_state["final_report"]
        word_count = len(report.split())
        sources = len(final_state.get("findings", []))

        print(f"\n✅ Research Complete!")
        print(f"   Word count: {word_count}")
        print(f"   Sources: {sources}")
        print(f"   Cycles: {final_state.get('research_depth', 0)}")

    return final_state


if __name__ == "__main__":
    # Test the graph
    logging.basicConfig(level=logging.INFO)

    # Test graph creation
    try:
        graph = compile_deep_research_graph()
        print("✓ Deep research graph compiled successfully!")
    except Exception as e:
        print(f"✗ Graph compilation failed: {e}")
