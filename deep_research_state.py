"""
DeepResearchState - Extended state for Deep Research mode.

Extends the base AgentState with fields for tracking research progress,
queries, findings, and report generation.
"""

import operator
from typing import Annotated, TypedDict, List, Optional


class SearchResult(TypedDict):
    """Single search result with summary."""

    query: str
    title: str
    url: str
    snippet: str
    summary: str
    source_quality: str  # academic, news, blog, marketing, unknown


class Finding(TypedDict):
    """Consolidated research finding with source."""

    content: str
    source_title: str
    source_url: str
    source_index: int  # For citation [1], [2], etc.


class DeepResearchStateDict(TypedDict):
    """
    Extended LangGraph state for Deep Research mode.

    Inherits concepts from AgentStateDict and adds research-specific fields.
    """

    # === Inherited from AgentState ===
    messages: Annotated[List[dict], operator.add]
    todo_list: str
    internal_monologue: str
    seedbox_manifest: List[str]
    last_tool_output: str
    iteration_count: int
    context_size: int
    consolidated_history: str
    current_action: str
    action_details: str

    # === Deep Research Specific ===
    research_topic: str  # Original topic extracted from query
    research_queries: List[str]  # Generated search queries
    search_results: List[dict]  # SearchResult dicts
    knowledge_gaps: List[str]  # Identified gaps for next cycle
    findings: List[dict]  # Finding dicts with sources
    research_depth: int  # Current cycle (0, 1, 2, 3)
    max_research_depth: int  # Limit (default: 3)
    final_report: str  # Generated Markdown report
    should_continue: bool  # Decision from reflection node


def create_deep_research_state(
    research_topic: str, max_depth: int = 3
) -> DeepResearchStateDict:
    """
    Create initial state for a deep research run.

    Args:
        research_topic: The topic to research.
        max_depth: Maximum number of research cycles.

    Returns:
        Initialized DeepResearchStateDict.
    """
    return DeepResearchStateDict(
        # Base state fields
        messages=[{"role": "user", "content": f"Research topic: {research_topic}"}],
        todo_list=f"🔬 Research: {research_topic[:80]}{'...' if len(research_topic) > 80 else ''}",
        internal_monologue="",
        seedbox_manifest=[],
        last_tool_output="",
        iteration_count=0,
        context_size=0,
        consolidated_history="",
        current_action="",
        action_details="",
        # Deep Research fields
        research_topic=research_topic,
        research_queries=[],
        search_results=[],
        knowledge_gaps=[],
        findings=[],
        research_depth=0,
        max_research_depth=max_depth,
        final_report="",
        should_continue=True,
    )


def estimate_research_tokens(state: DeepResearchStateDict) -> int:
    """
    Estimate total tokens used in research state.

    Args:
        state: Current research state.

    Returns:
        Estimated token count.
    """
    total = 0

    # Messages
    for msg in state.get("messages", []):
        total += len(str(msg.get("content", ""))) // 4

    # Research content
    total += len(state.get("research_topic", "")) // 4
    total += sum(len(q) // 4 for q in state.get("research_queries", []))
    total += sum(len(str(r)) // 4 for r in state.get("search_results", []))
    total += sum(len(g) // 4 for g in state.get("knowledge_gaps", []))
    total += sum(len(str(f)) // 4 for f in state.get("findings", []))
    total += len(state.get("final_report", "")) // 4

    return total


if __name__ == "__main__":
    # Quick test
    state = create_deep_research_state("Advances in LangGraph agent frameworks")
    print(f"Initial deep research state:")
    print(f"  Topic: {state['research_topic']}")
    print(f"  Max depth: {state['max_research_depth']}")
    print(f"  Current depth: {state['research_depth']}")
