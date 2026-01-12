"""
Research Planner Node - Generates search queries for Deep Research.

This node analyzes the research topic and any knowledge gaps to generate
diverse, targeted search queries for the next research cycle.
"""

import logging
import re
from typing import List

from deep_research_state import DeepResearchStateDict
from deep_research_config import DeepResearchConfig, DEFAULT_CONFIG
from llm_factory import create_llm

logger = logging.getLogger(__name__)


# Query generation prompt
QUERY_GENERATION_PROMPT = """You are a research planning expert. Your task is to generate diverse, effective search queries.

RESEARCH TOPIC: {topic}

{context_section}

Generate {query_count} diverse search queries that will:
1. Cover different facets and angles of the topic
2. Target authoritative sources (academic papers, official docs, expert analysis)
3. {gap_instruction}
4. Avoid redundancy with any previous queries shown above

RESPOND IN THIS EXACT FORMAT:
QUERIES:
1. [First query - specific and targeted]
2. [Second query - different angle or aspect]
3. [Third query - complementary perspective]
{additional_slots}

REASONING: [Brief explanation of why these queries provide comprehensive coverage]"""


def _build_context_section(state: DeepResearchStateDict) -> str:
    """Build context section showing previous findings and gaps."""
    sections = []

    # Previous findings
    if state.get("findings"):
        findings_preview = []
        for f in state["findings"][:5]:  # Limit to 5 for context
            findings_preview.append(f"- {f.get('content', '')[:200]}...")
        sections.append(f"PREVIOUS FINDINGS:\n" + "\n".join(findings_preview))

    # Knowledge gaps
    if state.get("knowledge_gaps"):
        gaps = "\n".join(f"- {g}" for g in state["knowledge_gaps"])
        sections.append(f"IDENTIFIED KNOWLEDGE GAPS:\n{gaps}")

    # Previous queries (to avoid repetition)
    if state.get("research_queries"):
        prev_queries = "\n".join(f"- {q}" for q in state["research_queries"])
        sections.append(f"PREVIOUS QUERIES (avoid similar):\n{prev_queries}")

    return "\n\n".join(sections) if sections else "This is the first research cycle."


def _parse_queries(response_text: str, expected_count: int) -> List[str]:
    """Parse query list from LLM response."""
    queries = []

    # Look for QUERIES: section
    queries_match = re.search(
        r"QUERIES:\s*\n(.*?)(?=REASONING:|$)", response_text, re.DOTALL | re.IGNORECASE
    )

    if queries_match:
        queries_section = queries_match.group(1)
        # Extract numbered items
        for line in queries_section.split("\n"):
            line = line.strip()
            # Match "1. query" or "- query" patterns
            match = re.match(r"^[\d]+[\.\)]\s*(.+)$", line)
            if match:
                query = match.group(1).strip()
                # Remove surrounding quotes if present
                if (query.startswith('"') and query.endswith('"')) or (
                    query.startswith("'") and query.endswith("'")
                ):
                    query = query[1:-1]
                if query and len(query) > 5:  # Sanity check
                    queries.append(query)

    # Fallback: if no queries found, try line-by-line
    if not queries:
        lines = response_text.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith(("QUERIES", "REASONING", "#")):
                if len(line) > 10 and len(line) < 200:
                    queries.append(line)

    return queries[:expected_count]


def research_planner_node(
    state: DeepResearchStateDict, config: DeepResearchConfig = DEFAULT_CONFIG
) -> dict:
    """
    LangGraph node that generates research queries.

    Args:
        state: Current deep research state.
        config: Research configuration.

    Returns:
        State updates with research_queries.
    """
    topic = state.get("research_topic", "")
    current_depth = state.get("research_depth", 0)

    logger.info(f"Research Planner - Cycle {current_depth}, Topic: {topic[:50]}...")

    # Determine query count based on cycle
    if current_depth == 0:
        query_count = config.initial_queries_count
        gap_instruction = (
            "Explore the topic broadly to establish foundational understanding"
        )
    else:
        query_count = config.followup_queries_count
        gap_instruction = "Focus specifically on filling the identified knowledge gaps"

    # Build context
    context_section = _build_context_section(state)

    # Build additional slots if more than 3 queries
    additional_slots = ""
    if query_count > 3:
        for i in range(4, query_count + 1):
            additional_slots += f"{i}. [Query {i} - unique angle]\n"

    prompt = QUERY_GENERATION_PROMPT.format(
        topic=topic,
        context_section=context_section,
        query_count=query_count,
        gap_instruction=gap_instruction,
        additional_slots=additional_slots,
    )

    try:
        # Create LLM (uses default from .env)
        llm = create_llm(
            provider=config.query_generator_provider,
            model=config.query_generator_model,
            temperature=config.query_temperature,
        )

        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are an expert research query generator. Be specific and diverse.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        queries = _parse_queries(response_text, query_count)

        if not queries:
            logger.warning("No queries parsed, using fallback")
            queries = [
                f"{topic} overview",
                f"{topic} latest research",
                f"{topic} best practices",
            ]

        logger.info(f"Generated {len(queries)} queries: {queries}")

        # Build message for history
        message = {
            "role": "assistant",
            "content": f"[RESEARCH PLANNER] Cycle {current_depth + 1}: Generated {len(queries)} queries\n"
            + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(queries)),
        }

        return {
            "research_queries": queries,
            "messages": [message],
            "internal_monologue": f"Research cycle {current_depth + 1}: Planning queries to {'explore topic' if current_depth == 0 else 'fill knowledge gaps'}",
        }

    except Exception as e:
        logger.error(f"Research planner error: {e}")

        # Fallback queries
        fallback_queries = [
            f"{topic}",
            f"{topic} comprehensive guide",
            f"{topic} recent developments",
        ]

        return {
            "research_queries": fallback_queries,
            "messages": [
                {
                    "role": "system",
                    "content": f"[RESEARCH PLANNER ERROR] {str(e)}. Using fallback queries.",
                }
            ],
            "internal_monologue": f"Error in query generation: {str(e)}",
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    from deep_research_state import create_deep_research_state

    state = create_deep_research_state("Latest advances in autonomous AI agents")
    result = research_planner_node(state)

    print(f"\nGenerated queries:")
    for i, q in enumerate(result.get("research_queries", []), 1):
        print(f"  {i}. {q}")
