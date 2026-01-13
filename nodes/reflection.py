"""
Reflection Node - Knowledge gap analysis for Deep Research.

Analyzes current research findings to identify gaps, contradictions,
and determine whether to continue researching or write the report.
"""

import logging
import re
from typing import List, Tuple

from deep_research_state import DeepResearchStateDict
from deep_research_config import DeepResearchConfig, DEFAULT_CONFIG
from llm_factory import create_llm
from nodes.schema import ReflectionOutput

logger = logging.getLogger(__name__)


# Reflection prompt
REFLECTION_PROMPT = """You are a research quality analyst. Critically evaluate the research findings.

ORIGINAL RESEARCH TOPIC: {topic}

CURRENT RESEARCH CYCLE: {current_depth} of {max_depth}

FINDINGS FROM ALL RESEARCH CYCLES:
{findings_text}

{previous_gaps_section}

Analyze the findings critically and provide your assessment:

1. WELL-COVERED ASPECTS: What parts of the topic are thoroughly researched?
2. KNOWLEDGE GAPS: What critical information is still missing?
   - Be specific (e.g., "Missing quantitative data on market size in 2024")
   - Prioritize gaps by importance to the topic
3. CONTRADICTIONS: Any conflicting information that needs clarification?
4. SOURCE QUALITY: Are sources authoritative and recent enough?
5. RECOMMENDATION: Should we continue researching?
   - YES if: Significant gaps remain AND we haven't reached max depth ({max_depth})
   - NO if: Coverage is comprehensive OR we've reached max depth

RESPOND IN THIS EXACT FORMAT:
WELL_COVERED:
- [Aspect 1]
- [Aspect 2]

KNOWLEDGE_GAPS:
- [Specific gap 1]
- [Specific gap 2]

CONTRADICTIONS:
- [Contradiction if any, or "None identified"]

SOURCE_QUALITY: [Brief assessment of source reliability and recency]

RECOMMENDATION: [YES or NO]
REASONING: [Why continue or stop researching]"""


def _build_findings_text(findings: List[dict], max_chars: int = 15000) -> str:
    """Build formatted findings text for prompt."""
    if not findings:
        return "No findings yet."

    parts = []
    total_chars = 0

    for f in findings:
        entry = (
            f"[{f.get('source_index', '?')}] {f.get('source_title', 'Unknown source')}\n"
            f"    URL: {f.get('source_url', 'N/A')}\n"
            f"    Summary: {f.get('content', '')[:500]}"
        )

        if total_chars + len(entry) > max_chars:
            parts.append("... [Additional findings truncated for context]")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n\n".join(parts)


# regex parsing function `_parse_reflection` is removed as we use structured output
# def _parse_reflection(response_text: str) -> Tuple[List[str], List[str], bool, str]: ...


def reflection_node(
    state: DeepResearchStateDict, config: DeepResearchConfig = DEFAULT_CONFIG
) -> dict:
    """
    LangGraph node that reflects on research progress.

    Args:
        state: Current deep research state with findings.
        config: Research configuration.

    Returns:
        State updates with knowledge_gaps and should_continue decision.
    """
    topic = state.get("research_topic", "")
    findings = state.get("findings", [])
    current_depth = state.get("research_depth", 0)
    max_depth = state.get("max_research_depth", config.max_research_depth)
    previous_gaps = state.get("knowledge_gaps", [])

    logger.info(f"Reflection - Cycle {current_depth}, Findings: {len(findings)}")

    if not findings:
        logger.warning("No findings to reflect on")
        return {
            "knowledge_gaps": ["No research data available - need to search first"],
            "should_continue": current_depth < max_depth,
            "research_depth": current_depth + 1,
            "messages": [
                {"role": "system", "content": "[REFLECTION] No findings available"}
            ],
        }

    # Build prompt components
    findings_text = _build_findings_text(findings)

    previous_gaps_section = ""
    if previous_gaps:
        gaps_list = "\n".join(f"- {g}" for g in previous_gaps)
        previous_gaps_section = (
            f"PREVIOUS KNOWLEDGE GAPS (from earlier cycles):\n{gaps_list}"
        )

    prompt = REFLECTION_PROMPT.format(
        topic=topic,
        current_depth=current_depth + 1,
        max_depth=max_depth,
        findings_text=findings_text,
        previous_gaps_section=previous_gaps_section,
    )

    try:
        llm = create_llm(
            provider=config.reflection_provider,
            model=config.reflection_model,
            temperature=config.reflection_temperature,
        )
        structured_llm = llm.with_structured_output(ReflectionOutput)

        parsed: ReflectionOutput = structured_llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a critical research analyst. Be thorough and specific.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        if parsed is None:
            raise ValueError("LLM returned None for structured output")

        well_covered = parsed.well_covered_aspects
        knowledge_gaps = parsed.knowledge_gaps
        llm_continue = parsed.should_continue
        reasoning = parsed.reasoning

        # Apply depth limit
        new_depth = current_depth + 1
        at_max_depth = new_depth >= max_depth

        # Final decision: continue only if LLM says yes AND not at max depth
        should_continue = llm_continue and not at_max_depth

        logger.info(
            f"Reflection complete: gaps={len(knowledge_gaps)}, "
            f"continue={should_continue}, depth={new_depth}/{max_depth}"
        )

        # Build message
        decision = "CONTINUE RESEARCH" if should_continue else "WRITE REPORT"
        message_content = (
            f"[REFLECTION] Cycle {new_depth}/{max_depth}\n"
            f"  Well covered: {len(well_covered)} aspects\n"
            f"  Knowledge gaps: {len(knowledge_gaps)}\n"
            f"  Decision: {decision}\n"
            f"  Reasoning: {reasoning[:200]}..."
        )

        return {
            "knowledge_gaps": knowledge_gaps,
            "should_continue": should_continue,
            "research_depth": new_depth,
            "messages": [{"role": "assistant", "content": message_content}],
            "internal_monologue": f"Reflection: {reasoning[:300]}",
        }

    except Exception as e:
        logger.error(f"Reflection error: {e}")

        # On error, proceed to report if we have findings
        new_depth = current_depth + 1
        return {
            "knowledge_gaps": [],
            "should_continue": False,  # Error recovery: proceed to report
            "research_depth": new_depth,
            "messages": [
                {
                    "role": "system",
                    "content": f"[REFLECTION ERROR] {str(e)}. Proceeding to report.",
                }
            ],
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    from deep_research_state import create_deep_research_state

    state = create_deep_research_state("AI Agents")
    state["findings"] = [
        {
            "source_index": 1,
            "source_title": "LangGraph Guide",
            "source_url": "https://example.com/1",
            "content": "LangGraph is a framework for building agents...",
        },
        {
            "source_index": 2,
            "source_title": "Agent Patterns",
            "source_url": "https://example.com/2",
            "content": "Common patterns include ReAct, reflection...",
        },
    ]

    result = reflection_node(state)

    print(f"\nKnowledge gaps: {result.get('knowledge_gaps', [])}")
    print(f"Should continue: {result.get('should_continue')}")
    print(f"New depth: {result.get('research_depth')}")
