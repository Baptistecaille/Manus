"""
Report Writer Node - Generates Markdown report for Deep Research.

Compiles all research findings into a comprehensive, well-cited
Markdown report and saves it to the Seedbox workspace.
"""

import logging
import re
from datetime import datetime
from typing import List

from deep_research_state import DeepResearchStateDict
from deep_research_config import DeepResearchConfig, DEFAULT_CONFIG
from llm_factory import create_llm
from seedbox_executor import SeedboxExecutor

logger = logging.getLogger(__name__)


# Report generation prompt
REPORT_PROMPT = """You are an expert research report writer. Create a comprehensive, professional report.

RESEARCH TOPIC: {topic}

RESEARCH CONDUCTED: {num_cycles} research cycles, {num_sources} sources analyzed

COMPILED RESEARCH FINDINGS:
{findings_text}

SOURCES LIST:
{sources_list}

Write a comprehensive research report in Markdown format following these requirements:

1. LENGTH: 2000-4000 words
2. TONE: Professional and accessible, suitable for both experts and informed readers
3. CITATIONS: Use inline citations [1], [2] for EVERY factual claim from sources
4. STRUCTURE: Use clear Markdown headers (##, ###)
5. ANALYSIS: Don't just summarize - synthesize, analyze, and provide insights
6. HONESTY: Note any contradictions, limitations, or areas needing more research

REPORT STRUCTURE (follow this exactly):
# [Compelling title based on the research topic]

## Executive Summary
[200-300 word overview of key findings and conclusions]

## 1. Introduction
[Context, why this topic matters, scope of the research]

## 2. Methodology
[Brief description of research approach and sources consulted]

## 3. Key Findings
### 3.1 [First major theme]
[Analysis with citations]

### 3.2 [Second major theme]
[Analysis with citations]

### 3.3 [Additional themes as needed]
[Analysis with citations]

## 4. Analysis & Insights
[Cross-cutting observations, patterns, implications]

## 5. Limitations & Gaps
[What couldn't be fully addressed, areas for future research]

## 6. Conclusions
[Key takeaways and recommendations]

---
## References
[Will be automatically appended - do not include]

Begin the report now. Remember to cite sources using [1], [2] notation."""


def _build_findings_text(findings: List[dict]) -> str:
    """Build formatted findings for prompt."""
    if not findings:
        return "No findings available."

    parts = []
    for f in findings:
        idx = f.get("source_index", "?")
        content = f.get("content", "")[:600]
        parts.append(f"[Source {idx}]: {content}")

    return "\n\n".join(parts)


def _build_sources_list(findings: List[dict]) -> str:
    """Build numbered sources list."""
    seen_urls = set()
    sources = []

    for f in findings:
        url = f.get("source_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            idx = f.get("source_index", len(sources) + 1)
            title = f.get("source_title", "Unknown source")
            sources.append(f"[{idx}] {title} - {url}")

    return "\n".join(sources) if sources else "No sources available."


def _sanitize_filename(text: str) -> str:
    """Create safe filename from text."""
    # Remove special characters, keep alphanumeric and spaces
    clean = re.sub(r"[^\w\s-]", "", text.lower())
    # Replace spaces with underscores
    clean = re.sub(r"\s+", "_", clean)
    # Limit length
    return clean[:50]


def report_writer_node(
    state: DeepResearchStateDict, config: DeepResearchConfig = DEFAULT_CONFIG
) -> dict:
    """
    LangGraph node that generates the final research report.

    Args:
        state: Current deep research state with all findings.
        config: Research configuration.

    Returns:
        State updates with final_report and save confirmation.
    """
    topic = state.get("research_topic", "Unknown Topic")
    findings = state.get("findings", [])
    research_depth = state.get("research_depth", 1)

    logger.info(f"Report Writer - Topic: {topic[:50]}, Findings: {len(findings)}")

    if not findings:
        logger.warning("No findings for report")
        fallback_report = f"""# Research Report: {topic}

## Executive Summary
This research was unable to gather sufficient information on the topic.

## Methodology
Attempted automated web search and analysis.

## Findings
No significant findings were collected.

## Conclusions
Please try the research again or refine the topic.
"""
        return {
            "final_report": fallback_report,
            "messages": [
                {
                    "role": "system",
                    "content": "[REPORT] Generated fallback report (no findings)",
                }
            ],
        }

    # Build prompt components
    findings_text = _build_findings_text(findings)
    sources_list = _build_sources_list(findings)

    prompt = REPORT_PROMPT.format(
        topic=topic,
        num_cycles=research_depth,
        num_sources=len(findings),
        findings_text=findings_text[:12000],  # Limit for context
        sources_list=sources_list,
    )

    try:
        llm = create_llm(
            provider=config.report_writer_provider,
            model=config.report_writer_model,
            temperature=config.report_temperature,
        )

        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are an expert research report writer. Write comprehensive, well-cited reports.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        report_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Append references section if not already present
        if "## References" not in report_content:
            report_content += "\n\n---\n\n## References\n\n" + sources_list

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.report_filename_prefix}_{_sanitize_filename(topic)}_{timestamp}.md"
        full_path = f"{config.report_save_path}/{filename}"

        # Save to Seedbox
        save_success = False
        save_message = ""

        try:
            executor = SeedboxExecutor()
            save_result = executor.write_file(full_path, report_content)
            save_success = save_result if isinstance(save_result, bool) else True
            save_message = f"Report saved to {full_path}"
            logger.info(save_message)
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            save_message = f"Report generated but save failed: {e}"

        # Build completion message
        word_count = len(report_content.split())
        message_content = (
            f"[REPORT COMPLETE]\n"
            f"  Topic: {topic[:60]}...\n"
            f"  Sources cited: {len(findings)}\n"
            f"  Word count: ~{word_count}\n"
            f"  Research cycles: {research_depth}\n"
            f"  {save_message}"
        )

        return {
            "final_report": report_content,
            "last_tool_output": save_message,
            "messages": [{"role": "assistant", "content": message_content}],
            "internal_monologue": f"Report generated: {word_count} words, {len(findings)} sources",
        }

    except Exception as e:
        logger.error(f"Report generation error: {e}")

        error_report = f"""# Research Report: {topic}

## Error
An error occurred during report generation: {str(e)}

## Collected Sources
{sources_list}

## Raw Findings
""" + "\n\n".join(
            f"- {f.get('content', '')[:200]}..." for f in findings[:10]
        )

        return {
            "final_report": error_report,
            "messages": [{"role": "system", "content": f"[REPORT ERROR] {str(e)}"}],
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    from deep_research_state import create_deep_research_state

    state = create_deep_research_state("Test Topic")
    state["findings"] = [
        {
            "source_index": 1,
            "source_title": "Example Source",
            "source_url": "https://example.com",
            "content": "This is sample content for testing.",
        },
    ]
    state["research_depth"] = 1

    # Note: This will try to save to Seedbox, which may fail without Docker
    result = report_writer_node(state)

    print(f"\nReport preview (first 500 chars):")
    print(result.get("final_report", "")[:500])
