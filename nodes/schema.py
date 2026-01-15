"""
Centralized Pydantic schemas for agent structured outputs.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# PROMPT ENHANCER SCHEMAS
# =============================================================================


class AnalysisResult(BaseModel):
    """Analysis of the original query."""

    original_query: str
    detected_intent: Literal[
        "code_generation",
        "web_research",
        "data_analysis",
        "file_manipulation",
        "mixed_workflow",
    ] = Field(description="Primary intent of the user request")
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    ambiguities_detected: List[str] = Field(default_factory=list)
    assumptions_made: List[str] = Field(default_factory=list)


class ContextEnrichment(BaseModel):
    """Context added to the query."""

    relevant_workspace_files: List[str] = Field(default_factory=list)
    suggested_tools: List[str] = Field(default_factory=list)
    technical_constraints_applied: List[str] = Field(default_factory=list)


class RiskFactor(BaseModel):
    """Individual risk factor detected."""

    type: str
    severity: int = Field(ge=1, le=10)
    description: str


class RiskAssessment(BaseModel):
    """Risk assessment result."""

    global_level: Literal["low", "medium", "high", "critical"]
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    recommended_hitl_level: Literal["strict", "moderate", "minimal"]


class ExecutionHints(BaseModel):
    """Hints for execution planning."""

    estimated_complexity: Literal["simple", "moderate", "complex"]
    suggested_timeout: int = 180
    requires_internet: bool = False
    requires_file_access: List[str] = Field(default_factory=list)


class EnhancerOutput(BaseModel):
    """Complete output from the Prompt Enhancer."""

    analysis: AnalysisResult
    enhanced_query: str
    context_enrichment: ContextEnrichment
    risk_assessment: RiskAssessment
    execution_hints: ExecutionHints


# =============================================================================
# PLANNER SCHEMAS
# =============================================================================


class PlannerOutput(BaseModel):
    """Structured output for the Planner node."""

    internal_monologue: str = Field(
        description="Chain-of-thought reasoning before deciding on the action. Reflect on previous tool outputs and current state."
    )
    todo_list: str = Field(
        description="Updated high-level task list. Use '✅ Done' and '🔲 Pending' markers."
    )
    next_action: Literal[
        "bash",
        "deep_research",
        "search",
        "playwright",
        "browser",
        "crawl",
        "edit",
        "plan",
        "ask",
        "consolidate",
        "complete",
        "document",
        "data_analysis",
    ] = Field(description="The next tool or action to execute.")
    action_details: str = Field(
        description="Specific arguments or command for the action. For 'bash', provide the command. For 'edit', provide the JSON/YAML edit spec."
    )
    reasoning: str = Field(
        description="Brief justification for why this specific action was chosen."
    )


# =============================================================================
# REFLECTION SCHEMAS
# =============================================================================


class ReflectionOutput(BaseModel):
    """Structured output for the Deep Research Reflection node."""

    well_covered_aspects: List[str] = Field(
        description="List of topic aspects that have been thoroughly researched."
    )
    knowledge_gaps: List[str] = Field(
        description="List of specific information that is missing or incomplete."
    )
    contradictions: List[str] = Field(
        description="Any conflicting information found between sources.",
        default_factory=list,
    )
    source_quality_assessment: str = Field(
        description="Brief assessment of the reliability and recency of sources."
    )
    should_continue: bool = Field(
        description="True if significant gaps remain and max depth is not reached. False otherwise."
    )
    reasoning: str = Field(
        description="Explanation for the decision to continue or stop."
    )


# =============================================================================
# SEARCH SUMMARY SCHEMAS
# =============================================================================


class SearchSummaryOutput(BaseModel):
    """Structured output for search result summarization."""

    summary: str = Field(
        description="Concise summary (max 300 words) relevant to the research topic."
    )
    source_type: Literal[
        "academic", "news", "blog", "marketing", "official", "unknown"
    ] = Field(description="Type/category of the source.")
    relevance: Literal["high", "medium", "low"] = Field(
        description="Relevance of this information to the research topic."
    )


# =============================================================================
# SWE PLANNER SCHEMAS
# =============================================================================


class SWEPlannerOutput(BaseModel):
    """Structured output for the SWE Planner node."""

    internal_monologue: str = Field(
        description="Technical reasoning about the current situation and what to do next."
    )
    todo_list: str = Field(
        description="Current state of technical tasks. Use '✅ Done', '🔲 Pending' markers."
    )
    next_action: Literal["bash", "edit", "complete"] = Field(
        description="The next action to execute."
    )
    action_details: str = Field(
        description="For 'bash': the exact shell command(s) to run. For 'edit': JSON with path/content. For 'complete': final summary."
    )
    reasoning: str = Field(description="Brief justification for this technical step.")
