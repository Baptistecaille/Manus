"""
DeepResearchConfig - Configuration for Deep Research mode.

All model selections use the default from llm_factory (configured via .env),
ensuring consistent model usage across the application.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeepResearchConfig:
    """
    Configuration settings for Deep Research mode.

    All model settings default to None, which means the global
    LLM_PROVIDER and LLM_MODEL from .env will be used.
    """

    # === Search Settings ===
    search_provider: str = "duckduckgo"
    max_search_results: int = 10  # Results per query
    max_research_depth: int = 3  # Maximum research cycles
    initial_queries_count: int = 4  # Queries in first cycle
    followup_queries_count: int = 3  # Queries in subsequent cycles

    # === Model Settings (None = use default from .env) ===
    # These can be overridden if specific models are needed per task
    query_generator_provider: Optional[str] = None
    query_generator_model: Optional[str] = None
    summarization_provider: Optional[str] = None
    summarization_model: Optional[str] = None
    reflection_provider: Optional[str] = None
    reflection_model: Optional[str] = None
    report_writer_provider: Optional[str] = None
    report_writer_model: Optional[str] = None

    # === Temperature Settings ===
    query_temperature: float = 0.7  # Higher for query diversity
    summarization_temperature: float = 0.3
    reflection_temperature: float = 0.2
    report_temperature: float = 0.4

    # === Token Limits ===
    max_summary_tokens: int = 800  # Per search result summary
    max_reflection_tokens: int = 1500  # Reflection analysis
    max_report_tokens: int = 10000  # Final report

    # === Output Settings ===
    report_save_path: str = "/workspace"  # Seedbox save location
    report_filename_prefix: str = "deep_research"

    # === Retry Settings ===
    search_max_retries: int = 3
    search_retry_delay: float = 2.0  # Seconds between retries

    @classmethod
    def from_env(cls) -> "DeepResearchConfig":
        """
        Create config from environment variables.

        Environment variables (all optional):
            DEEP_RESEARCH_MAX_DEPTH: Maximum research cycles
            DEEP_RESEARCH_MAX_RESULTS: Results per search query
            DEEP_RESEARCH_INITIAL_QUERIES: Number of initial queries

        Returns:
            DeepResearchConfig with env overrides applied.
        """
        return cls(
            max_research_depth=int(os.getenv("DEEP_RESEARCH_MAX_DEPTH", "3")),
            max_search_results=int(os.getenv("DEEP_RESEARCH_MAX_RESULTS", "10")),
            initial_queries_count=int(os.getenv("DEEP_RESEARCH_INITIAL_QUERIES", "4")),
        )


# Default config instance
DEFAULT_CONFIG = DeepResearchConfig()


if __name__ == "__main__":
    # Quick test
    config = DeepResearchConfig.from_env()
    print("Deep Research Configuration:")
    print(f"  Search provider: {config.search_provider}")
    print(f"  Max results: {config.max_search_results}")
    print(f"  Max depth: {config.max_research_depth}")
    print(f"  Initial queries: {config.initial_queries_count}")
    print(f"  Query temperature: {config.query_temperature}")
