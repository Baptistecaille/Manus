"""
Tests for Deep Research functionality.

Tests the deep research state, config, graph routing, and node parsing.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_research_state import (
    DeepResearchStateDict,
    create_deep_research_state,
    estimate_research_tokens,
)
from deep_research_config import DeepResearchConfig, DEFAULT_CONFIG
from deep_research_graph import should_continue_research


class TestDeepResearchState:
    """Tests for deep research state management."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_deep_research_state("Test topic")

        assert state["research_topic"] == "Test topic"
        assert state["research_depth"] == 0
        assert state["max_research_depth"] == 3
        assert state["research_queries"] == []
        assert state["findings"] == []
        assert state["should_continue"] is True

    def test_create_state_with_custom_depth(self):
        """Test state creation with custom max depth."""
        state = create_deep_research_state("Another topic", max_depth=5)

        assert state["max_research_depth"] == 5

    def test_estimate_tokens(self):
        """Test token estimation for research state."""
        state = create_deep_research_state("Short topic")
        state["findings"] = [
            {
                "content": "A" * 400,
                "source_title": "Test",
                "source_url": "http://test.com",
                "source_index": 1,
            }
        ]

        tokens = estimate_research_tokens(state)
        assert tokens > 0
        assert tokens > 100  # Should account for findings


class TestDeepResearchConfig:
    """Tests for deep research configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DEFAULT_CONFIG

        assert config.search_provider == "duckduckgo"
        assert config.max_research_depth == 3
        assert config.initial_queries_count == 4
        assert config.max_search_results == 10

    def test_from_env(self):
        """Test config creation from environment."""
        config = DeepResearchConfig.from_env()

        # Should use defaults when env vars not set
        assert config.max_research_depth >= 1
        assert config.max_search_results >= 1

    def test_model_defaults_to_none(self):
        """Test that model settings default to None (use global config)."""
        config = DeepResearchConfig()

        assert config.query_generator_model is None
        assert config.summarization_model is None
        assert config.reflection_model is None
        assert config.report_writer_model is None


class TestContinueResearchDecision:
    """Tests for the should_continue_research router function."""

    def test_stops_at_max_depth(self):
        """Test that research stops when max depth is reached."""
        state = create_deep_research_state("Topic", max_depth=3)
        state["research_depth"] = 3
        state["should_continue"] = True
        state["knowledge_gaps"] = ["gap1", "gap2"]

        result = should_continue_research(state)
        assert result == "write_report"

    def test_continues_with_gaps(self):
        """Test that research continues when gaps exist."""
        state = create_deep_research_state("Topic", max_depth=3)
        state["research_depth"] = 1
        state["should_continue"] = True
        state["knowledge_gaps"] = ["gap1", "gap2"]

        result = should_continue_research(state)
        assert result == "plan_next_cycle"

    def test_stops_without_gaps(self):
        """Test that research stops when no gaps remain."""
        state = create_deep_research_state("Topic", max_depth=3)
        state["research_depth"] = 1
        state["should_continue"] = True
        state["knowledge_gaps"] = []

        result = should_continue_research(state)
        assert result == "write_report"

    def test_stops_when_reflection_says_no(self):
        """Test that research stops when reflection recommends it."""
        state = create_deep_research_state("Topic", max_depth=3)
        state["research_depth"] = 1
        state["should_continue"] = False
        state["knowledge_gaps"] = ["gap1"]

        result = should_continue_research(state)
        assert result == "write_report"

    def test_continues_at_zero_depth(self):
        """Test first cycle continues when should_continue is True."""
        state = create_deep_research_state("Topic", max_depth=3)
        state["research_depth"] = 0
        state["should_continue"] = True
        state["knowledge_gaps"] = ["initial gap"]

        result = should_continue_research(state)
        assert result == "plan_next_cycle"


class TestGraphCompilation:
    """Tests for deep research graph compilation."""

    def test_graph_compiles(self):
        """Test that deep research graph compiles successfully."""
        from deep_research_graph import compile_deep_research_graph

        graph = compile_deep_research_graph()
        assert graph is not None

    def test_graph_with_custom_config(self):
        """Test graph compilation with custom config."""
        from deep_research_graph import compile_deep_research_graph

        config = DeepResearchConfig(max_research_depth=5, max_search_results=5)
        graph = compile_deep_research_graph(config)
        assert graph is not None


class TestRouterIntegration:
    """Tests for router deep_research action."""

    def test_router_handles_deep_research(self):
        """Test that main router routes to deep_research."""
        from router import router
        from agent_state import create_initial_state

        state = create_initial_state("Test")
        state["current_action"] = "deep_research"

        result = router(state)
        assert result == "deep_research"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
