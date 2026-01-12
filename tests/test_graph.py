"""
Tests for the LangGraph agent graph.

These tests verify graph compilation and basic execution flow.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_state import (
    AgentStateDict,
    create_initial_state,
    estimate_tokens,
    calculate_context_size,
)
from router import router
from agent_graph import create_agent_graph, compile_graph


class TestAgentState:
    """Tests for agent state management."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state("Test query")

        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "user"
        assert "Test query" in state["messages"][0]["content"]
        assert state["iteration_count"] == 0
        assert state["context_size"] > 0

    def test_estimate_tokens(self):
        """Test token estimation."""
        # Empty string
        assert estimate_tokens("") == 0

        # Simple text (4 chars per token approx)
        text = "Hello World!"  # 12 chars = ~3 tokens
        assert 2 <= estimate_tokens(text) <= 4

    def test_calculate_context_size(self):
        """Test full context size calculation."""
        state = create_initial_state("Short query")
        state["todo_list"] = "Some todo items here"
        state["internal_monologue"] = "Agent thinking..."

        size = calculate_context_size(state)
        assert size > 0
        assert size == state["context_size"] or size > state["context_size"]


class TestRouter:
    """Tests for the router logic."""

    def test_bash_action(self):
        """Test routing to bash executor."""
        state = create_initial_state("Test")
        state["current_action"] = "bash"

        assert router(state) == "bash_executor"

    def test_complete_action(self):
        """Test routing to end on complete."""
        state = create_initial_state("Test")
        state["current_action"] = "complete"

        assert router(state) == "end"

    def test_consolidation_action(self):
        """Test routing to consolidator."""
        state = create_initial_state("Test")
        state["current_action"] = "consolidate"

        assert router(state) == "consolidator"

    def test_context_size_trigger(self):
        """Test automatic consolidation trigger."""
        state = create_initial_state("Test")
        state["context_size"] = 100000  # Above threshold
        state["current_action"] = "bash"

        assert router(state) == "consolidator"

    def test_max_iterations(self):
        """Test max iterations safety check."""
        state = create_initial_state("Test")
        state["iteration_count"] = 31
        state["current_action"] = "bash"

        assert router(state) == "end"

    def test_unknown_action(self):
        """Test fallback to planner for unknown actions."""
        state = create_initial_state("Test")
        state["current_action"] = "unknown_action"

        assert router(state) == "planner"


class TestGraphCompilation:
    """Tests for graph compilation."""

    def test_graph_creates_successfully(self):
        """Test that graph can be created."""
        graph = create_agent_graph()
        assert graph is not None

    def test_graph_compiles_successfully(self):
        """Test that graph can be compiled."""
        compiled = compile_graph()
        assert compiled is not None

    def test_graph_has_expected_nodes(self):
        """Test that graph has all expected nodes."""
        graph = create_agent_graph()

        # Check nodes exist
        expected_nodes = [
            "planner",
            "bash_executor",
            "consolidator",
            "search",
            "playwright",
        ]
        for node in expected_nodes:
            assert node in graph.nodes, f"Missing node: {node}"


class TestIntegration:
    """Integration tests (require Docker running)."""

    @pytest.fixture
    def docker_running(self):
        """Check if Docker container is running."""
        import subprocess

        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", "name=manus-sandbox"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def test_graph_single_step(self, docker_running):
        """Test running the graph for one step."""
        if not docker_running:
            pytest.skip("Docker container not running")

        compiled = compile_graph()
        state = create_initial_state("List files in /workspace")

        # Run for just first event
        for event in compiled.stream(state):
            # Just verify we get an event
            assert isinstance(event, dict)
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
