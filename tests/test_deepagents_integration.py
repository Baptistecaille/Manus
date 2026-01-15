"""
Integration tests for Deep Agents middleware components.

Tests the integration of FilesystemMiddleware, TodoListMiddleware,
SubAgentMiddleware, and eviction handling into the Manus agent.
"""

import os
import pytest
from pathlib import Path
from typing import Dict, Any


# Test fixtures
@pytest.fixture
def test_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def sample_state(test_workspace) -> Dict[str, Any]:
    """Create a sample agent state for testing."""
    return {
        "messages": [{"role": "user", "content": "Test query"}],
        "original_query": "Test query",
        "enhanced_query": "",
        "detected_intent": "file_manipulation",
        "confidence_score": 0.9,
        "intent_reasoning": "",
        "workspace_context": {"current_directory": str(test_workspace)},
        "technical_constraints": {},
        "todo_list": "",
        "internal_monologue": "",
        "seedbox_manifest": [],
        "last_tool_output": "",
        "execution_plan": {},
        "plan_validation_status": "pending",
        "plan_modification_requests": [],
        "current_step_id": "",
        "executor_outputs": [],
        "current_action": "",
        "action_details": "",
        "pending_bash_commands": [],
        "bash_validation_status": "pending",
        "hitl_mode": "minimal",
        "current_breakpoint": "",
        "human_interventions": [],
        "awaiting_human_input": False,
        "global_risk_level": "low",
        "risk_factors": [],
        "safety_checks_passed": True,
        "iteration_count": 0,
        "max_iterations": 30,
        "context_size": 100,
        "consolidated_history": "",
        "error_log": [],
        "execution_status": "running",
        "final_report": "",
        "session_id": "",
        "plan": None,
        "completed_steps": [],
        "memory": {},
        "artifacts": [],
        "current_tool": None,
        "error_count": 0,
        "feedback_required": False,
        "background_mode": False,
        "browser_session": None,
        "plan_file_path": None,
        "findings_file_path": None,
        "actions_since_refresh": 0,
        "actions_since_save": 0,
        "auto_approve": False,
        # Deep Agents fields
        "current_directory": str(test_workspace),
        "filesystem_history": [],
        "todos": None,
        "active_subagents": [],
        "subagent_depth": 0,
        "evicted_results": [],
        "tool_results": [],
        "tool_name": None,
        "tool_params": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEEPAGENTS CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeepAgentsConfig:
    """Tests for DeepAgentsConfig initialization and tool retrieval."""

    def test_config_creation(self, test_workspace):
        """Test that DeepAgentsConfig can be created."""
        from middleware.deepagents_setup import DeepAgentsConfig

        config = DeepAgentsConfig(str(test_workspace))
        assert config.workspace == test_workspace

    def test_config_lazy_init(self, test_workspace):
        """Test that middlewares are lazily initialized."""
        from middleware.deepagents_setup import DeepAgentsConfig

        config = DeepAgentsConfig(str(test_workspace))
        # Should not be initialized yet
        assert config._initialized == False

    def test_tool_names_constants(self):
        """Test that tool name constants are defined."""
        from middleware.deepagents_setup import (
            FILESYSTEM_TOOLS,
            PLANNING_TOOLS,
            SUBAGENT_TOOLS,
            ALL_DEEPAGENTS_TOOLS,
        )

        assert "ls" in FILESYSTEM_TOOLS
        assert "read_file" in FILESYSTEM_TOOLS
        assert "write_file" in FILESYSTEM_TOOLS
        assert "write_todos" in PLANNING_TOOLS
        assert "task" in SUBAGENT_TOOLS
        assert len(ALL_DEEPAGENTS_TOOLS) == len(FILESYSTEM_TOOLS) + len(
            PLANNING_TOOLS
        ) + len(SUBAGENT_TOOLS)

    def test_is_deepagents_tool(self):
        """Test tool detection helper."""
        from middleware.deepagents_setup import is_deepagents_tool

        assert is_deepagents_tool("ls") == True
        assert is_deepagents_tool("read_file") == True
        assert is_deepagents_tool("write_todos") == True
        assert is_deepagents_tool("task") == True
        assert is_deepagents_tool("unknown_tool") == False

    def test_get_tool_category(self):
        """Test tool category detection."""
        from middleware.deepagents_setup import get_tool_category

        assert get_tool_category("ls") == "filesystem"
        assert get_tool_category("write_file") == "filesystem"
        assert get_tool_category("write_todos") == "planning"
        assert get_tool_category("task") == "subagent"
        assert get_tool_category("unknown") is None


# ═══════════════════════════════════════════════════════════════════════════════
# FILESYSTEM EXECUTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilesystemExecutor:
    """Tests for filesystem_executor_node."""

    def test_native_ls(self, test_workspace, sample_state):
        """Test native ls implementation."""
        from nodes.filesystem_executor import filesystem_executor_node

        # Create test files
        (test_workspace / "test.txt").write_text("hello")
        (test_workspace / "subdir").mkdir()

        # Set WORKSPACE_DIR for the test
        os.environ["WORKSPACE_DIR"] = str(test_workspace)

        state = sample_state.copy()
        state["current_action"] = "ls"
        state["action_details"] = {"path": str(test_workspace)}

        result = filesystem_executor_node(state)

        assert result["last_tool_output"] is not None
        assert "test.txt" in result["last_tool_output"]
        assert "subdir" in result["last_tool_output"]

    def test_native_write_file(self, test_workspace, sample_state):
        """Test native write_file implementation."""
        from nodes.filesystem_executor import filesystem_executor_node

        # Set WORKSPACE_DIR for the test
        os.environ["WORKSPACE_DIR"] = str(test_workspace)

        test_file = test_workspace / "new_file.txt"

        state = sample_state.copy()
        state["current_action"] = "write_file"
        state["action_details"] = {
            "path": str(test_file),
            "content": "Hello from test!",
        }

        result = filesystem_executor_node(state)

        assert result["last_tool_output"] is not None
        assert "Successfully wrote" in result["last_tool_output"]
        assert test_file.exists()
        assert test_file.read_text() == "Hello from test!"

    def test_native_read_file(self, test_workspace, sample_state):
        """Test native read_file implementation."""
        from nodes.filesystem_executor import filesystem_executor_node

        # Set WORKSPACE_DIR for the test
        os.environ["WORKSPACE_DIR"] = str(test_workspace)

        test_file = test_workspace / "read_test.txt"
        test_file.write_text("Test content here")

        state = sample_state.copy()
        state["current_action"] = "read_file"
        state["action_details"] = {"path": str(test_file)}

        result = filesystem_executor_node(state)

        assert "Test content here" in result["last_tool_output"]

    def test_native_glob(self, test_workspace, sample_state):
        """Test native glob implementation."""
        from nodes.filesystem_executor import filesystem_executor_node

        # Create test files
        (test_workspace / "file1.py").write_text("# python")
        (test_workspace / "file2.py").write_text("# python")
        (test_workspace / "other.txt").write_text("text")

        # Set WORKSPACE_DIR for the test
        os.environ["WORKSPACE_DIR"] = str(test_workspace)

        state = sample_state.copy()
        state["current_action"] = "glob"
        state["action_details"] = {"pattern": "*.py"}

        result = filesystem_executor_node(state)

        assert "file1.py" in result["last_tool_output"]
        assert "file2.py" in result["last_tool_output"]

    def test_filesystem_history_tracking(self, test_workspace, sample_state):
        """Test that filesystem operations are tracked in history."""
        from nodes.filesystem_executor import filesystem_executor_node

        state = sample_state.copy()
        state["current_action"] = "ls"
        state["action_details"] = {"path": str(test_workspace)}

        result = filesystem_executor_node(state)

        assert "filesystem_history" in result
        assert len(result["filesystem_history"]) == 1
        assert result["filesystem_history"][0]["operation"] == "ls"


# ═══════════════════════════════════════════════════════════════════════════════
# EVICTION HANDLER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvictionHandler:
    """Tests for eviction handler."""

    def test_eviction_handler_creation(self, test_workspace):
        """Test EvictionHandler can be created."""
        from middleware.eviction_handler import EvictionHandler

        handler = EvictionHandler(str(test_workspace))
        assert handler.workspace == test_workspace
        assert handler.threshold_tokens == 5000

    def test_should_evict_small_content(self, test_workspace):
        """Test that small content is not evicted."""
        from middleware.eviction_handler import EvictionHandler

        handler = EvictionHandler(str(test_workspace))
        small_content = "Small text that won't be evicted"

        assert handler.should_evict(small_content) == False

    def test_should_evict_large_content(self, test_workspace):
        """Test that large content is evicted."""
        from middleware.eviction_handler import EvictionHandler

        handler = EvictionHandler(str(test_workspace))
        # Create content > 5000 tokens (~20000 chars)
        large_content = "x" * 25000

        assert handler.should_evict(large_content) == True

    def test_evict_creates_file(self, test_workspace):
        """Test that eviction creates a file."""
        from middleware.eviction_handler import EvictionHandler

        handler = EvictionHandler(str(test_workspace))
        large_content = "y" * 30000

        result = handler.evict(large_content, "test_result")

        assert result["type"] == "evicted"
        assert "file" in result
        assert Path(result["file"]).exists()
        assert result["summary"].startswith("yy")

    def test_maybe_evict_small(self, test_workspace):
        """Test maybe_evict returns content as-is for small inputs."""
        from middleware.eviction_handler import EvictionHandler

        handler = EvictionHandler(str(test_workspace))
        small_content = "Small text"

        result = handler.maybe_evict(small_content, "test")

        assert result == small_content

    def test_process_results(self, test_workspace):
        """Test processing a list of results."""
        from middleware.eviction_handler import EvictionHandler

        handler = EvictionHandler(str(test_workspace))
        results = [
            "small result",
            "z" * 30000,  # Large
            "another small",
        ]

        processed = handler.process_results(results)

        assert len(processed) == 3
        assert processed[0] == "small result"
        assert isinstance(processed[1], dict)
        assert processed[1]["type"] == "evicted"
        assert processed[2] == "another small"


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouterDeepAgents:
    """Tests for Deep Agents routing in router.py."""

    def test_filesystem_tool_routing(self, sample_state):
        """Test that filesystem actions route to filesystem_executor."""
        from router import router

        for action in ["ls", "read_file", "write_file", "filesystem"]:
            state = sample_state.copy()
            state["current_action"] = action

            result = router(state)
            assert (
                result == "filesystem_executor"
            ), f"Action '{action}' should route to filesystem_executor"

    def test_subagent_tool_routing(self, sample_state):
        """Test that subagent actions route to subagent_executor."""
        from router import router

        for action in ["task", "subagent"]:
            state = sample_state.copy()
            state["current_action"] = action

            result = router(state)
            assert (
                result == "subagent_executor"
            ), f"Action '{action}' should route to subagent_executor"

    def test_tool_name_routing(self, sample_state):
        """Test routing via tool_name field."""
        from router import router

        state = sample_state.copy()
        state["current_action"] = ""
        state["tool_name"] = "read_file"

        result = router(state)
        assert result == "filesystem_executor"

    def test_deepagents_availability_flag(self):
        """Test that DEEPAGENTS_AVAILABLE flag is set."""
        from router import DEEPAGENTS_AVAILABLE, FILESYSTEM_TOOLS

        # Should be True if middleware imports succeed, False otherwise
        assert isinstance(DEEPAGENTS_AVAILABLE, bool)
        # Filesystem tools should always be defined
        assert "ls" in FILESYSTEM_TOOLS


# ═══════════════════════════════════════════════════════════════════════════════
# SUBAGENT EXECUTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubagentExecutor:
    """Tests for subagent_executor_node."""

    def test_subagent_missing_description(self, sample_state):
        """Test subagent handling with missing task description."""
        from nodes.subagent_executor import subagent_executor_node

        state = sample_state.copy()
        state["action_details"] = {}

        result = subagent_executor_node(state)

        assert (
            "Error" in result["last_tool_output"]
            or "error" in result["last_tool_output"].lower()
        )

    def test_subagent_depth_limit(self, sample_state):
        """Test subagent depth limit is enforced."""
        from nodes.subagent_executor import subagent_executor_node

        state = sample_state.copy()
        state["subagent_depth"] = 5  # Above max depth
        state["action_details"] = {"task_description": "Test task"}

        result = subagent_executor_node(state)

        assert (
            "depth" in result["last_tool_output"].lower()
            or "max" in result["last_tool_output"].lower()
        )

    def test_subagent_simulation(self, sample_state):
        """Test subagent simulation when Deep Agents not available."""
        from nodes.subagent_executor import subagent_executor_node

        state = sample_state.copy()
        state["action_details"] = {"task_description": "Research Python history"}

        result = subagent_executor_node(state)

        # Should return some output (either real or simulated)
        assert result["last_tool_output"] is not None
        assert len(result["last_tool_output"]) > 0

    def test_subagent_tracking(self, sample_state):
        """Test that subagent IDs are tracked."""
        from nodes.subagent_executor import subagent_executor_node

        state = sample_state.copy()
        state["action_details"] = {"task_description": "Test task"}

        result = subagent_executor_node(state)

        assert "active_subagents" in result
        assert len(result["active_subagents"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end integration tests."""

    def test_agent_state_has_deepagents_fields(self):
        """Test that AgentStateDict includes Deep Agents fields."""
        from agent_state import create_initial_state

        state = create_initial_state("Test query")

        # Verify Deep Agents fields exist
        assert "current_directory" in state
        assert "filesystem_history" in state
        assert "todos" in state
        assert "active_subagents" in state
        assert "subagent_depth" in state
        assert "evicted_results" in state
        assert "tool_results" in state
        assert "tool_name" in state
        assert "tool_params" in state

    def test_graph_includes_deepagents_nodes(self):
        """Test that agent graph includes Deep Agents nodes."""
        from agent_graph import create_agent_graph

        workflow = create_agent_graph(
            enable_deep_research=False, enable_hitl=False, enable_prompt_enhancer=False
        )

        # Check that Deep Agents nodes are registered
        node_names = list(workflow.nodes.keys())
        assert "filesystem_executor" in node_names
        assert "subagent_executor" in node_names

    def test_router_edge_map_includes_deepagents(self):
        """Test that router descriptions include Deep Agents."""
        from router import get_next_node_description

        fs_desc = get_next_node_description("filesystem_executor")
        sub_desc = get_next_node_description("subagent_executor")

        assert "filesystem" in fs_desc.lower()
        assert "sub-agent" in sub_desc.lower() or "subagent" in sub_desc.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
