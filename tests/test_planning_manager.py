"""
Unit tests for PlanningManager node.

Uses pytest and pytest-asyncio for async test support.
All tests use temporary directories to avoid filesystem side effects.
"""

import pytest
from pathlib import Path

from nodes.planning_manager import (
    PlanningManager,
    planning_manager_node,
    refresh_plan_node,
    should_refresh_plan,
)


class TestPlanningManagerInit:
    """Tests for PlanningManager initialization."""

    def test_default_init(self):
        """Test default initialization."""
        manager = PlanningManager()
        assert manager.workspace_dir == Path("/workspace")
        assert manager.plan_path == Path("/workspace/task_plan.md")

    def test_custom_workspace(self, tmp_path):
        """Test custom workspace directory."""
        manager = PlanningManager(str(tmp_path))
        assert manager.workspace_dir == tmp_path


class TestInitializePlan:
    """Tests for initialize_plan method."""

    async def test_initialize_single_phase(self, tmp_path):
        """Test initializing a plan with default single phase."""
        manager = PlanningManager(str(tmp_path))

        result = await manager.initialize_plan("Build a REST API")

        assert result["success"] is True
        assert result["goal"] == "Build a REST API"
        assert result["phases_count"] == 1
        assert (tmp_path / "task_plan.md").exists()

        content = (tmp_path / "task_plan.md").read_text()
        assert "Build a REST API" in content
        assert "Phase 1" in content

    async def test_initialize_multi_phase(self, tmp_path):
        """Test initializing a plan with multiple phases."""
        manager = PlanningManager(str(tmp_path))

        phases = [
            {"name": "Setup", "description": "Create project structure"},
            {"name": "Develop", "description": "Write the code"},
            {"name": "Test", "description": "Run tests"},
        ]

        result = await manager.initialize_plan("Build app", phases=phases)

        assert result["phases_count"] == 3

        content = (tmp_path / "task_plan.md").read_text()
        assert "Phase 1: Setup" in content
        assert "Phase 2: Develop" in content
        assert "Phase 3: Test" in content


class TestRefreshPlan:
    """Tests for refresh_plan method (anti goal-drift)."""

    async def test_refresh_returns_goal(self, tmp_path):
        """Test that refresh returns the correct goal."""
        manager = PlanningManager(str(tmp_path))

        await manager.initialize_plan("Research Python history")
        plan = await manager.refresh_plan()

        assert plan["goal"] == "Research Python history"
        assert "raw_content" in plan

    async def test_refresh_on_nonexistent_raises(self, tmp_path):
        """Test that refresh raises if plan doesn't exist."""
        manager = PlanningManager(str(tmp_path))

        with pytest.raises(FileNotFoundError):
            await manager.refresh_plan()

    async def test_refresh_returns_phase_info(self, tmp_path):
        """Test that refresh returns phase tracking data."""
        manager = PlanningManager(str(tmp_path))

        phases = [
            {"name": "Research", "description": "Find sources"},
            {"name": "Write", "description": "Create content"},
        ]
        await manager.initialize_plan("Test goal", phases=phases)

        plan = await manager.refresh_plan()

        assert plan["current_phase"] == "Phase 1"
        assert plan["completed_count"] == 0
        assert plan["total_phases"] == 2


class TestUpdatePhaseStatus:
    """Tests for update_phase_status method."""

    async def test_mark_phase_complete(self, tmp_path):
        """Test marking a phase as complete."""
        manager = PlanningManager(str(tmp_path))

        phases = [
            {"name": "Setup", "description": "Initial setup"},
            {"name": "Build", "description": "Build it"},
        ]
        await manager.initialize_plan("Test", phases=phases)

        result = await manager.update_phase_status(1, completed=True)

        assert result["success"] is True
        assert result["completed"] is True

        # Verify file content
        content = (tmp_path / "task_plan.md").read_text()
        assert "[x] **Phase 1:" in content

    async def test_update_nonexistent_phase(self, tmp_path):
        """Test updating a phase that doesn't exist."""
        manager = PlanningManager(str(tmp_path))

        await manager.initialize_plan("Test")
        result = await manager.update_phase_status(99, completed=True)

        assert result["success"] is False
        assert "not found" in result.get("error", "")


class TestLogError:
    """Tests for log_error method."""

    async def test_log_error_basic(self, tmp_path):
        """Test logging a basic error."""
        manager = PlanningManager(str(tmp_path))

        await manager.initialize_plan("Test")
        result = await manager.log_error("Connection failed")

        assert result["success"] is True

        content = (tmp_path / "task_plan.md").read_text()
        assert "Connection failed" in content
        assert "No errors logged" not in content

    async def test_log_error_with_solution(self, tmp_path):
        """Test logging error with solution and result."""
        manager = PlanningManager(str(tmp_path))

        await manager.initialize_plan("Test")
        await manager.log_error(
            "API timeout", solution="Increase timeout", result="Fixed"
        )

        content = (tmp_path / "task_plan.md").read_text()
        assert "API timeout" in content
        assert "Increase timeout" in content
        assert "Fixed" in content


class TestSaveFindings:
    """Tests for save_findings method (2-Action Rule)."""

    async def test_save_findings_creates_file(self, tmp_path):
        """Test that save_findings creates a findings file."""
        manager = PlanningManager(str(tmp_path))

        result = await manager.save_findings(
            query="Python history",
            sources=["Wikipedia", "Python.org"],
            discoveries=["Created 1991", "By Guido van Rossum"],
            links=["https://python.org"],
        )

        assert result["success"] is True
        assert (tmp_path / "findings.md").exists()

        content = (tmp_path / "findings.md").read_text()
        assert "Python history" in content
        assert "Wikipedia" in content
        assert "Created 1991" in content

    async def test_save_findings_appends(self, tmp_path):
        """Test that save_findings appends to existing file."""
        manager = PlanningManager(str(tmp_path))

        await manager.save_findings(
            query="First query",
            sources=["Source 1"],
            discoveries=["Discovery 1"],
        )

        await manager.save_findings(
            query="Second query",
            sources=["Source 2"],
            discoveries=["Discovery 2"],
        )

        content = (tmp_path / "findings.md").read_text()
        assert "First query" in content
        assert "Second query" in content


class TestLogAction:
    """Tests for log_action method."""

    async def test_log_action_creates_progress(self, tmp_path):
        """Test that log_action creates progress file."""
        manager = PlanningManager(str(tmp_path))

        result = await manager.log_action(
            "Executed pip install", "Successfully installed package", success=True
        )

        assert result["success"] is True
        assert (tmp_path / "progress.md").exists()

    async def test_log_action_tracks_counts(self, tmp_path):
        """Test that log_action tracks pass/fail counts."""
        manager = PlanningManager(str(tmp_path))

        await manager.log_action("Action 1", "Pass", success=True)
        await manager.log_action("Action 2", "Pass", success=True)
        await manager.log_action("Action 3", "Fail", success=False)

        content = (tmp_path / "progress.md").read_text()
        assert "Passed: 2" in content
        assert "Failed: 1" in content


class TestShouldRefreshPlan:
    """Tests for should_refresh_plan conditional function."""

    def test_refresh_after_many_actions(self):
        """Test refresh triggers after 10+ actions."""
        state = {"actions_since_refresh": 15, "current_tool": "search"}
        assert should_refresh_plan(state) == "refresh"

    def test_refresh_on_critical_tool(self):
        """Test refresh triggers on critical tools."""
        state = {"actions_since_refresh": 2, "current_tool": "bash"}
        assert should_refresh_plan(state) == "refresh"

        state = {"actions_since_refresh": 2, "current_tool": "edit"}
        assert should_refresh_plan(state) == "refresh"

    def test_skip_on_normal_action(self):
        """Test skip when conditions not met."""
        state = {"actions_since_refresh": 5, "current_tool": "search"}
        assert should_refresh_plan(state) == "skip"


class TestPlanningManagerNode:
    """Tests for LangGraph node functions."""

    async def test_planning_manager_node(self, tmp_path):
        """Test the planning_manager_node function."""
        state = {
            "workspace_dir": str(tmp_path),
            "enhanced_query": "Build a web scraper",
        }

        result = await planning_manager_node(state)

        assert "plan_file_path" in result
        assert result["actions_since_refresh"] == 0
        assert (tmp_path / "task_plan.md").exists()

    async def test_refresh_plan_node(self, tmp_path):
        """Test the refresh_plan_node function."""
        # First initialize
        state = {
            "workspace_dir": str(tmp_path),
            "enhanced_query": "Test goal",
        }
        await planning_manager_node(state)

        # Then refresh
        result = await refresh_plan_node(state)

        assert "messages" in result
        assert result["actions_since_refresh"] == 0
        assert any("GOAL REFRESH" in str(m) for m in result.get("messages", []))
