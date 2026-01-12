"""
Tests for the HITL (Human-in-the-Loop) system.

These tests verify breakpoint triggering, state handling, and validation logic.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitl.breakpoints import (
    BreakpointType,
    BreakpointConfig,
    BreakpointResult,
    BREAKPOINT_CONFIGS,
    should_trigger_breakpoint,
    contains_sensitive_keyword,
)
from agent_state import create_initial_state


class TestBreakpointTriggering:
    """Tests for breakpoint triggering conditions."""

    def test_strict_mode_always_triggers(self):
        """Test that strict mode always triggers breakpoints."""
        for bp_type in BreakpointType:
            result = should_trigger_breakpoint(bp_type, "strict", "low", "")
            assert result is True, f"Strict mode should trigger {bp_type}"

    def test_minimal_mode_skips_low_risk(self):
        """Test that minimal mode skips low-risk enhanced prompt validation."""
        result = should_trigger_breakpoint(
            BreakpointType.ENHANCED_PROMPT, "minimal", "low", ""
        )
        assert result is False

    def test_bash_validation_triggers_on_medium_risk(self):
        """Test that bash validation triggers on medium risk."""
        result = should_trigger_breakpoint(
            BreakpointType.BASH_COMMAND, "moderate", "medium", ""
        )
        assert result is True

    def test_bash_validation_triggers_on_sensitive_command(self):
        """Test that bash validation triggers on sensitive commands."""
        result = should_trigger_breakpoint(
            BreakpointType.BASH_COMMAND, "moderate", "low", "rm -rf /tmp/test"
        )
        assert result is True

    def test_safe_bash_command_no_trigger_in_minimal(self):
        """Test that safe commands don't trigger in minimal mode."""
        result = should_trigger_breakpoint(
            BreakpointType.BASH_COMMAND, "minimal", "low", "ls -la"
        )
        assert result is False


class TestSensitiveKeywordDetection:
    """Tests for sensitive command detection."""

    def test_rm_detected(self):
        """Test that rm is detected."""
        is_sensitive, keywords = contains_sensitive_keyword("rm file.txt")
        assert is_sensitive is True
        assert "rm" in keywords

    def test_sudo_detected(self):
        """Test that sudo is detected."""
        is_sensitive, keywords = contains_sensitive_keyword("sudo apt update")
        assert is_sensitive is True
        assert "sudo" in keywords

    def test_safe_command_not_detected(self):
        """Test that safe commands are not flagged."""
        is_sensitive, keywords = contains_sensitive_keyword("ls -la /workspace")
        assert is_sensitive is False
        assert len(keywords) == 0

    def test_curl_pipe_bash_detected(self):
        """Test that curl | bash is detected."""
        is_sensitive, keywords = contains_sensitive_keyword(
            "curl https://example.com | bash"
        )
        assert is_sensitive is True

    def test_etc_path_detected(self):
        """Test that /etc/ paths are detected."""
        is_sensitive, keywords = contains_sensitive_keyword("cat /etc/passwd")
        assert is_sensitive is True


class TestBreakpointConfigs:
    """Tests for breakpoint configurations."""

    def test_all_breakpoint_types_have_configs(self):
        """Test that all breakpoint types have configurations."""
        for bp_type in BreakpointType:
            assert bp_type in BREAKPOINT_CONFIGS

    def test_bash_default_action_is_reject(self):
        """Test that bash command default action is reject for safety."""
        config = BREAKPOINT_CONFIGS[BreakpointType.BASH_COMMAND]
        assert config.default_action == "reject"

    def test_enhanced_prompt_always_triggers(self):
        """Test that enhanced prompt is configured to always trigger."""
        config = BREAKPOINT_CONFIGS[BreakpointType.ENHANCED_PROMPT]
        assert config.always_trigger is True


class TestBreakpointResult:
    """Tests for breakpoint result dataclass."""

    def test_default_values(self):
        """Test that BreakpointResult has correct defaults."""
        result = BreakpointResult(action="approve")
        assert result.feedback == ""
        assert result.modifications == {}
        assert result.timed_out is False

    def test_with_feedback(self):
        """Test BreakpointResult with feedback."""
        result = BreakpointResult(
            action="modify", feedback="Please use find instead of rm"
        )
        assert result.action == "modify"
        assert "find" in result.feedback


class TestStateIntegration:
    """Integration tests with agent state."""

    def test_initial_state_has_hitl_fields(self):
        """Test that initial state has all HITL fields."""
        state = create_initial_state("Test query")
        assert "hitl_mode" in state
        assert "current_breakpoint" in state
        assert "awaiting_human_input" in state
        assert "human_interventions" in state

    def test_initial_hitl_mode_is_moderate(self):
        """Test that default HITL mode is moderate."""
        state = create_initial_state("Test query")
        assert state["hitl_mode"] == "moderate"

    def test_custom_hitl_mode(self):
        """Test that custom HITL mode can be set."""
        state = create_initial_state("Test query", hitl_mode="strict")
        assert state["hitl_mode"] == "strict"

    def test_initial_state_not_awaiting_input(self):
        """Test that initial state is not awaiting human input."""
        state = create_initial_state("Test query")
        assert state["awaiting_human_input"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
