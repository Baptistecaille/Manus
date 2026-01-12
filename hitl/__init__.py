"""
HITL - Human-in-the-Loop module for Manus Agent.

Provides breakpoint management, CLI interface, and user interaction handling.
"""

from hitl.breakpoints import (
    BreakpointType,
    BreakpointConfig,
    BreakpointResult,
    should_trigger_breakpoint,
    BREAKPOINT_CONFIGS,
)
from hitl.handler import hitl_handler_node

__all__ = [
    "BreakpointType",
    "BreakpointConfig",
    "BreakpointResult",
    "should_trigger_breakpoint",
    "BREAKPOINT_CONFIGS",
    "hitl_handler_node",
]
