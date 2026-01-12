"""
Breakpoints - HITL breakpoint definitions and configuration.

Defines the types of breakpoints, their configuration, and triggering logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class BreakpointType(str, Enum):
    """Types of HITL breakpoints."""

    ENHANCED_PROMPT = "enhanced_prompt_validation"
    PLAN_VALIDATION = "plan_validation"
    BASH_COMMAND = "bash_command_validation"


@dataclass
class BreakpointConfig:
    """Configuration for a breakpoint."""

    breakpoint_type: BreakpointType
    timeout_seconds: int = 180
    default_action: Literal["approve", "reject", "skip"] = "approve"
    warning_at_seconds: int = 30
    description: str = ""

    # Conditions for triggering
    always_trigger: bool = False
    trigger_on_risk_levels: list[str] = field(default_factory=list)


@dataclass
class BreakpointResult:
    """Result of a breakpoint interaction."""

    action: Literal["approve", "reject", "modify", "skip", "quit"]
    feedback: str = ""
    modifications: dict = field(default_factory=dict)
    response_time_seconds: float = 0.0
    timed_out: bool = False


# Default breakpoint configurations
BREAKPOINT_CONFIGS = {
    BreakpointType.ENHANCED_PROMPT: BreakpointConfig(
        breakpoint_type=BreakpointType.ENHANCED_PROMPT,
        timeout_seconds=300,
        default_action="approve",
        warning_at_seconds=60,
        description="Validate the enhanced/optimized query before planning",
        always_trigger=True,  # Always trigger after prompt enhancement
    ),
    BreakpointType.PLAN_VALIDATION: BreakpointConfig(
        breakpoint_type=BreakpointType.PLAN_VALIDATION,
        timeout_seconds=180,
        default_action="approve",
        warning_at_seconds=45,
        description="Validate the execution plan before proceeding",
        always_trigger=True,  # Always show plan for validation
    ),
    BreakpointType.BASH_COMMAND: BreakpointConfig(
        breakpoint_type=BreakpointType.BASH_COMMAND,
        timeout_seconds=60,
        default_action="reject",  # Safety: reject by default
        warning_at_seconds=15,
        description="Validate potentially risky bash commands",
        trigger_on_risk_levels=["medium", "high", "critical"],
    ),
}


# Sensitive keywords that trigger bash command validation
BASH_SENSITIVE_KEYWORDS = [
    # Destructive commands
    "rm",
    "rmdir",
    "unlink",
    "dd",
    "mkfs",
    "fdisk",
    "parted",
    # Privilege escalation
    "sudo",
    "su",
    "doas",
    "chmod",
    "chown",
    "chgrp",
    # Network operations
    "curl | bash",
    "curl | sh",
    "wget | bash",
    "wget | sh",
    "nc",
    "netcat",
    # Dangerous patterns
    "eval",
    "exec",
    "> /dev/",
    "| /dev/",
    # System paths
    "/etc/",
    "/sys/",
    "/proc/",
    "/boot/",
    "/var/log/",
    # Process control
    "kill",
    "pkill",
    "killall",
    "reboot",
    "shutdown",
    "init",
]


def contains_sensitive_keyword(command: str) -> tuple[bool, list[str]]:
    """
    Check if a command contains sensitive keywords.

    Args:
        command: The bash command to check.

    Returns:
        Tuple of (is_sensitive, list of matched keywords).
    """
    command_lower = command.lower()
    matched = []

    for keyword in BASH_SENSITIVE_KEYWORDS:
        if keyword.lower() in command_lower:
            matched.append(keyword)

    return len(matched) > 0, matched


def should_trigger_breakpoint(
    breakpoint_type: BreakpointType,
    hitl_mode: str,
    risk_level: str = "low",
    command: str = "",
) -> bool:
    """
    Determine if a breakpoint should be triggered.

    Args:
        breakpoint_type: Type of breakpoint to check.
        hitl_mode: Current HITL mode (strict, moderate, minimal).
        risk_level: Current risk level.
        command: Command string (for bash validation).

    Returns:
        True if the breakpoint should trigger.
    """
    config = BREAKPOINT_CONFIGS.get(breakpoint_type)
    if not config:
        return False

    # In strict mode, always trigger all breakpoints
    if hitl_mode == "strict":
        return True

    # Handle always_trigger
    if config.always_trigger:
        # In minimal mode, skip even always_trigger breakpoints if risk is low
        if hitl_mode == "minimal" and risk_level == "low":
            return False
        return True

    # Check risk level conditions
    if config.trigger_on_risk_levels and risk_level in config.trigger_on_risk_levels:
        return True

    # Special case for bash commands - check for sensitive keywords
    if breakpoint_type == BreakpointType.BASH_COMMAND and command:
        is_sensitive, _ = contains_sensitive_keyword(command)
        if is_sensitive:
            return True

    # In minimal mode, don't trigger optional breakpoints
    if hitl_mode == "minimal":
        return False

    # Moderate mode - trigger based on risk
    if hitl_mode == "moderate":
        if risk_level in ["medium", "high", "critical"]:
            return True

    return False


if __name__ == "__main__":
    # Quick test
    print("Testing breakpoint triggering logic...")

    test_cases = [
        (BreakpointType.ENHANCED_PROMPT, "strict", "low", ""),
        (BreakpointType.ENHANCED_PROMPT, "minimal", "low", ""),
        (BreakpointType.BASH_COMMAND, "moderate", "low", "ls -la"),
        (BreakpointType.BASH_COMMAND, "moderate", "low", "rm -rf /tmp/test"),
        (BreakpointType.BASH_COMMAND, "minimal", "high", "echo hello"),
    ]

    for bp_type, mode, risk, cmd in test_cases:
        result = should_trigger_breakpoint(bp_type, mode, risk, cmd)
        print(
            f"{bp_type.value} | mode={mode} | risk={risk} | cmd='{cmd[:20]}' → {result}"
        )
