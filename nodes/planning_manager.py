"""
Planning Manager Node - Filesystem-based anti-goal-drift for Manus agent.

Implements the "$2B pattern" by persisting plans to disk and re-reading
them before critical decisions to prevent context drift in long sessions.

The core insight: Context = RAM (volatile, limited) → Goal Drift after 50 calls
                 Filesystem = Disk (persistent) → RE-READ plan = Goal refresh
"""

import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))


class PlanningManager:
    """
    Manages filesystem-based task planning for context persistence.

    This is the core of the "$2B pattern" - by persisting plans to disk
    and re-reading them before critical decisions, we prevent goal drift
    that occurs when LLM context becomes saturated.

    Attributes:
        workspace_dir: Directory for plan files.
        plan_path: Path to task_plan.md file.
        findings_path: Path to findings.md file.
        progress_path: Path to progress.md file.

    Example:
        >>> manager = PlanningManager("/workspace")
        >>> await manager.initialize_plan("Build a REST API")
        >>> # After many operations...
        >>> plan_data = await manager.refresh_plan()  # Re-reads from disk!
        >>> print(plan_data["goal"])  # Goal is fresh, not drifted
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        """
        Initialize the planning manager.

        Args:
            workspace_dir: Directory to store plan files.
                           If None, tries /workspace, then WORKSPACE_DIR env,
                           then ./workspace (local fallback).
        """
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
        else:
            # Smart fallback: prefer container path, then env, then local
            default_path = Path("/workspace")
            env_path = os.getenv("WORKSPACE_DIR")

            if env_path:
                self.workspace_dir = Path(env_path)
            elif default_path.exists() and os.access(default_path, os.W_OK):
                self.workspace_dir = default_path
            else:
                # Local fallback - use current directory + workspace
                self.workspace_dir = Path.cwd() / "workspace"

        self.plan_path = self.workspace_dir / "task_plan.md"
        self.findings_path = self.workspace_dir / "findings.md"
        self.progress_path = self.workspace_dir / "progress.md"
        self._templates_dir = (
            Path(__file__).parent.parent / "skills" / "planning" / "templates"
        )

    async def initialize_plan(
        self,
        goal: str,
        phases: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """
        Create a new task plan from the goal.

        Args:
            goal: The main objective for this task.
            phases: Optional list of phase dicts with 'name' and 'description'.
                   If not provided, creates a single-phase plan.

        Returns:
            Dict with plan metadata including file path.

        Example:
            >>> result = await manager.initialize_plan(
            ...     "Research Python history",
            ...     phases=[
            ...         {"name": "Research", "description": "Find key sources"},
            ...         {"name": "Compile", "description": "Create summary"}
            ...     ]
            ... )
            >>> print(result["plan_path"])
        """
        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Generate phases markdown
        if phases:
            phases_md = "\n".join(
                f"- [ ] **Phase {i+1}: {p['name']}** - {p.get('description', '')}"
                for i, p in enumerate(phases)
            )
            total = len(phases)
        else:
            phases_md = f"- [ ] **Phase 1: Execute** - {goal[:100]}"
            total = 1

        # Load template
        template_path = self._templates_dir / "task_plan.md"
        if template_path.exists():
            template = template_path.read_text()
        else:
            template = self._default_plan_template()

        # Fill template
        timestamp = datetime.now().isoformat()
        content = (
            template.replace("{goal}", goal)
            .replace("{phases}", phases_md)
            .replace("{current_phase}", "Phase 1")
            .replace("{completed}", "0")
            .replace("{total}", str(total))
            .replace("{errors}", "_No errors logged._")
            .replace("{timestamp}", timestamp)
        )

        # Write to disk
        await self._write_file(self.plan_path, content)

        logger.info(f"Initialized plan at {self.plan_path}")

        return {
            "success": True,
            "plan_path": str(self.plan_path),
            "goal": goal,
            "phases_count": total,
            "timestamp": timestamp,
        }

    async def refresh_plan(self) -> dict[str, Any]:
        """
        RE-READ plan from disk to refresh context (anti goal-drift).

        This is the KEY method of the $2B pattern. Call this before
        any critical decision to ensure the goal hasn't drifted.

        Returns:
            Dict with goal, current_phase, phases, errors, and raw_content.

        Raises:
            FileNotFoundError: If plan file doesn't exist.

        Example:
            >>> plan = await manager.refresh_plan()
            >>> print(f"Current goal: {plan['goal']}")
            >>> print(f"Current phase: {plan['current_phase']}")
        """
        if not self.plan_path.exists():
            raise FileNotFoundError(f"Plan file not found: {self.plan_path}")

        content = await self._read_file(self.plan_path)
        parsed = self._parse_plan(content)

        logger.debug(f"Plan refreshed: goal='{parsed['goal'][:50]}...'")

        return {
            "goal": parsed["goal"],
            "current_phase": parsed["current_phase"],
            "completed_count": parsed["completed"],
            "total_phases": parsed["total"],
            "phases": parsed["phases"],
            "errors": parsed["errors"],
            "raw_content": content,  # For direct context injection
        }

    async def update_phase_status(
        self,
        phase_number: int,
        completed: bool = True,
    ) -> dict[str, Any]:
        """
        Update the completion status of a phase.

        Args:
            phase_number: 1-indexed phase number.
            completed: True to mark complete, False to mark incomplete.

        Returns:
            Dict with success status and updated counts.

        Example:
            >>> await manager.update_phase_status(1, completed=True)
            >>> # Phase 1 is now marked [x] in task_plan.md
        """
        if not self.plan_path.exists():
            return {"success": False, "error": "Plan not initialized"}

        content = await self._read_file(self.plan_path)

        # Find and update the phase checkbox
        old_marker = "[ ]" if completed else "[x]"
        new_marker = "[x]" if completed else "[ ]"

        # Pattern: - [ ] **Phase N:
        pattern = rf"(- ){re.escape(old_marker)}( \*\*Phase {phase_number}:)"
        replacement = rf"\1{new_marker}\2"

        updated_content, count = re.subn(pattern, replacement, content)

        if count == 0:
            return {
                "success": False,
                "error": f"Phase {phase_number} not found or already updated",
            }

        # Update the completed count in Progress section
        parsed = self._parse_plan(updated_content)
        new_completed = sum(1 for p in parsed["phases"] if p.get("completed", False))
        updated_content = re.sub(
            r"\*\*Completed:\*\* \d+/\d+",
            f"**Completed:** {new_completed}/{parsed['total']}",
            updated_content,
        )

        # Update timestamp
        updated_content = self._update_timestamp(updated_content)

        await self._write_file(self.plan_path, updated_content)

        logger.info(
            f"Phase {phase_number} marked {'complete' if completed else 'incomplete'}"
        )

        return {
            "success": True,
            "phase": phase_number,
            "completed": completed,
            "total_completed": new_completed,
            "total_phases": parsed["total"],
        }

    async def log_error(
        self,
        error: str,
        solution: Optional[str] = None,
        result: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Log an error to the task plan.

        Args:
            error: Description of the error.
            solution: Optional attempted solution.
            result: Optional result of the solution.

        Returns:
            Dict with success status.

        Example:
            >>> await manager.log_error(
            ...     "API returned 500",
            ...     solution="Retry with exponential backoff",
            ...     result="Succeeded on 3rd attempt"
            ... )
        """
        if not self.plan_path.exists():
            return {"success": False, "error": "Plan not initialized"}

        content = await self._read_file(self.plan_path)

        # Build error entry
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_entry = f"\n### [{timestamp}] {error}\n"
        if solution:
            error_entry += f"- **Solution:** {solution}\n"
        if result:
            error_entry += f"- **Result:** {result}\n"

        # Find errors section and append
        if "_No errors logged._" in content:
            content = content.replace("_No errors logged._", error_entry.strip())
        else:
            # Append before Last Updated section
            content = re.sub(
                r"(## 🕒 Last Updated)",
                f"{error_entry}\n\\1",
                content,
            )

        content = self._update_timestamp(content)
        await self._write_file(self.plan_path, content)

        logger.warning(f"Error logged: {error[:50]}...")

        return {"success": True, "error_logged": error}

    async def save_findings(
        self,
        query: str,
        sources: list[str],
        discoveries: list[str],
        links: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Save research findings to disk (2-Action Rule output).

        Called after every 2 search/view/browse actions to persist
        discovered information.

        Args:
            query: The research query.
            sources: List of source descriptions.
            discoveries: List of key discoveries.
            links: Optional list of relevant URLs.

        Returns:
            Dict with success status and file path.

        Example:
            >>> await manager.save_findings(
            ...     "Python history",
            ...     sources=["Wikipedia", "Python.org"],
            ...     discoveries=["Created in 1991", "By Guido van Rossum"],
            ...     links=["https://python.org/about/"]
            ... )
        """
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Load template
        template_path = self._templates_dir / "findings.md"
        if template_path.exists():
            template = template_path.read_text()
        else:
            template = self._default_findings_template()

        # Format content
        sources_md = (
            "\n".join(f"- {s}" for s in sources) if sources else "_No sources._"
        )
        discoveries_md = (
            "\n".join(f"- {d}" for d in discoveries)
            if discoveries
            else "_No discoveries._"
        )
        links_md = "\n".join(f"- {l}" for l in links) if links else "_No links._"
        timestamp = datetime.now().isoformat()

        content = (
            template.replace("{query}", query)
            .replace("{sources}", sources_md)
            .replace("{discoveries}", discoveries_md)
            .replace("{links}", links_md)
            .replace("{timestamp}", timestamp)
        )

        # Append mode if file exists
        if self.findings_path.exists():
            existing = await self._read_file(self.findings_path)
            # Append new findings section
            content = existing + f"\n\n---\n\n{content}"

        await self._write_file(self.findings_path, content)

        logger.info(f"Findings saved for query: {query[:50]}...")

        return {
            "success": True,
            "findings_path": str(self.findings_path),
            "sources_count": len(sources),
            "discoveries_count": len(discoveries),
        }

    async def log_action(
        self,
        action: str,
        result: str,
        success: bool = True,
    ) -> dict[str, Any]:
        """
        Log an action to the progress file.

        Args:
            action: Description of the action taken.
            result: Outcome of the action.
            success: Whether the action succeeded.

        Returns:
            Dict with success status.

        Example:
            >>> await manager.log_action(
            ...     "Executed: pip install requests",
            ...     "Successfully installed requests==2.28.0",
            ...     success=True
            ... )
        """
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Initialize progress file if needed
        if not self.progress_path.exists():
            template_path = self._templates_dir / "progress.md"
            if template_path.exists():
                template = template_path.read_text()
            else:
                template = self._default_progress_template()

            start_time = datetime.now().isoformat()
            content = (
                template.replace("{start_time}", start_time)
                .replace("{actions}", "")
                .replace("{passed}", "0")
                .replace("{failed}", "0")
                .replace("{timestamp}", start_time)
            )
            await self._write_file(self.progress_path, content)

        content = await self._read_file(self.progress_path)

        # Build action entry
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = "✅" if success else "❌"
        action_entry = f"\n### [{timestamp}] {status} {action}\n- {result}\n"

        # Append to actions section
        content = re.sub(
            r"(## 📝 Actions\n)",
            f"\\1{action_entry}",
            content,
        )

        # Update test counts
        passed_match = re.search(r"✅ Passed: (\d+)", content)
        failed_match = re.search(r"❌ Failed: (\d+)", content)
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0

        if success:
            passed += 1
        else:
            failed += 1

        content = re.sub(r"✅ Passed: \d+", f"✅ Passed: {passed}", content)
        content = re.sub(r"❌ Failed: \d+", f"❌ Failed: {failed}", content)

        content = self._update_timestamp(content)
        await self._write_file(self.progress_path, content)

        return {"success": True, "total_actions": passed + failed}

    def _parse_plan(self, content: str) -> dict[str, Any]:
        """Parse a task plan markdown file into structured data."""
        result = {
            "goal": "",
            "current_phase": "",
            "completed": 0,
            "total": 0,
            "phases": [],
            "errors": [],
        }

        # Extract goal
        goal_match = re.search(r"## 🎯 Goal\n(.+?)(?=\n##|\Z)", content, re.DOTALL)
        if goal_match:
            result["goal"] = goal_match.group(1).strip()

        # Extract current phase
        phase_match = re.search(r"\*\*Current Phase:\*\* (.+?)(?=\n|$)", content)
        if phase_match:
            result["current_phase"] = phase_match.group(1).strip()

        # Extract progress counts
        progress_match = re.search(r"\*\*Completed:\*\* (\d+)/(\d+)", content)
        if progress_match:
            result["completed"] = int(progress_match.group(1))
            result["total"] = int(progress_match.group(2))

        # Extract phases
        phase_pattern = r"- \[([ x])\] \*\*Phase (\d+): ([^*]+)\*\*(?:\s*-\s*(.+))?"
        for match in re.finditer(phase_pattern, content):
            result["phases"].append(
                {
                    "completed": match.group(1) == "x",
                    "number": int(match.group(2)),
                    "name": match.group(3).strip(),
                    "description": match.group(4).strip() if match.group(4) else "",
                }
            )

        # Count phases if not in progress section
        if result["total"] == 0:
            result["total"] = len(result["phases"])

        return result

    def _update_timestamp(self, content: str) -> str:
        """Update the Last Updated timestamp in content."""
        timestamp = datetime.now().isoformat()
        return re.sub(
            r"(## 🕒 Last Updated\n).+",
            f"\\g<1>{timestamp}",
            content,
        )

    async def _read_file(self, path: Path) -> str:
        """Read file content asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, path.read_text)

    async def _write_file(self, path: Path, content: str) -> None:
        """Write file content asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, path.write_text, content)

    def _default_plan_template(self) -> str:
        """Default template if file template not found."""
        return """# Task Plan: {goal}

## 🎯 Goal
{goal}

## 📋 Phases

{phases}

## 📊 Progress
- **Current Phase:** {current_phase}
- **Completed:** {completed}/{total}

## ❌ Errors

{errors}

## 🕒 Last Updated
{timestamp}
"""

    def _default_findings_template(self) -> str:
        """Default findings template."""
        return """# Findings: {query}

## 📚 Sources

{sources}

## 🔍 Discoveries

{discoveries}

## 🔗 Related Links

{links}

## 🕒 Last Updated
{timestamp}
"""

    def _default_progress_template(self) -> str:
        """Default progress template."""
        return """# Session Progress Log

## ⏰ Session Started
{start_time}

## 📝 Actions

{actions}

## 🧪 Tests
- ✅ Passed: {passed}
- ❌ Failed: {failed}

## 🕒 Last Updated
{timestamp}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Global manager instance
_planning_manager: Optional[PlanningManager] = None


def _get_manager(workspace_dir: Optional[str] = None) -> PlanningManager:
    """Get or create the global planning manager instance."""
    global _planning_manager
    if _planning_manager is None:
        _planning_manager = PlanningManager(workspace_dir)
    return _planning_manager


def planning_manager_node(state: dict) -> dict:
    """
    LangGraph node to initialize a task plan.

    Reads the user query from state and creates a task_plan.md file.

    Args:
        state: Agent state dict with 'original_query' or 'enhanced_query'.

    Returns:
        Updated state with plan_file_path set.
    """
    # Get workspace from state if explicitly provided, else let smart fallback work
    workspace = state.get("workspace_dir") or os.getenv("WORKSPACE_DIR") or None
    manager = _get_manager(workspace)

    # Use enhanced query if available, otherwise original
    goal = state.get("enhanced_query") or state.get("original_query", "Unknown task")

    # Run async function synchronously
    async def _init():
        return await manager.initialize_plan(goal)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is already running, create a new one in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = executor.submit(asyncio.run, _init()).result()
        else:
            result = loop.run_until_complete(_init())
    except RuntimeError:
        # No event loop, create one
        result = asyncio.run(_init())

    logger.info(f"Plan initialized: {result['plan_path']}")

    return {
        "plan_file_path": result["plan_path"],
        "actions_since_refresh": 0,
        "actions_since_save": 0,
    }


def refresh_plan_node(state: dict) -> dict:
    """
    LangGraph node to refresh plan from disk (anti goal-drift).

    Re-reads task_plan.md to ensure goal hasn't drifted in context.

    Args:
        state: Agent state dict with plan_file_path.

    Returns:
        Updated state with refreshed plan data.
    """
    workspace = state.get("workspace_dir", os.getenv("WORKSPACE_DIR", "/workspace"))
    manager = _get_manager(workspace)

    async def _refresh():
        try:
            plan_data = await manager.refresh_plan()

            # Inject goal reminder into messages
            goal_reminder = {
                "role": "system",
                "content": (
                    f"[GOAL REFRESH] Current objective: {plan_data['goal']}\n"
                    f"Current phase: {plan_data['current_phase']}\n"
                    f"Progress: {plan_data['completed_count']}/{plan_data['total_phases']}"
                ),
            }

            logger.debug("Plan refreshed from disk")

            return {
                "messages": [goal_reminder],
                "actions_since_refresh": 0,
                "current_phase": plan_data["current_phase"],
            }

        except FileNotFoundError:
            logger.warning("Plan file not found during refresh")
            return {"actions_since_refresh": 0}

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(asyncio.run, _refresh()).result()
        else:
            return loop.run_until_complete(_refresh())
    except RuntimeError:
        return asyncio.run(_refresh())


def should_refresh_plan(state: dict) -> str:
    """
    Conditional edge function to determine if plan should be refreshed.

    Refresh triggers:
    - More than 10 actions since last refresh
    - Current tool is a critical operation (bash, write, edit)

    Args:
        state: Current agent state.

    Returns:
        "refresh" to go to refresh_plan_node, "skip" to bypass.
    """
    actions = state.get("actions_since_refresh", 0)
    current_tool = state.get("current_tool") or ""

    # Critical tools that warrant a goal check
    critical_tools = {"bash", "write", "edit", "delete", "deploy"}

    if actions > 10:
        logger.debug(f"Triggering refresh: {actions} actions since last refresh")
        return "refresh"

    if current_tool.lower() in critical_tools:
        logger.debug(f"Triggering refresh: critical tool '{current_tool}'")
        return "refresh"

    return "skip"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


async def increment_action_counter(state: dict) -> dict:
    """
    Utility to increment action counters in state.

    Call this after each action to track when refresh/save is needed.

    Args:
        state: Current agent state.

    Returns:
        State update with incremented counters.
    """
    return {
        "actions_since_refresh": state.get("actions_since_refresh", 0) + 1,
        "actions_since_save": state.get("actions_since_save", 0) + 1,
    }


if __name__ == "__main__":
    # Quick test
    import asyncio

    async def test():
        manager = PlanningManager("/tmp/test_workspace")

        # Initialize
        result = await manager.initialize_plan(
            "Build a REST API with authentication",
            phases=[
                {"name": "Setup", "description": "Create project structure"},
                {"name": "Auth", "description": "Implement authentication"},
                {"name": "API", "description": "Build endpoints"},
            ],
        )
        print(f"✓ Plan initialized: {result['plan_path']}")

        # Refresh
        plan = await manager.refresh_plan()
        print(f"✓ Goal: {plan['goal'][:50]}...")
        print(f"✓ Phases: {len(plan['phases'])}")

        # Update phase
        await manager.update_phase_status(1, completed=True)
        print("✓ Phase 1 marked complete")

        # Log action
        await manager.log_action("Test action", "Test result", success=True)
        print("✓ Action logged")

        # Save findings
        await manager.save_findings(
            "Test query",
            sources=["Source 1"],
            discoveries=["Discovery 1"],
        )
        print("✓ Findings saved")

        print("\n✓ All tests passed!")

    asyncio.run(test())
