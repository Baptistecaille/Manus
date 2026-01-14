"""
Takeover Mode - Manual control mode for Manus agent.

Allows users to temporarily take control of agent execution,
execute manual commands, and resume automation.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


class TakeoverMode:
    """
    Manual takeover mode for agent execution.

    Allows users to pause automation, execute commands manually,
    and resume agent control.

    Attributes:
        active_sessions: Dict of session_id to takeover state.

    Example:
        >>> takeover = TakeoverMode()
        >>> await takeover.enable_takeover("session123")
        >>> result = await takeover.execute_manual_command("ls -la")
        >>> await takeover.resume_automation("session123")
    """

    def __init__(self) -> None:
        """Initialize takeover mode."""
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.console = Console()
        logger.debug("TakeoverMode initialized")

    async def enable_takeover(
        self,
        session_id: str,
        reason: str = "User requested manual control",
    ) -> bool:
        """
        Enable takeover mode for a session.

        Pauses agent automation and gives control to the user.

        Args:
            session_id: Session identifier.
            reason: Reason for enabling takeover.

        Returns:
            True if takeover enabled successfully.

        Example:
            >>> success = await takeover.enable_takeover("sess123", "Debugging issue")
        """
        if not session_id:
            raise ValueError("Session ID cannot be empty")

        logger.info(f"Enabling takeover for session: {session_id}")

        self.active_sessions[session_id] = {
            "enabled": True,
            "enabled_at": datetime.now().isoformat(),
            "reason": reason,
            "command_history": [],
        }

        self.console.print()
        self.console.print(
            Panel(
                f"[bold yellow]📌 Takeover Mode Enabled[/bold yellow]\n\n"
                f"Session: {session_id}\n"
                f"Reason: {reason}\n\n"
                f"[dim]You now have manual control. Use execute_manual_command() "
                f"to run commands, or resume_automation() to return control to the agent.[/dim]",
                border_style="yellow",
            )
        )

        return True

    async def execute_manual_command(
        self,
        command: str,
        session_id: Optional[str] = None,
        working_dir: str = "/workspace",
    ) -> dict[str, Any]:
        """
        Execute a command manually during takeover mode.

        Args:
            command: Command to execute.
            session_id: Optional session ID for tracking.
            working_dir: Working directory for command.

        Returns:
            Dict with command result.

        Example:
            >>> result = await takeover.execute_manual_command("python --version")
            >>> print(result["stdout"])
        """
        if not command:
            raise ValueError("Command cannot be empty")

        logger.info(f"Manual command: {command}")

        # Record in session history if tracking
        if session_id and session_id in self.active_sessions:
            self.active_sessions[session_id]["command_history"].append(
                {
                    "command": command,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Execute command
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60,  # 60 second timeout
            )

            result = {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "command": command,
            }

            # Display result
            if result["success"]:
                self.console.print(f"\n[green]✓ Command succeeded[/green]")
            else:
                self.console.print(
                    f"\n[red]✗ Command failed (code {result['return_code']})[/red]"
                )

            if result["stdout"]:
                self.console.print(
                    Panel(result["stdout"], title="Output", border_style="dim")
                )
            if result["stderr"]:
                self.console.print(
                    Panel(result["stderr"], title="Errors", border_style="red")
                )

            return result

        except asyncio.TimeoutError:
            logger.error(f"Command timeout: {command}")
            return {
                "success": False,
                "error": "timeout",
                "message": "Command timed out after 60 seconds",
                "command": command,
            }
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command,
            }

    async def resume_automation(
        self,
        session_id: str,
        with_context: bool = True,
    ) -> bool:
        """
        Resume agent automation after takeover.

        Args:
            session_id: Session identifier.
            with_context: If True, pass takeover context back to agent.

        Returns:
            True if automation resumed successfully.

        Example:
            >>> await takeover.resume_automation("sess123")
        """
        if not session_id:
            raise ValueError("Session ID cannot be empty")

        if session_id not in self.active_sessions:
            logger.warning(f"Session not in takeover mode: {session_id}")
            return False

        session = self.active_sessions[session_id]
        session["enabled"] = False
        session["disabled_at"] = datetime.now().isoformat()

        logger.info(f"Resuming automation for session: {session_id}")

        # Display summary
        commands_run = len(session.get("command_history", []))
        self.console.print()
        self.console.print(
            Panel(
                f"[bold green]▶ Automation Resumed[/bold green]\n\n"
                f"Session: {session_id}\n"
                f"Commands executed during takeover: {commands_run}\n\n"
                f"[dim]Agent has regained control.[/dim]",
                border_style="green",
            )
        )

        # Remove from active sessions
        del self.active_sessions[session_id]

        return True

    def is_takeover_active(self, session_id: str) -> bool:
        """
        Check if takeover mode is active for a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if takeover is active.
        """
        session = self.active_sessions.get(session_id)
        return session is not None and session.get("enabled", False)

    def get_takeover_context(self, session_id: str) -> dict[str, Any]:
        """
        Get context from takeover session.

        Useful for passing manual execution context back to the agent.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with takeover context.
        """
        session = self.active_sessions.get(session_id, {})
        return {
            "was_in_takeover": session_id in self.active_sessions,
            "command_history": session.get("command_history", []),
            "reason": session.get("reason", ""),
            "enabled_at": session.get("enabled_at"),
        }

    async def interactive_session(
        self,
        session_id: str,
        working_dir: str = "/workspace",
    ) -> dict[str, Any]:
        """
        Start an interactive takeover session.

        Provides a REPL-like interface for manual command execution.

        Args:
            session_id: Session identifier.
            working_dir: Working directory for commands.

        Returns:
            Session summary dict.
        """
        await self.enable_takeover(session_id, "Interactive session started")

        self.console.print(
            "\n[cyan]Enter commands (type 'exit' to resume automation):[/cyan]\n"
        )

        commands_executed = 0

        while True:
            try:
                command = Prompt.ask("[bold]Command[/bold]")

                if command.lower() in ("exit", "quit", "resume"):
                    break

                if command.strip():
                    await self.execute_manual_command(command, session_id, working_dir)
                    commands_executed += 1

            except KeyboardInterrupt:
                self.console.print(
                    "\n[yellow]Interrupted. Resuming automation...[/yellow]"
                )
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

        await self.resume_automation(session_id)

        return {
            "session_id": session_id,
            "commands_executed": commands_executed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_takeover_instance: Optional[TakeoverMode] = None


def get_takeover_mode() -> TakeoverMode:
    """Get or create singleton TakeoverMode instance."""
    global _takeover_instance
    if _takeover_instance is None:
        _takeover_instance = TakeoverMode()
    return _takeover_instance


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    async def test_takeover_mode():
        """Test takeover mode capabilities."""
        print("=== Takeover Mode Test ===\n")

        takeover = TakeoverMode()
        session_id = "test_session"

        # Enable takeover
        print("1. Enabling takeover...")
        await takeover.enable_takeover(session_id, "Testing takeover mode")

        # Check status
        print(f"\n2. Takeover active: {takeover.is_takeover_active(session_id)}")

        # Execute a command
        print("\n3. Executing manual command...")
        result = await takeover.execute_manual_command(
            "echo 'Hello from takeover mode'", session_id
        )
        print(f"   Success: {result['success']}")

        # Get context
        print("\n4. Getting takeover context...")
        context = takeover.get_takeover_context(session_id)
        print(f"   Commands executed: {len(context['command_history'])}")

        # Resume automation
        print("\n5. Resuming automation...")
        await takeover.resume_automation(session_id)

        print("\n=== Test Complete ===")

    asyncio.run(test_takeover_mode())
