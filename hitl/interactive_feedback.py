"""
Interactive Feedback - Enhanced HITL feedback system for Manus agent.

Provides async methods for requesting user approval, input, and notifications
during agent execution.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

console = Console()


class InteractiveFeedback:
    """
    Enhanced interactive feedback system for HITL interactions.

    Provides methods for requesting approval, user input, showing previews,
    and sending notifications during agent execution.

    Example:
        >>> feedback = InteractiveFeedback()
        >>> approved = await feedback.request_approval(
        ...     "Execute bash command",
        ...     {"command": "rm -rf /tmp/old_files"}
        ... )
    """

    def __init__(self, timeout_seconds: int = 60) -> None:
        """
        Initialize interactive feedback.

        Args:
            timeout_seconds: Default timeout for user responses.
        """
        self.timeout_seconds = timeout_seconds
        self.console = Console()
        logger.debug(f"InteractiveFeedback initialized (timeout={timeout_seconds}s)")

    async def request_approval(
        self,
        action: str,
        details: dict[str, Any],
        risk_level: str = "medium",
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Request user approval for a critical action.

        Args:
            action: Description of the action requiring approval.
            details: Dict with action details to display.
            risk_level: Risk level ('low', 'medium', 'high', 'critical').
            timeout: Custom timeout in seconds.

        Returns:
            True if approved, False if rejected or timeout.

        Example:
            >>> approved = await feedback.request_approval(
            ...     "Delete files",
            ...     {"path": "/tmp/old", "count": 15}
            ... )
        """
        logger.info(f"Requesting approval: {action}")

        # Display approval request
        self._display_approval_request(action, details, risk_level)

        # Get user input with timeout
        try:
            result = await asyncio.wait_for(
                self._get_approval_input(),
                timeout=timeout or self.timeout_seconds,
            )
            logger.info(f"Approval result: {result}")
            return result

        except asyncio.TimeoutError:
            self.console.print("\n[yellow]⏱ Timeout - Action skipped[/yellow]")
            logger.warning(f"Approval timeout for: {action}")
            return False

    def _display_approval_request(
        self,
        action: str,
        details: dict[str, Any],
        risk_level: str,
    ) -> None:
        """Display formatted approval request."""
        # Color based on risk
        color_map = {
            "low": "green",
            "medium": "yellow",
            "high": "orange1",
            "critical": "red",
        }
        color = color_map.get(risk_level, "yellow")

        # Create details table
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value")

        for key, value in details.items():
            table.add_row(str(key), str(value))

        # Display panel
        self.console.print()
        self.console.print(
            Panel(
                table,
                title=f"[bold {color}]🔐 Approval Required: {action}[/bold {color}]",
                subtitle=f"Risk: {risk_level.upper()}",
                border_style=color,
            )
        )

    async def _get_approval_input(self) -> bool:
        """Get user approval input."""
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: Confirm.ask("\n[bold]Approve this action?[/bold]", default=False),
        )

    async def request_input(
        self,
        question: str,
        default: Optional[str] = None,
        choices: Optional[list[str]] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Request text input from the user.

        Args:
            question: Question to ask the user.
            default: Default value if user presses Enter.
            choices: Optional list of valid choices.
            timeout: Custom timeout in seconds.

        Returns:
            User's input string.

        Example:
            >>> name = await feedback.request_input(
            ...     "Enter project name",
            ...     default="my_project"
            ... )
        """
        logger.info(f"Requesting input: {question}")

        self.console.print()
        self.console.print(f"[bold cyan]❓ {question}[/bold cyan]")

        if choices:
            self.console.print(f"[dim]Choices: {', '.join(choices)}[/dim]")

        try:
            result = await asyncio.wait_for(
                self._get_text_input(default, choices),
                timeout=timeout or self.timeout_seconds,
            )
            logger.info(f"User input received: {result[:50]}...")
            return result

        except asyncio.TimeoutError:
            self.console.print(
                f"\n[yellow]⏱ Timeout - Using default: {default}[/yellow]"
            )
            return default or ""

    async def _get_text_input(
        self,
        default: Optional[str],
        choices: Optional[list[str]],
    ) -> str:
        """Get text input from user."""
        loop = asyncio.get_event_loop()

        def get_input():
            if choices:
                return Prompt.ask(
                    "Your choice",
                    choices=choices,
                    default=default,
                )
            return Prompt.ask("Your input", default=default or "")

        return await loop.run_in_executor(None, get_input)

    async def show_preview(
        self,
        content: Any,
        content_type: str = "text",
        title: str = "Preview",
    ) -> None:
        """
        Show a preview of content to the user.

        Args:
            content: Content to display.
            content_type: Type of content ('text', 'json', 'table', 'code').
            title: Title for the preview panel.

        Example:
            >>> await feedback.show_preview(
            ...     {"result": "success", "files": 5},
            ...     content_type="json",
            ...     title="Execution Result"
            ... )
        """
        logger.debug(f"Showing preview: {title}")

        self.console.print()

        if content_type == "json":
            import json

            formatted = json.dumps(content, indent=2, ensure_ascii=False)
            self.console.print(Panel(formatted, title=title, border_style="blue"))

        elif content_type == "table" and isinstance(content, list):
            if content and isinstance(content[0], dict):
                table = Table(title=title)
                for key in content[0].keys():
                    table.add_column(str(key))
                for row in content:
                    table.add_row(*[str(v) for v in row.values()])
                self.console.print(table)
            else:
                self.console.print(Panel(str(content), title=title))

        elif content_type == "code":
            from rich.syntax import Syntax

            syntax = Syntax(str(content), "python", theme="monokai")
            self.console.print(Panel(syntax, title=title, border_style="green"))

        else:
            # Default text display
            self.console.print(Panel(str(content), title=title, border_style="white"))

    async def notify_completion(
        self,
        task: str,
        result: dict[str, Any],
        success: bool = True,
    ) -> None:
        """
        Notify user of task completion.

        Args:
            task: Description of completed task.
            result: Result details dict.
            success: Whether task succeeded.

        Example:
            >>> await feedback.notify_completion(
            ...     "File processing",
            ...     {"processed": 10, "errors": 0},
            ...     success=True
            ... )
        """
        logger.info(f"Task completed: {task} (success={success})")

        self.console.print()

        if success:
            icon = "✅"
            style = "green"
            status = "Completed Successfully"
        else:
            icon = "❌"
            style = "red"
            status = "Failed"

        # Create result table
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value")

        for key, value in result.items():
            table.add_row(str(key), str(value))

        self.console.print(
            Panel(
                table,
                title=f"[bold {style}]{icon} {task}: {status}[/bold {style}]",
                border_style=style,
            )
        )

    async def notify_progress(
        self,
        message: str,
        progress: float = 0.0,
        details: Optional[str] = None,
    ) -> None:
        """
        Show progress notification.

        Args:
            message: Progress message.
            progress: Progress percentage (0.0 - 1.0).
            details: Optional additional details.
        """
        percentage = int(progress * 100)
        bar_width = 20
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        self.console.print(f"[cyan]⏳ {message}[/cyan] [{bar}] {percentage}%")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_feedback_instance: Optional[InteractiveFeedback] = None


def get_feedback() -> InteractiveFeedback:
    """Get or create singleton InteractiveFeedback instance."""
    global _feedback_instance
    if _feedback_instance is None:
        _feedback_instance = InteractiveFeedback()
    return _feedback_instance


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    async def test_interactive_feedback():
        """Test interactive feedback capabilities."""
        print("=== Interactive Feedback Test ===\n")

        feedback = InteractiveFeedback(timeout_seconds=30)

        # Test preview
        print("1. Testing preview...")
        await feedback.show_preview(
            {"status": "ready", "items": 5}, content_type="json", title="Status Preview"
        )

        # Test notification
        print("\n2. Testing completion notification...")
        await feedback.notify_completion(
            "Data Processing", {"files": 10, "duration": "2.5s"}, success=True
        )

        # Test progress
        print("\n3. Testing progress notification...")
        await feedback.notify_progress(
            "Processing files", 0.75, "75 of 100 files complete"
        )

        # Test approval (interactive)
        print("\n4. Testing approval request...")
        approved = await feedback.request_approval(
            "Execute bash command",
            {"command": "echo 'hello world'", "risk": "low"},
            risk_level="low",
            timeout=10,
        )
        print(f"   Approved: {approved}")

        print("\n=== Test Complete ===")

    asyncio.run(test_interactive_feedback())
