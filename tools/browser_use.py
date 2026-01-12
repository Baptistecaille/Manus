"""
Browser Automation Tool using browser-use library.

Provides intelligent browser automation via LangChain tool interface.
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BrowserUseInput(BaseModel):
    """Input schema for browser automation."""

    task: str = Field(
        description="Browser automation task to execute (e.g., 'navigate to google.com and search for LangGraph')"
    )
    headless: bool = Field(
        default=True, description="Run browser in headless mode (no GUI)"
    )
    max_steps: int = Field(
        default=20, description="Maximum number of browser actions", ge=1, le=100
    )


class BrowserUseTool(BaseTool):
    """
    Browser automation tool using browser-use library.

    Executes browser automation tasks with natural language commands.
    This tool can navigate websites, click elements, fill forms, extract data, etc.
    """

    name: str = "browser_use"
    description: str = (
        "Automate browser interactions using natural language commands. "
        "Can navigate to URLs, click buttons, fill forms, extract data, take screenshots. "
        "Examples: 'Navigate to github.com', 'Search for X on google', 'Fill login form'."
    )
    args_schema: type[BaseModel] = BrowserUseInput

    def _run(self, task: str, headless: bool = True, max_steps: int = 20) -> str:
        """
        Execute browser automation task.

        Args:
            task: Natural language description of browser task
            headless: Run in headless mode
            max_steps: Maximum automation steps

        Returns:
            Result of browser automation
        """
        try:
            from browser_use import Agent
            from langchain_openai import ChatOpenAI
            import asyncio
            import os

            async def run_browser_task():
                # Initialize LLM for browser-use
                llm = ChatOpenAI(
                    model=os.getenv("LLM_MODEL", "gpt-4"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                )

                # Create browser automation agent
                agent = Agent(
                    task=task,
                    llm=llm,
                    max_actions_per_step=max_steps,
                    headless=headless,
                )

                # Execute the task
                result = await agent.run()

                # Extract result
                if result:
                    return f"Browser task completed successfully.\n\nTask: {task}\n\nResult: {result}"
                else:
                    return (
                        f"Browser task completed but returned no result. Task: {task}"
                    )

            # Run async function
            return asyncio.run(run_browser_task())

        except ImportError:
            logger.error("browser-use not installed")
            return (
                "Error: browser-use library not installed. "
                "Run: uv pip install browser-use"
            )
        except Exception as e:
            logger.error(f"Browser automation error: {e}")
            return f"Browser automation error: {str(e)}"

    async def _arun(self, task: str, headless: bool = True, max_steps: int = 20) -> str:
        """Async version - proper async implementation."""
        try:
            from browser_use import Agent
            from langchain_openai import ChatOpenAI
            import os

            # Initialize LLM for browser-use
            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-4"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            # Create browser automation agent
            agent = Agent(
                task=task,
                llm=llm,
                max_actions_per_step=max_steps,
                headless=headless,
            )

            # Execute the task
            result = await agent.run()

            # Extract result
            if result:
                return f"Browser task completed successfully.\n\nTask: {task}\n\nResult: {result}"
            else:
                return f"Browser task completed but returned no result. Task: {task}"

        except ImportError:
            logger.error("browser-use not installed")
            return "Error: browser-use library not installed"
        except Exception as e:
            logger.error(f"Browser automation error: {e}")
            return f"Browser automation error: {str(e)}"
