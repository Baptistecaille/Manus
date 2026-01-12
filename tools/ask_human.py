"""
AskHuman Tool for user interaction.

Integrates with existing HITL system to request user input during execution.
"""

import logging
from typing import Optional, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AskHumanInput(BaseModel):
    """Input schema for asking human."""

    question: str = Field(description="Question to ask the user")
    options: Optional[List[str]] = Field(
        default=None,
        description="Optional list of predefined answer options",
    )
    required: bool = Field(
        default=True,
        description="Whether an answer is required to proceed",
    )


class AskHumanTool(BaseTool):
    """
    Ask Human tool for requesting user input.

    Integrates with the existing HITL (Human-in-the-Loop) system to
    pause execution and get user feedback or decisions.
    """

    name: str = "ask_human"
    description: str = (
        "Ask the user a question and wait for their response. "
        "Use this when you need clarification, approval, or a decision from the user. "
        "Can provide optional answer choices."
    )
    args_schema: type[BaseModel] = AskHumanInput

    def _run(
        self,
        question: str,
        options: Optional[List[str]] = None,
        required: bool = True,
    ) -> str:
        """
        Ask user a question and get response.

        Args:
            question: Question to ask
            options: Optional predefined answers
            required: Whether response is required

        Returns:
            User's response
        """
        try:
            # Import HITL CLI handler
            from hitl.cli_handler import CLIHandler

            handler = CLIHandler()

            # Format question with options if provided
            formatted_question = question
            if options:
                formatted_question += "\n\nOptions:"
                for i, option in enumerate(options, 1):
                    formatted_question += f"\n  {i}. {option}"

            # Display question to user
            print(f"\n{'='*60}")
            print("🤔 AGENT NEEDS YOUR INPUT")
            print(f"{'='*60}")
            print(f"\n{formatted_question}\n")

            # Get user input
            if options:
                print(
                    f"Please enter a number (1-{len(options)}) or type your own answer:"
                )
            else:
                print("Please provide your answer:")

            response = input("> ").strip()

            # Handle numeric response for options
            if options and response.isdigit():
                option_idx = int(response) - 1
                if 0 <= option_idx < len(options):
                    response = options[option_idx]

            # Check for required response
            if required and not response:
                return "Error: No response provided but response was required"

            logger.info(f"User response to '{question}': {response}")
            return f"User responded: {response}"

        except Exception as e:
            logger.error(f"AskHuman error: {e}")
            return f"Error getting user input: {str(e)}"

    async def _arun(
        self,
        question: str,
        options: Optional[List[str]] = None,
        required: bool = True,
    ) -> str:
        """Async version - calls sync version (user interaction is inherently blocking)."""
        return self._run(question, options, required)
