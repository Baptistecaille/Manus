"""
Planning Tool adapted from OpenManus.

Generates detailed action plans for complex tasks with structured output.
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PlanningInput(BaseModel):
    """Input schema for planning tool."""

    task: str = Field(description="Complex task that requires planning")
    detail_level: str = Field(
        default="medium",
        description="Planning detail level: low, medium, or high",
    )


class PlanningTool(BaseTool):
    """
    Planning tool for generating structured action plans.

    Creates detailed, step-by-step plans for complex tasks with dependencies,
    estimates, and success criteria.
    """

    name: str = "planning"
    description: str = (
        "Generate a detailed action plan for a complex task. "
        "Returns structured plan with steps, dependencies, estimates, and success criteria. "
        "Use this for multi-step projects before execution."
    )
    args_schema: type[BaseModel] = PlanningInput

    def _run(self, task: str, detail_level: str = "medium") -> str:
        """
        Generate action plan.

        Args:
            task: Task description
            detail_level: Level of detail in plan

        Returns:
            Structured action plan
        """
        try:
            from llm_factory import create_llm

            # Get LLM instance
            llm = create_llm()

            # Create planning prompt
            prompt = self._create_planning_prompt(task, detail_level)

            # Generate plan using LLM
            response = llm.invoke(prompt)
            plan = response.content if hasattr(response, "content") else str(response)

            return f"Action Plan for: {task}\n\n{plan}"

        except Exception as e:
            logger.error(f"Planning error: {e}")
            # Fallback to simple planning
            return self._simple_plan(task, detail_level)

    def _create_planning_prompt(self, task: str, detail_level: str) -> str:
        """Create prompt for plan generation."""
        detail_instructions = {
            "low": "Provide a high-level overview with 3-5 main phases.",
            "medium": "Provide detailed steps with dependencies and estimates.",
            "high": "Provide comprehensive plan with sub-tasks, risks, and contingencies.",
        }

        instruction = detail_instructions.get(
            detail_level, detail_instructions["medium"]
        )

        return f"""You are an expert project planner. Create a structured action plan for the following task.

Task: {task}

Instructions: {instruction}

Format your response as follows:

## Goal
[Clear statement of what will be achieved]

## Prerequisites
- [List any prerequisites or requirements]

## Steps
1. [Step name]
   - Description: [What to do]
   - Dependencies: [What must be done first]
   - Estimated effort: [Time/complexity estimate]
   - Success criteria: [How to know it's done]

[Continue for all steps...]

## Success Criteria
- [How to measure overall success]

## Potential Risks
- [Risk]: [Mitigation strategy]

Please generate the plan now:"""

    def _simple_plan(self, task: str, detail_level: str) -> str:
        """Fallback simple planning when LLM is unavailable."""
        return f"""Action Plan for: {task}

## Goal
Complete the task: {task}

## Steps
1. Analyze Requirements
   - Break down the task into components
   - Identify necessary resources

2. Design Approach
   - Choose appropriate tools and methods
   - Plan execution sequence

3. Execute
   - Implement the solution step by step
   - Test and validate each step

4. Review and Refine
   - Check results against requirements
   - Make necessary adjustments

## Success Criteria
- Task completed as specified
- All requirements met
- Results validated

Note: This is a generic plan. For detailed planning, ensure LLM is properly configured."""

    async def _arun(self, task: str, detail_level: str = "medium") -> str:
        """Async version - calls sync version."""
        return self._run(task, detail_level)
