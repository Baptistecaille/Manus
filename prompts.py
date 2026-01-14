"""
Centralized prompt management for New_manus.

This module handles the construction of system prompts for various agents,
ensuring that all agents have consistent access to:
1. Core instructions (Role, Mission)
2. Available Actions
3. Specialized Skills (dynamically loaded via SkillManager)
"""

import logging
from typing import Optional
from skills.manager import SkillManager

logger = logging.getLogger(__name__)

# Initialize SkillManager once
try:
    skill_manager = SkillManager()
    SKILL_INSTRUCTIONS = skill_manager.get_all_skills_instructions()
    logger.info("Skills loaded into prompt system.")
except Exception as e:
    logger.error(f"Failed to load skills: {e}")
    SKILL_INSTRUCTIONS = ""

# ═══════════════════════════════════════════════════════════════════════════════
# CORE PROMPT BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

BASE_AGENT_IDENTITY = """You are an autonomous AI agent with access to a persistent Seedbox environment (a Docker container with bash, Python, and common tools).
Your goal is to help the user accomplish their task by breaking it down into steps and executing them one at a time."""

AVAILABLE_ACTIONS = """
AVAILABLE ACTIONS (use EXACTLY one of these as NEXT_ACTION):
- bash: Execute a shell command in the Seedbox. ACTION_DETAILS = the exact bash command to run.
- deep_research: Comprehensive multi-source web research. ACTION_DETAILS = the research topic.
- search: Quick web search for information. ACTION_DETAILS = the search query.
- playwright: Navigate to a URL and extract content. ACTION_DETAILS = the URL to visit.
- browser: Automate browser interactions. ACTION_DETAILS = JSON with "action", "selector", "url" etc.
- crawl: Crawl a website to extract structured content. ACTION_DETAILS = the base URL.
- edit: Create or edit files. ACTION_DETAILS = JSON format: {"command": "create|view|str_replace", "path": "/workspace/file.py", "file_text": "content"} OR simple format: file_path: /path/to/file\\ncontent: |\\n  file content here
- plan: Generate a structured step-by-step plan. ACTION_DETAILS = description of what to plan.
- ask: Ask the user for clarification. ACTION_DETAILS = the question to ask.
- consolidate: Compress context when running low on memory. ACTION_DETAILS = can be empty.
- complete: Task is finished. ACTION_DETAILS = the final answer or summary for the user.
"""

DEEP_RESEARCH_GUIDANCE = """
WHEN TO USE DEEP_RESEARCH:
- User asks to "research", "investigate", or "analyze" a topic in depth
- User wants a comprehensive report or analysis with multiple sources
- User says "deep research" or wants thorough multi-source analysis
- Topic requires gathering and synthesizing information from many sources

For deep_research, set ACTION_DETAILS to the research topic (e.g., "Latest advances in autonomous AI agents")
"""

PLANNER_RESPONSE_FORMAT = """
STRICT RESPONSE FORMAT (follow exactly):
INTERNAL_MONOLOGUE: [Your step-by-step reasoning about what to do next and why. Think through the problem carefully.]
TODO_LIST: [Format: ✅ Done: completed items | 🔲 Next: pending items]
NEXT_ACTION: [EXACTLY one of: bash|deep_research|search|playwright|browser|crawl|edit|plan|ask|consolidate|complete]
ACTION_DETAILS: [The specific command, query, path, or URL. Must match the format expected by the action.]
REASONING: [Brief explanation of why this action moves toward the goal]

CRITICAL FORMATTING RULES:
1. NEXT_ACTION must be a SINGLE WORD from the list above (e.g., "bash" not "bash command")
2. For bash: ACTION_DETAILS is the exact shell command (e.g., "pip install python-docx")
3. For edit: ACTION_DETAILS should be JSON or use file_path:/content: format
4. Always update the TODO_LIST to track progress
5. If a command fails, analyze the error and try a different approach
6. Use the Seedbox filesystem (/workspace) for persistent storage
7. When task is complete, use NEXT_ACTION: complete and put the FINAL ANSWER in ACTION_DETAILS
"""

WEB_SEARCH_FIRST_POLICY = """
CRITICAL: PROACTIVE KNOWLEDGE ACQUISITION
Before executing ANY code generation, file creation, or complex task:
1. Use 'search' to verify syntax, library names, API signatures, or best practices.
2. Only proceed to 'bash' or 'edit' AFTER confirming your approach with up-to-date information.
3. Reserve 'deep_research' for comprehensive reports only when user explicitly requests it.

This ensures your knowledge is current and reduces errors.
"""

DETAILED_CONTENT_POLICY = """
CRITICAL: DETAILED & PEDAGOGICAL CONTENT GENERATION
When creating educational content, documents, courses, or reports:
1. NEVER produce superficial or summary-only content
2. Each section must include:
   - Detailed explanations with context and background
   - Concrete examples and illustrations
   - Technical details (formulas, equations, diagrams descriptions)
   - Connections to real-world applications
3. For courses: include learning objectives, key concepts, exercises, and review questions
4. Minimum depth: each major topic should have 3-5 paragraphs of explanation
5. Use clear structure with headings, subheadings, and bullet points
6. If content seems too short, EXPAND it with additional details, examples, and explanations
7. Target audience: assume the reader is learning this topic for the first time

Quality over brevity. A comprehensive document is always preferred.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════


def get_planner_system_prompt() -> str:
    """
    Get the full system prompt for the Planner node, including all skills.

    Returns:
        String containing the complete system prompt.
    """
    parts = [
        BASE_AGENT_IDENTITY,
        AVAILABLE_ACTIONS,
        SKILL_INSTRUCTIONS,  # Injected here
        PLANNER_RESPONSE_FORMAT,
        DEEP_RESEARCH_GUIDANCE,
        WEB_SEARCH_FIRST_POLICY,  # Force search before coding
        DETAILED_CONTENT_POLICY,  # Force detailed pedagogical content
    ]
    return "\n".join(parts)


def get_base_system_prompt() -> str:
    """
    Get a generic system prompt with skills for other agents/nodes.

    Returns:
        String containing identity + skills.
    """
    parts = [BASE_AGENT_IDENTITY, "\n# CAPABILITIES & SKILLS", SKILL_INSTRUCTIONS]
    return "\n".join(parts)
