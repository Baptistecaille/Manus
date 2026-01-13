import logging
import re
from typing import Dict, Any

from agent_state import AgentStateDict, estimate_tokens, calculate_context_size
from llm_factory import create_llm
from prompts import get_base_system_prompt

logger = logging.getLogger(__name__)

# Specialized System Prompt for SWE Agent
SWE_SYSTEM_PROMPT = f"""{get_base_system_prompt()}

ROLE: Expert Software Engineer Agent (SWE).
FOCUS: Code implementation, debugging, refactoring, and file system operations.

AVAILABLE ACTIONS:
- bash: Execute shell commands (git, ls, grep, python scripts, etc.)
- edit: View, create, and modify files.
- complete: Task finished.

STRICT RESPONSE FORMAT:
INTERNAL_MONOLOGUE: [Technical reasoning]
TODO_LIST: [State of technical tasks]
NEXT_ACTION: [bash|edit|complete]
ACTION_DETAILS: [Command or Edit parameters]
REASONING: [Why this technical step is needed]

GUIDELINES:
1. Prefer 'bash' for exploration (ls, find, grep) and running scripts.
2. Use 'edit' ONLY for viewing or modifying TEXT files (not binary like .docx).
3. For document creation (.docx, .pdf, etc.):
   - Write a Python script inline using bash: echo '...' > script.py && python script.py
   - OR use cat << 'EOF' > script.py ... EOF && python script.py
4. NEVER use 'edit' to create .docx files directly - they are binary!
5. Verify your changes (e.g., ls -la to confirm file creation).

EXAMPLE for .docx creation:
ACTION: bash
ACTION_DETAILS: cat << 'EOF' > /workspace/create_doc.py
from docx import Document
doc = Document()
doc.add_heading('Title', 0)
doc.add_paragraph('Content')
doc.save('/workspace/output.docx')
EOF
python /workspace/create_doc.py
"""

USER_TEMPLATE = """CURRENT CONTEXT:
{context}

STATE:
- Files: {seedbox_manifest}
- Last Output: {last_tool_output}

Iteration: {iteration_count}

What is the next TECHNICAL step?"""


def _build_context(state: AgentStateDict) -> str:
    # Simplified context builder for SWE
    messages = state.get("messages", [])
    return "\n".join(
        [
            f"[{m.get('role', '?').upper()}]: {m.get('content', '')}"
            for m in messages[-10:]
        ]
    )  # Focus on recent history


def _parse_response(response_text: str) -> dict:
    # Reusing similar parsing logic but stricter valid actions
    result = {
        "internal_monologue": "",
        "todo_list": "",
        "next_action": "complete",
        "action_details": "",
    }

    # Simple regex parsing (same as main planner for consistency)
    patterns = {
        "internal_monologue": r"INTERNAL_MONOLOGUE:\s*(.+?)(?=TODO_LIST:|$)",
        "todo_list": r"TODO_LIST:\s*(.+?)(?=NEXT_ACTION:|$)",
        "next_action": r"NEXT_ACTION:\s*(\w+)",
        "action_details": r"ACTION_DETAILS:\s*(.+?)(?=REASONING:|$)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            result[field] = match.group(1).strip()

    # Normalize and Validate
    result["next_action"] = result["next_action"].lower().strip()
    if result["next_action"] not in ["bash", "edit", "complete"]:
        # Fallback for hallucinated actions
        logger.warning(f"SWE Planner invalid action: {result['next_action']}")
        result["next_action"] = "complete"  # Safe fail

    return result


def swe_planner_node(state: AgentStateDict) -> dict:
    """Specialized Planner for SWE tasks."""
    logger.info(f"SWE Planner - Iteration {state.get('iteration_count', 0)}")

    context = _build_context(state)
    user_message = USER_TEMPLATE.format(
        context=context,
        seedbox_manifest=state.get("seedbox_manifest", [])[:10],
        last_tool_output=str(state.get("last_tool_output", ""))[:2000],
        iteration_count=state.get("iteration_count", 0),
    )

    try:
        llm = create_llm(temperature=0.1)  # Low temp for code precision
        messages = [
            {"role": "system", "content": SWE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = llm.invoke(messages)
        response_text = response.content

        parsed = _parse_response(response_text)

        return {
            "messages": [{"role": "assistant", "content": response_text}],
            "internal_monologue": parsed["internal_monologue"],
            "todo_list": parsed["todo_list"],
            "current_action": parsed["next_action"],
            "action_details": parsed["action_details"],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    except Exception as e:
        logger.error(f"SWE Planner crash: {e}")
        return {"current_action": "complete", "action_details": f"Error: {str(e)}"}
