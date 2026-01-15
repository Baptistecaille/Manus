"""
Document Executor Node.

Wraps the DocumentSkill to allow the agent to create Word documents.
"""

import logging
import json
from typing import Any, Dict

from agent_state import AgentStateDict
from skills.document_skill import DocumentSkill

logger = logging.getLogger(__name__)


def document_executor_node(state: AgentStateDict) -> Dict[str, Any]:
    """
    Execute document creation tasks.

    Args:
        state: Current agent state with action_details.

    Returns:
        Updated state with artifact information.
    """
    logger.info("=== Document Executor Node ===")

    action_details = state.get("action_details", "")
    iteration = state.get("iteration_count", 0)

    # Initialize skill
    skill = DocumentSkill()

    try:
        # Parse action details
        # Expecting JSON or structured string. For robustness, try to parse JSON first.
        content = {}
        filename = "output.docx"

        if action_details.strip().startswith("{"):
            try:
                data = json.loads(action_details)
                content = data.get("content", {})
                filename = data.get("filename", filename)
            except json.JSONDecodeError:
                logger.warning("Failed to parse action_details as JSON")

        # If content is still empty, try to infer from text or fail gracefully
        if not content:
            # Fallback: Treat as simple text content if not JSON
            content = {
                "title": "Generated Document",
                "sections": [{"heading": "Content", "content": action_details}],
            }

        # Create document
        logger.info(f"Creating document: {filename}")
        file_path = (
            "output.docx"  # Default placeholder in case async run fails to return
        )

        # Run async method synchronously (since we are in a sync node wrapper or need to manage loop)
        # Run async method
        import asyncio

        loop = asyncio.get_event_loop()

        if loop.is_running():
            # If we're already in a loop (e.g. verify script), we can't use run_until_complete
            # But since this is a synchronous node, we have a dilemma.
            # Ideally, nodes should be async.
            # For now, we use a thread-based workaround if loop is running
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                file_path = executor.submit(
                    asyncio.run, skill.create_word_document(filename, content)
                ).result()
        else:
            file_path = loop.run_until_complete(
                skill.create_word_document(filename, content)
            )

        msg = f"Document created successfully at: {file_path}"
        logger.info(msg)

        return {
            "last_tool_output": msg,
            "iteration_count": iteration + 1,
            "artifacts": state.get("artifacts", [])
            + [{"name": filename, "path": file_path, "type": "docx"}],
            "current_action": "document",
        }

    except Exception as e:
        error_msg = f"Document creation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "last_tool_output": error_msg,
            "iteration_count": iteration + 1,
            "current_action": "document",
        }
