"""
Bash Executor Node - Executes shell commands in the Seedbox.

This node takes the ACTION_DETAILS from the planner and executes
it as a bash command in the Docker/SSH sandbox.
"""

import logging

from agent_state import AgentStateDict
from seedbox_executor import SeedboxExecutor

logger = logging.getLogger(__name__)

# Global executor instance (initialized on first use)
_executor: SeedboxExecutor | None = None


def _get_executor() -> SeedboxExecutor:
    """Get or create the seedbox executor."""
    global _executor
    if _executor is None:
        _executor = SeedboxExecutor()
    return _executor


def bash_executor_node(state: AgentStateDict) -> dict:
    """
    LangGraph node that executes bash commands in the Seedbox.

    Extracts the command from action_details, executes it via
    SeedboxExecutor, and updates the state with results.

    Args:
        state: Current agent state with action_details containing the command.

    Returns:
        Dict of state updates including last_tool_output and seedbox_manifest.
    """
    command = state.get("action_details", "").strip()

    if not command:
        logger.warning("Bash executor called with empty command")
        return {
            "last_tool_output": "Error: No command provided",
            "messages": [
                {
                    "role": "system",
                    "content": "Bash execution failed: No command provided",
                }
            ],
        }

    logger.info(f"Executing bash command: {command[:100]}...")

    try:
        executor = _get_executor()
        result = executor.execute_bash(command)

        # Format output for the agent
        output_parts = [
            f"Command: {command}",
            f"Exit Code: {result.get('exit_code', -1)}",
            f"Success: {result.get('success', False)}",
        ]

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        error = result.get("error", "")

        if stdout:
            output_parts.append(f"Output:\n{stdout}")
        if stderr:
            output_parts.append(f"Errors:\n{stderr}")
        if error:
            output_parts.append(f"Error:\n{error}")

        tool_output = "\n".join(output_parts)

        # Refresh seedbox manifest
        try:
            manifest = executor.list_files("/workspace")
        except Exception as e:
            logger.warning(f"Failed to refresh manifest: {e}")
            manifest = state.get("seedbox_manifest", [])

        # Create a tool result message
        tool_message = {"role": "system", "content": f"[BASH RESULT]\n{tool_output}"}

        logger.info(
            f"Bash execution {'succeeded' if result.get('success') else 'failed'}"
        )

        return {
            "last_tool_output": tool_output,
            "seedbox_manifest": manifest,
            "messages": [tool_message],
        }

    except Exception as e:
        logger.error(f"Bash executor error: {e}")

        error_output = f"Command: {command}\nError: {str(e)}"

        return {
            "last_tool_output": error_output,
            "messages": [{"role": "system", "content": f"[BASH ERROR] {str(e)}"}],
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    from agent_state import create_initial_state

    state = create_initial_state("Test")
    state["action_details"] = "echo 'Hello from Seedbox!'"

    result = bash_executor_node(state)
    print(f"\nBash result:")
    print(result.get("last_tool_output"))
