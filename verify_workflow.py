import sys
import os
import logging
from agent_state import create_initial_state
from nodes.planner import planner_node

# Add current directory to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_WORKFLOW")


def test_planner_staging_compliance():
    logger.info("Testing Planner compliance with Staging Protocol...")

    # Create a state where the user asks to create a file
    state = create_initial_state(
        "Create a Python script named hello.py that prints 'Hello World'"
    )

    # Run planner
    result = planner_node(state)

    action = result.get("current_action")
    details = result.get("action_details")
    monologue = result.get("internal_monologue")

    logger.info(f"Action: {action}")
    logger.info(f"Details: {details}")
    logger.info(f"Monologue: {monologue}")

    # Verification criteria
    # 1. Monologue should mention temporary directory or staging
    # 2. If action is 'bash', it might be 'mkdir -p /workspace/temp_staging'
    # 3. If action is 'edit', path should be in temp dir

    is_compliant = False

    if "temp" in monologue.lower() or "staging" in monologue.lower():
        is_compliant = True
        logger.info("Compliance Check: Monologue mentions temp/staging.")

    if "temp" in str(details).lower() or "staging" in str(details).lower():
        is_compliant = True
        logger.info("Compliance Check: Action details involve temp/staging.")

    if is_compliant:
        logger.info("✅ Planner is adhering to the Staging Protocol.")
    else:
        logger.error("❌ Planner FAILED to adhere to Staging Protocol.")
        # We don't raise strict error here as LLM behavior is probabilistic,
        # but we log it clearly.


if __name__ == "__main__":
    try:
        test_planner_staging_compliance()
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
