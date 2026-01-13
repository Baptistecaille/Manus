"""
Manus Agent - Main entry point.

Run the autonomous agent with a given task/query.
Provides real-time progress display in verbose mode.
"""

import argparse
import logging
import sys
import uuid
from typing import Optional

from dotenv import load_dotenv

from agent_state import create_initial_state, AgentStateDict
from agent_graph import compile_graph
from router import get_next_node_description

# Load environment variables
load_dotenv()


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    elif verbose:
        level = logging.INFO
        format_str = "%(asctime)s [%(levelname)s] %(message)s"
    else:
        level = logging.WARNING
        format_str = "%(message)s"

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_str))

    # File handler - always logs DEBUG level for comprehensive debugging
    file_handler = logging.FileHandler("manus_agent.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    logging.basicConfig(
        level=logging.DEBUG,  # Set root logger to DEBUG to capture all
        handlers=[console_handler, file_handler],
    )


def print_progress(state: AgentStateDict, step: int) -> None:
    """Print formatted progress update."""
    action = state.get("current_action", "planning")
    monologue = state.get("internal_monologue", "")
    todo = state.get("todo_list", "")

    print(f"\n{'='*60}")
    print(f"Step {step} | Action: {action.upper()}")
    print(f"{'='*60}")

    if monologue:
        # Show first 200 chars of monologue
        preview = monologue[:200] + ("..." if len(monologue) > 200 else "")
        print(f"💭 Thinking: {preview}")

    if todo:
        print(f"📋 Todo: {todo[:100]}{'...' if len(todo) > 100 else ''}")

    # Show last tool output preview
    tool_output = state.get("last_tool_output", "")
    if tool_output:
        first_line = tool_output.split("\n")[0][:80]
        print(f"🔧 Last output: {first_line}")


def run_agent(
    user_query: str,
    verbose: bool = True,
    debug: bool = False,
    max_steps: Optional[int] = None,
) -> AgentStateDict:
    """
    Run the Manus agent with a given task.

    Args:
        user_query: The task or query for the agent to execute.
        verbose: If True, print progress at each step.
        debug: If True, enable debug logging.
        max_steps: Maximum number of steps (overrides MAX_ITERATIONS).

    Returns:
        Final agent state after execution.
    """
    setup_logging(verbose=verbose, debug=debug)
    logger = logging.getLogger(__name__)

    print("\n" + "🤖 " + "=" * 56 + " 🤖")
    print("   MANUS AGENT - Autonomous Task Execution")
    print("🤖 " + "=" * 56 + " 🤖")
    print(f"\n📌 Task: {user_query}\n")

    # Create initial state
    state = create_initial_state(user_query)

    # Compile the graph
    logger.info("Compiling agent graph...")
    try:
        graph = compile_graph()
    except Exception as e:
        logger.error(f"Failed to compile graph: {e}")
        raise

    # Run the graph
    logger.info("Starting agent execution...")
    step = 0

    try:
        # Generate unique thread_id for this execution
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 150,
        }

        # Use stream mode for real-time updates
        for event in graph.stream(state, config=config):
            step += 1

            # Extract the current state from the event
            if isinstance(event, dict):
                # event is {node_name: state_updates}
                for node_name, updates in event.items():
                    if isinstance(updates, dict):
                        state.update(updates)

                    if verbose:
                        print_progress(state, step)
                        print(f"   📍 Node: {node_name}")

            # Check for max steps
            if max_steps and step >= max_steps:
                logger.warning(f"Max steps ({max_steps}) reached, stopping")
                break

        print(f"\n{'='*60}")
        print("✅ Agent execution completed!")
        print(f"   Total steps: {step}")
        print(f"   Iterations: {state.get('iteration_count', 0)}")
        print(f"   Context size: ~{state.get('context_size', 0)} tokens")
        print(f"{'='*60}\n")

        # Print final state
        if state.get("todo_list"):
            print(f"📋 Final Todo: {state['todo_list']}")

        if state.get("current_action") == "complete":
            print(
                f"\n🏁 Final Result:\n{state.get('action_details', 'No result provided')}"
            )

        if state.get("seedbox_manifest"):
            files = state["seedbox_manifest"][:10]
            print(f"📁 Workspace files: {', '.join(files)}")
            if len(state["seedbox_manifest"]) > 10:
                print(f"   ... and {len(state['seedbox_manifest']) - 10} more")

        return state

    except KeyboardInterrupt:
        print("\n\n⚠️ Agent interrupted by user")
        return state
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manus Agent - Autonomous AI task execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "List files in workspace"
  python main.py "Create a Python script that prints Hello World" --verbose
  python main.py "Research the latest LangGraph developments" --deep-research
        """,
    )

    parser.add_argument(
        "query", nargs="?", help="The task or query for the agent to execute"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)",
    )

    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Disable verbose output"
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )

    parser.add_argument(
        "--max-steps", type=int, help="Maximum number of execution steps"
    )

    parser.add_argument(
        "--deep-research",
        action="store_true",
        help="Enable deep research mode for comprehensive multi-source analysis",
    )

    args = parser.parse_args()

    # Handle quiet mode
    verbose = not args.quiet

    # If no query provided, enter interactive mode
    if not args.query:
        print("🤖 Manus Agent - Interactive Mode")
        print("Enter your task (or 'quit' to exit):\n")

        while True:
            try:
                query = input(">>> ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if query:
                    run_agent(
                        query,
                        verbose=verbose,
                        debug=args.debug,
                        max_steps=args.max_steps,
                    )
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        # Enhance query for deep research mode
        query = args.query
        if args.deep_research:
            query = f"""[DEEP RESEARCH MODE ENABLED]

User Query: {args.query}

Instructions: Conduct comprehensive research using multiple web sources.
Use the deep_research action with ACTION_DETAILS set to the research topic.
The research system will:
1. Generate diverse search queries
2. Analyze results from DuckDuckGo
3. Identify knowledge gaps and iterate (up to 3 cycles)
4. Produce a detailed Markdown report with citations
5. Save the report in /workspace/

Proceed with NEXT_ACTION: deep_research"""
            print("🔬 Deep Research Mode Enabled")
            print(f"   Topic: {args.query}\n")

        result = run_agent(
            query, verbose=verbose, debug=args.debug, max_steps=args.max_steps
        )

        # Display deep research results
        if args.deep_research and result.get("final_report"):
            report = result["final_report"]
            print("\n" + "=" * 70)
            print("📊 DEEP RESEARCH REPORT GENERATED")
            print("=" * 70)

            # Extract and display executive summary
            lines = report.split("\n")
            in_summary = False
            summary_lines = []
            for line in lines:
                if "Executive Summary" in line:
                    in_summary = True
                    continue
                if in_summary:
                    if line.startswith("##"):
                        break
                    summary_lines.append(line)

            if summary_lines:
                print("\n" + "\n".join(summary_lines[:15]).strip())

            print("\n" + "=" * 70)
            print("📁 Full report saved in Seedbox /workspace/")
            print("=" * 70)


if __name__ == "__main__":
    main()
