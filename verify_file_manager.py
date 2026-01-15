"""
Verify File Manager Integration.

Checks:
1. Agent graph compilation.
2. Router selection logic for file manager.
"""

import logging
import sys

# Ensure current dir is in path
sys.path.append(".")

from agent_graph import compile_graph
from router import select_optimal_executor, router


def test_graph_compilation():
    print("Testing graph compilation...")
    try:
        graph = compile_graph(enable_checkpointing=False)
        nodes = list(graph.nodes.keys())
        if "file_manager_executor" in nodes:
            print("✓ file_manager_executor found in graph")
        else:
            print("✗ file_manager_executor NOT found in graph")
            return False

        print("✓ Graph compiled successfully")
        return True
    except Exception as e:
        print(f"✗ Graph compilation failed: {e}")
        return False


def test_router_logic():
    print("\nTesting router logic...")

    # Test keyword matching
    tasks = [
        ("Please organize files in /downloads", "file_manager_executor"),
        ("Compress these text files", "file_manager_executor"),
        ("Convert data.csv to json", "file_manager_executor"),
        ("Write a report about photosynthesis", "document_executor"),
    ]

    executors = ["file_manager_executor", "document_executor", "bash_executor"]

    all_passed = True
    for task, expected in tasks:
        selected = select_optimal_executor(task, executors)
        if selected == expected:
            print(f"✓ '{task[:30]}...' -> {selected}")
        else:
            print(f"✗ '{task[:30]}...' -> {selected} (expected {expected})")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    compilation_ok = test_graph_compilation()
    router_ok = test_router_logic()

    if compilation_ok and router_ok:
        print("\n=== VERIFICATION SUCCESSFUL ===")
        sys.exit(0)
    else:
        print("\n=== VERIFICATION FAILED ===")
        sys.exit(1)
