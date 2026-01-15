import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.getcwd())
load_dotenv()

from agent_state import AgentStateDict, create_initial_state
from nodes.document_executor import document_executor_node
from nodes.data_analysis_executor import data_analysis_executor_node
from nodes.planning_executor import planning_executor_node
from nodes.planning_manager import PlanningManager


async def verify_modules():
    print("🧪 Verifying New Modules...")

    # 1. Test Document Executor
    print("\n📄 Testing Document Executor...")
    state = create_initial_state("Test Doc")
    state["action_details"] = (
        '{"filename": "test_report.docx", "content": {"title": "Test", "sections": [{"heading": "H1", "content": "Body"}]}}'
    )

    result = document_executor_node(state)
    if "artifacts" in result and result["artifacts"][0]["type"] == "docx":
        print("✅ Document created successfully")
    else:
        print(f"❌ Document creation failed: {result.get('last_tool_output')}")

    # 2. Test Data Analysis (Mock)
    print("\n📊 Testing Data Analysis...")
    # Create a dummy CSV first
    with open("test_data.csv", "w") as f:
        f.write("id,value\n1,10\n2,20")

    state["action_details"] = "test_data.csv | Calculate mean"
    result = data_analysis_executor_node(state)
    if "Analysis Result" in result.get("last_tool_output", ""):
        print("✅ Data analysis executed")
    else:
        print(f"❌ Data analysis failed: {result.get('last_tool_output')}")

    # 3. Test Planning Persistence
    print("\n🧠 Testing Planning Persistence...")
    state["action_details"] = "Build a rocket | high"
    result = planning_executor_node(state)

    # Check if plan file exists
    manager = PlanningManager()
    if manager.plan_path.exists():
        content = manager.plan_path.read_text()
        if "Build a rocket" in content:
            print("✅ Plan persisted to disk")
        else:
            print("❌ Plan file content mismatch")
    else:
        print(f"❌ Plan file not created: {manager.plan_path}")

    # Cleanup
    if os.path.exists("output.docx"):
        os.remove("output.docx")
    if os.path.exists("test_report.docx"):
        os.remove("test_report.docx")
    if os.path.exists("test_data.csv"):
        os.remove("test_data.csv")
    if manager.plan_path.exists():
        os.remove(manager.plan_path)


if __name__ == "__main__":
    asyncio.run(verify_modules())
