"""
Check Memory Setup.

Verifies that MongoDB dependencies are installed and attempts to connect
to the default MongoDB instance.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())


async def check_memory():
    print("🧠 Checking Memory Manager Setup...\n")

    # 1. Check Dependencies
    try:
        import motor.motor_asyncio
        import pymongo

        print("✅ Dependencies installed (motor, pymongo)")
    except ImportError:
        print("❌ Dependencies MISSING")
        print("   Run: uv pip install motor pymongo")
        return

    # 2. Check Connection
    print("\n🔌 Testing Connection to MongoDB (localhost:27017)...")
    from nodes.memory_manager import MemoryManager

    manager = MemoryManager(ttl_days=1)
    try:
        if await manager.connect():
            print("✅ Connection SUCCESSFUL")
            print("   Memory system is fully operational.")
            await manager.close()
    except Exception as e:
        print(f"❌ Connection FAILED: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Is MongoDB running?")
        print("      Run: docker run -d -p 27017:27017 mongo")
        print("   2. Is the MONGODB_URI environment variable set correctly?")
        print(f"      Current URI: {manager.mongodb_uri}")


if __name__ == "__main__":
    asyncio.run(check_memory())
