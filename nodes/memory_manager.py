"""
Memory Manager Node - Persistent memory with MongoDB for Manus agent.

Provides async session management, conversation persistence, plan storage,
and context retrieval using MongoDB.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# MongoDB dependencies (optional - gracefully handle if not installed)
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, OperationFailure

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("MongoDB dependencies not installed. Memory persistence disabled.")


class MemoryManager:
    """
    Node for session memory management using MongoDB.

    Provides persistent storage for conversations, execution plans,
    and artifacts with session-based organization.

    Attributes:
        mongodb_uri: MongoDB connection URI.
        db_name: Database name.
        ttl_days: Days to keep session data before auto-cleanup.

    Example:
        >>> manager = MemoryManager()
        >>> await manager.connect()
        >>> await manager.save_conversation("session123", messages)
        >>> await manager.close()
    """

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        db_name: str = "manus_agent",
        ttl_days: int = 7,
    ) -> None:
        """
        Initialize the memory manager.

        Args:
            mongodb_uri: MongoDB connection URI. Defaults to MONGODB_URI env var.
            db_name: Database name. Defaults to 'manus_agent'.
            ttl_days: Days to keep session data.
        """
        self.mongodb_uri = mongodb_uri or os.getenv(
            "MONGODB_URI", "mongodb://localhost:27017"
        )
        self.db_name = db_name
        self.ttl_days = ttl_days

        self._client: Any = None
        self._db: Any = None
        self._connected = False

        logger.debug(f"MemoryManager initialized (db={db_name})")

    async def connect(self) -> bool:
        """
        Connect to MongoDB.

        Returns:
            True if connected successfully.

        Raises:
            RuntimeError: If MongoDB is not available or connection fails.
        """
        if not MONGODB_AVAILABLE:
            logger.error("MongoDB dependencies not installed")
            raise RuntimeError(
                "MongoDB dependencies not installed. "
                "Run: uv pip install motor pymongo"
            )

        if self._connected:
            return True

        logger.info(f"Connecting to MongoDB: {self.mongodb_uri[:30]}...")

        try:
            self._client = AsyncIOMotorClient(self.mongodb_uri)
            # Test connection
            await self._client.admin.command("ping")
            self._db = self._client[self.db_name]

            # Create indexes
            await self._create_indexes()

            self._connected = True
            logger.info("MongoDB connected successfully")
            return True

        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise RuntimeError(f"MongoDB connection failed: {e}") from e

    async def _create_indexes(self) -> None:
        """Create database indexes for efficient queries."""
        try:
            # Conversations collection indexes
            conversations = self._db.conversations
            await conversations.create_index([("session_id", ASCENDING)])
            await conversations.create_index([("timestamp", DESCENDING)])
            await conversations.create_index(
                [("created_at", ASCENDING)],
                expireAfterSeconds=self.ttl_days * 24 * 3600,
            )

            # Plans collection indexes
            plans = self._db.plans
            await plans.create_index([("session_id", ASCENDING)], unique=True)

            # Artifacts collection indexes
            artifacts = self._db.artifacts
            await artifacts.create_index([("session_id", ASCENDING)])
            await artifacts.create_index([("artifact_id", ASCENDING)])

            logger.debug("Database indexes created")

        except OperationFailure as e:
            logger.warning(f"Index creation issue: {e}")

    def _ensure_connected(self) -> None:
        """Ensure MongoDB is connected."""
        if not self._connected:
            raise RuntimeError("Not connected to MongoDB. Call connect() first.")

    async def save_conversation(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> bool:
        """
        Save conversation messages for a session.

        Args:
            session_id: Unique session identifier.
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            True if saved successfully.

        Example:
            >>> await manager.save_conversation("sess123", [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"}
            ... ])
        """
        self._ensure_connected()

        if not session_id:
            raise ValueError("Session ID cannot be empty")

        logger.debug(f"Saving conversation for session {session_id}")

        doc = {
            "session_id": session_id,
            "messages": messages,
            "message_count": len(messages),
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow(),
        }

        try:
            result = await self._db.conversations.update_one(
                {"session_id": session_id},
                {"$set": doc},
                upsert=True,
            )
            logger.info(f"Conversation saved: {session_id} ({len(messages)} messages)")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False

    async def load_context(self, session_id: str) -> dict[str, Any]:
        """
        Load full context for a session.

        Retrieves conversation, plan, and artifacts.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with 'messages', 'plan', 'artifacts'.
        """
        self._ensure_connected()

        if not session_id:
            raise ValueError("Session ID cannot be empty")

        logger.debug(f"Loading context for session {session_id}")

        context: dict[str, Any] = {
            "session_id": session_id,
            "messages": [],
            "plan": None,
            "artifacts": [],
        }

        try:
            # Load conversation
            conv = await self._db.conversations.find_one({"session_id": session_id})
            if conv:
                context["messages"] = conv.get("messages", [])

            # Load plan
            plan_doc = await self._db.plans.find_one({"session_id": session_id})
            if plan_doc:
                context["plan"] = plan_doc.get("plan")

            # Load artifacts
            cursor = self._db.artifacts.find({"session_id": session_id})
            context["artifacts"] = await cursor.to_list(length=100)

            logger.info(
                f"Context loaded: {len(context['messages'])} messages, "
                f"{len(context['artifacts'])} artifacts"
            )
            return context

        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return context

    async def store_plan(
        self,
        session_id: str,
        plan: dict[str, Any],
    ) -> bool:
        """
        Store execution plan for a session.

        Args:
            session_id: Session identifier.
            plan: Plan dict with steps, dependencies, etc.

        Returns:
            True if stored successfully.

        Example:
            >>> await manager.store_plan("sess123", {
            ...     "steps": ["step1", "step2"],
            ...     "estimated_duration": 60
            ... })
        """
        self._ensure_connected()

        if not session_id:
            raise ValueError("Session ID cannot be empty")

        logger.debug(f"Storing plan for session {session_id}")

        doc = {
            "session_id": session_id,
            "plan": plan,
            "updated_at": datetime.utcnow(),
        }

        try:
            await self._db.plans.update_one(
                {"session_id": session_id},
                {"$set": doc},
                upsert=True,
            )
            logger.info(f"Plan stored: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store plan: {e}")
            return False

    async def get_plan(self, session_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve execution plan for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Plan dict or None if not found.
        """
        self._ensure_connected()

        if not session_id:
            raise ValueError("Session ID cannot be empty")

        try:
            doc = await self._db.plans.find_one({"session_id": session_id})
            return doc.get("plan") if doc else None

        except Exception as e:
            logger.error(f"Failed to get plan: {e}")
            return None

    async def store_artifact(
        self,
        session_id: str,
        artifact: dict[str, Any],
    ) -> str:
        """
        Store an artifact (file, data, etc.) for a session.

        Args:
            session_id: Session identifier.
            artifact: Artifact dict with 'name', 'type', 'data', etc.

        Returns:
            Generated artifact ID.

        Example:
            >>> artifact_id = await manager.store_artifact("sess123", {
            ...     "name": "output.txt",
            ...     "type": "file",
            ...     "path": "/workspace/output.txt"
            ... })
        """
        self._ensure_connected()

        if not session_id:
            raise ValueError("Session ID cannot be empty")

        # Generate artifact ID
        import uuid

        artifact_id = str(uuid.uuid4())[:8]

        doc = {
            "session_id": session_id,
            "artifact_id": artifact_id,
            "artifact": artifact,
            "created_at": datetime.utcnow(),
        }

        try:
            await self._db.artifacts.insert_one(doc)
            logger.info(f"Artifact stored: {artifact_id}")
            return artifact_id

        except Exception as e:
            logger.error(f"Failed to store artifact: {e}")
            raise RuntimeError(f"Failed to store artifact: {e}") from e

    async def get_relevant_history(
        self,
        session_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get relevant conversation history based on query.

        Simple keyword-based relevance matching.
        For advanced semantic search, integrate a vector database.

        Args:
            session_id: Session identifier.
            query: Query string to match against.
            limit: Maximum number of messages to return.

        Returns:
            List of relevant messages.
        """
        self._ensure_connected()

        if not session_id:
            raise ValueError("Session ID cannot be empty")

        try:
            conv = await self._db.conversations.find_one({"session_id": session_id})
            if not conv:
                return []

            messages = conv.get("messages", [])

            # Simple keyword matching
            query_words = set(query.lower().split())
            scored_messages: list[tuple[int, dict]] = []

            for msg in messages:
                content = msg.get("content", "").lower()
                score = sum(1 for word in query_words if word in content)
                if score > 0:
                    scored_messages.append((score, msg))

            # Sort by score descending and take top N
            scored_messages.sort(key=lambda x: x[0], reverse=True)
            relevant = [msg for _, msg in scored_messages[:limit]]

            logger.debug(f"Found {len(relevant)} relevant messages")
            return relevant

        except Exception as e:
            logger.error(f"Failed to get relevant history: {e}")
            return []

    async def clear_session(self, session_id: str) -> bool:
        """
        Clear all data for a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if cleared successfully.
        """
        self._ensure_connected()

        if not session_id:
            raise ValueError("Session ID cannot be empty")

        logger.info(f"Clearing session: {session_id}")

        try:
            await self._db.conversations.delete_many({"session_id": session_id})
            await self._db.plans.delete_many({"session_id": session_id})
            await self._db.artifacts.delete_many({"session_id": session_id})
            logger.info(f"Session cleared: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        List recent sessions.

        Args:
            limit: Maximum sessions to return.

        Returns:
            List of session info dicts.
        """
        self._ensure_connected()

        try:
            pipeline = [
                {
                    "$group": {
                        "_id": "$session_id",
                        "message_count": {"$first": "$message_count"},
                        "last_updated": {"$max": "$timestamp"},
                    }
                },
                {"$sort": {"last_updated": -1}},
                {"$limit": limit},
            ]

            cursor = self._db.conversations.aggregate(pipeline)
            sessions = await cursor.to_list(length=limit)

            return [
                {
                    "session_id": s["_id"],
                    "message_count": s.get("message_count", 0),
                    "last_updated": s.get("last_updated"),
                }
                for s in sessions
            ]

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("MongoDB connection closed")

    async def __aenter__(self) -> "MemoryManager":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

# Global manager instance for node function
_memory_manager: Optional[MemoryManager] = None


async def get_memory_manager() -> MemoryManager:
    """Get or create shared memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        try:
            await _memory_manager.connect()
        except Exception as e:
            logger.warning(f"Memory manager connection failed: {e}")
            _memory_manager = None
            raise
    return _memory_manager


async def memory_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node for memory management.

    Saves current conversation and loads relevant context.

    Args:
        state: Current agent state.

    Returns:
        State updates with loaded memory context.
    """
    session_id = state.get("session_id", "")
    if not session_id:
        logger.debug("No session_id, skipping memory node")
        return {}

    try:
        manager = await get_memory_manager()

        # Save current conversation
        messages = state.get("messages", [])
        if messages:
            await manager.save_conversation(session_id, messages)

        # Save plan if present
        execution_plan = state.get("execution_plan")
        if execution_plan:
            await manager.store_plan(session_id, execution_plan)

        # Load relevant history if query present
        query = state.get("enhanced_query") or state.get("original_query", "")
        relevant_history = []
        if query:
            relevant_history = await manager.get_relevant_history(session_id, query)

        return {
            "memory": {
                "loaded": True,
                "relevant_history": relevant_history,
            }
        }

    except Exception as e:
        logger.warning(f"Memory node error: {e}")
        return {
            "memory": {
                "loaded": False,
                "error": str(e),
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    async def test_memory_manager():
        """Quick test of memory manager (requires MongoDB)."""
        print("=== Memory Manager Test ===\n")

        if not MONGODB_AVAILABLE:
            print("⚠ MongoDB dependencies not installed")
            print("Run: uv pip install motor pymongo")
            return

        try:
            async with MemoryManager() as manager:
                session_id = "test_session_123"

                # Test save conversation
                print("1. Saving conversation...")
                messages = [
                    {"role": "user", "content": "Create a Python script"},
                    {"role": "assistant", "content": "I'll help you create a script"},
                ]
                success = await manager.save_conversation(session_id, messages)
                print(f"   Saved: {success}")

                # Test store plan
                print("\n2. Storing plan...")
                plan = {"steps": ["analyze", "code", "test"], "duration": 30}
                success = await manager.store_plan(session_id, plan)
                print(f"   Stored: {success}")

                # Test load context
                print("\n3. Loading context...")
                context = await manager.load_context(session_id)
                print(f"   Messages: {len(context['messages'])}")
                print(f"   Plan: {context['plan']}")

                # Test relevant history
                print("\n4. Getting relevant history...")
                history = await manager.get_relevant_history(
                    session_id, "Python script"
                )
                print(f"   Relevant messages: {len(history)}")

                # Cleanup
                print("\n5. Clearing session...")
                await manager.clear_session(session_id)
                print("   Cleared")

        except Exception as e:
            print(f"✗ Test failed: {e}")
            print("\nMake sure MongoDB is running:")
            print("  docker run -d -p 27017:27017 mongo")

        print("\n=== Test Complete ===")

    asyncio.run(test_memory_manager())
