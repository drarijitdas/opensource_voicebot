"""
Simplified Trace Context Management Using Native MLflow.

This module provides minimal context management for conversation sessions
and turn tracking, delegating actual tracing to MLflow's native features.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlflow

logger = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """
    Lightweight trace context for a conversation session.

    Attributes:
        session_id: Unique session identifier
        turn_number: Current turn number in the conversation
        metadata: Additional metadata for the session
    """

    session_id: str
    turn_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_new(cls, session_id: Optional[str] = None) -> "TraceContext":
        """
        Create a new trace context for a conversation session.

        Args:
            session_id: Optional session ID (will be generated if not provided)

        Returns:
            TraceContext: New trace context
        """
        session_id = session_id or str(uuid.uuid4())
        return cls(session_id=session_id)

    def next_turn(self):
        """Increment the turn number."""
        self.turn_number += 1
        logger.debug(f"Turn {self.turn_number} started for session {self.session_id}")

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the context."""
        self.metadata[key] = value


class TraceContextManager:
    """
    Manager for trace context in the pipeline.

    This class manages conversation session state and provides utilities
    for logging metrics to the active MLflow run.

    Example:
        >>> trace_mgr = TraceContextManager(mlflow_client)
        >>> ctx = trace_mgr.initialize_conversation()
        >>>
        >>> # Use MLflow's native tracing
        >>> with mlflow.start_span("stt") as span:
        >>>     span.set_inputs({"audio_len": 1024})
        >>>     # Do STT work...
        >>>     span.set_outputs({"transcription": "hello"})
        >>>
        >>> # Start next turn
        >>> trace_mgr.start_turn()
    """

    def __init__(self, mlflow_client: "MLflowAsyncClient"):
        """
        Initialize trace context manager.

        Args:
            mlflow_client: MLflow client for logging
        """
        self.mlflow_client = mlflow_client
        self.current_context: Optional[TraceContext] = None

        # Conversation-level data for judge evaluation
        self.conversation_data = {
            "user_queries": [],
            "assistant_responses": [],
            "retrieved_contexts": [],
        }

    def initialize_conversation(
        self,
        session_id: Optional[str] = None
    ) -> TraceContext:
        """
        Initialize a new conversation session.

        This should be called when a new WebRTC connection is established.

        Args:
            session_id: Optional session ID

        Returns:
            TraceContext: New trace context
        """
        self.current_context = TraceContext.create_new(session_id)
        logger.info(f"Initialized conversation: session_id={self.current_context.session_id}")

        # Tag the session in MLflow
        if self.mlflow_client:
            self.mlflow_client.set_tag("session_id", self.current_context.session_id)

        return self.current_context

    def start_turn(self):
        """
        Start a new conversation turn.

        This should be called when the user starts speaking (e.g., on
        transcription start event).
        """
        if not self.current_context:
            logger.warning("No active conversation context, cannot start turn")
            return

        self.current_context.next_turn()
        logger.debug(f"Started turn {self.current_context.turn_number}")

    def record_turn_data(
        self,
        user_query: Optional[str] = None,
        assistant_response: Optional[str] = None,
        retrieved_context: Optional[str] = None,
    ):
        """
        Record data for the current turn (for judge evaluation).

        Args:
            user_query: User's query (transcription)
            assistant_response: Assistant's response
            retrieved_context: Retrieved RAG context
        """
        if user_query:
            self.conversation_data["user_queries"].append(user_query)
        if assistant_response:
            self.conversation_data["assistant_responses"].append(assistant_response)
        if retrieved_context:
            self.conversation_data["retrieved_contexts"].append(retrieved_context)

    def get_turn_data(self) -> Dict[str, Any]:
        """
        Get data for the current turn (for judge evaluation).

        Returns:
            dict: Turn data with user_query, assistant_response, retrieved_context
        """
        if not self.conversation_data["user_queries"]:
            return {}

        # Get the most recent data
        return {
            "session_id": self.current_context.session_id if self.current_context else None,
            "turn_number": self.current_context.turn_number if self.current_context else 0,
            "user_query": self.conversation_data["user_queries"][-1],
            "assistant_response": (
                self.conversation_data["assistant_responses"][-1]
                if self.conversation_data["assistant_responses"]
                else ""
            ),
            "retrieved_context": (
                self.conversation_data["retrieved_contexts"][-1]
                if self.conversation_data["retrieved_contexts"]
                else ""
            ),
        }


# For standalone testing
if __name__ == "__main__":
    import asyncio
    from server.observability.mlflow_client import MLflowAsyncClient

    async def test_trace_context():
        # Create client
        client = MLflowAsyncClient("http://localhost:5000")
        await client.start()

        # Create trace manager
        trace_mgr = TraceContextManager(client)

        # Initialize conversation
        ctx = trace_mgr.initialize_conversation()
        print(f"Session: {ctx.session_id}")

        # First turn
        trace_mgr.start_turn()
        with mlflow.start_span("stt") as span:
            span.set_inputs({"audio_len": 1024})
            await asyncio.sleep(0.1)
            span.set_outputs({"transcription": "hello"})

        # Second turn
        trace_mgr.start_turn()
        with mlflow.start_span("llm") as span:
            span.set_inputs({"prompt": "Hello"})
            await asyncio.sleep(0.1)
            span.set_outputs({"response": "Hi!"})

        # Stop client
        await client.stop()

    # Run test
    asyncio.run(test_trace_context())
