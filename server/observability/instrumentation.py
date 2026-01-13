"""
Instrumentation Helpers for Service Tracing.

This module provides decorators and utilities for automatically instrumenting
services with tracing spans.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def traced_service(
    service_name: str,
    span_type: str = "UNKNOWN",
    extract_inputs: Optional[Callable[[Any, Any], Dict[str, Any]]] = None,
    extract_outputs: Optional[Callable[[Any], Dict[str, Any]]] = None,
):
    """
    Decorator for automatic span creation around service methods.

    This decorator wraps async methods to create MLflow spans automatically.
    It extracts inputs and outputs using the provided functions.

    Args:
        service_name: Name of the service (e.g., "stt", "llm_generation")
        span_type: Span type (e.g., "LLM", "RETRIEVAL", "TOOL")
        extract_inputs: Function to extract inputs from method args/kwargs
        extract_outputs: Function to extract outputs from method result

    Example:
        >>> @traced_service("llm_generation", span_type="LLM")
        >>> async def generate(self, prompt: str):
        >>>     return await self._generate(prompt)

    Note:
        The method's class must have a `_trace_manager` attribute.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get trace manager from service
            trace_mgr = getattr(self, "_trace_manager", None)
            if not trace_mgr:
                # No tracing, just call the function
                return await func(self, *args, **kwargs)

            # Extract inputs
            inputs = {}
            if extract_inputs:
                try:
                    inputs = extract_inputs(args, kwargs)
                except Exception as e:
                    logger.error(f"Failed to extract inputs: {e}")

            # Create span
            start_time = time.perf_counter()
            async with trace_mgr.span(
                name=service_name,
                span_type=span_type,
                inputs=inputs,
            ) as span_id:
                # Call the function
                result = await func(self, *args, **kwargs)

                # Extract outputs
                outputs = {}
                if extract_outputs:
                    try:
                        outputs = extract_outputs(result)
                    except Exception as e:
                        logger.error(f"Failed to extract outputs: {e}")

                # Add latency
                outputs["latency_s"] = time.perf_counter() - start_time

                # Log outputs
                await trace_mgr.log_output(span_id, outputs)

                return result

        return wrapper

    return decorator


def extract_text_input(args, kwargs) -> Dict[str, Any]:
    """
    Extract text input from args/kwargs.

    Assumes the first positional argument is the text input.
    """
    if args and len(args) > 0:
        text = args[0]
        return {
            "text": str(text)[:1000],  # Truncate to 1000 chars
            "text_length": len(str(text)),
        }
    return {}


def extract_text_output(result) -> Dict[str, Any]:
    """
    Extract text output from result.

    Assumes the result is a string.
    """
    if isinstance(result, str):
        return {
            "text": result[:1000],  # Truncate to 1000 chars
            "text_length": len(result),
        }
    return {}


def extract_audio_input(args, kwargs) -> Dict[str, Any]:
    """
    Extract audio input from args/kwargs.

    Assumes the first positional argument is audio bytes.
    """
    if args and len(args) > 0:
        audio = args[0]
        if isinstance(audio, bytes):
            return {"audio_length_bytes": len(audio)}
    return {}


# For standalone testing
if __name__ == "__main__":
    import asyncio
    from server.observability.mlflow_client import MLflowAsyncClient
    from server.observability.trace_context import TraceContextManager

    class DummyService:
        def __init__(self, trace_mgr):
            self._trace_manager = trace_mgr

        @traced_service(
            "test_service",
            extract_inputs=extract_text_input,
            extract_outputs=extract_text_output,
        )
        async def process(self, text: str) -> str:
            await asyncio.sleep(0.1)
            return text.upper()

    async def test_tracing():
        client = MLflowAsyncClient("http://localhost:5000")
        await client.start()

        trace_mgr = TraceContextManager(client)
        trace_mgr.initialize_conversation()

        service = DummyService(trace_mgr)
        result = await service.process("hello")
        print(f"Result: {result}")

        await client.stop()

    asyncio.run(test_tracing())
