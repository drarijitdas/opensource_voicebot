from loguru import logger
import sys
import asyncio
import time
from typing import Optional
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

import mlflow

from pipecat.frames.frames import StopFrame, CancelFrame, TTSSpeakFrame
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from server.bot.processors.parser import ModalRagStreamingJsonParser
from server.bot.services.modal_services import ModalTunnelManager
from server.observability.metrics import LatencyMetrics, TokenMetrics

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass


class ModalOpenAILLMService(OpenAILLMService):
    def __init__(
        self,
        *args,
        modal_tunnel_manager: ModalTunnelManager = None,
        base_url: str = None,
        **kwargs
    ):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key"

        self.modal_tunnel_manager = modal_tunnel_manager
        self.base_url = base_url
        self._connect_client_task = None

        if self.modal_tunnel_manager:
            logger.info("Using Modal Tunnels")
        if self.base_url:
            logger.info(f"Using URL: {self.base_url}")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
        else:
            self._connect_client_task = asyncio.create_task(self._delayed_create_client(**kwargs))

        super().__init__(*args, base_url=base_url, **kwargs)

        self._client = None

        # Create the JSON parser instance
        self.json_parser = ModalRagStreamingJsonParser(self)

        # Trace manager (injected by bot orchestration)
        self._trace_manager: Optional["TraceContextManager"] = None

    async def _get_url(self):
        if self.modal_tunnel_manager:
            print("Getting URL from modal tunnel manager")
            url = await self.modal_tunnel_manager.get_url()
            print(f"Got URL from tunnel manager: {url}")
            if not url.endswith("/v1"):
                url = f"{url}/v1"
            self.base_url = url
        return self.base_url

    async def _delayed_create_client(self, **kwargs):
        print("Delayed creating client task started...")
        self.base_url = await self._get_url()
        print(f"Got Base URL from _get_url: {self.base_url}")
            
        print(f"Creating client with base URL: {self.base_url}")
        self._client = self.create_client(
            base_url=self.base_url,
            **kwargs
        )
            

    # @classmethod
    # async def from_tunnel_manager(cls, modal_tunnel_manager: ModalTunnelManager, **kwargs):

    #     if not kwargs.get("base_url", None):
    #         base_url = await modal_tunnel_manager.get_url()
            
    #         return cls(
    #             modal_tunnel_manager=modal_tunnel_manager,
    #             base_url=base_url,
    #             **kwargs
    #         )
    #     else:
    #         return cls(
    #             modal_tunnel_manager=modal_tunnel_manager,
    #             **kwargs
    #         )

    async def stop(self, frame: StopFrame):
        await super().stop(frame)
        await self._cleanup()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._cleanup()

    async def _cleanup(self):
        # Cancel the delayed client creation task if it's still running
        if self._connect_client_task and not self._connect_client_task.done():
            self._connect_client_task.cancel()
            try:
                await self._connect_client_task
            except asyncio.CancelledError:
                pass

        if self.modal_tunnel_manager:
            await self.modal_tunnel_manager.close()

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):

        if self._connect_client_task and not self._connect_client_task.done():
            await self.push_frame(TTSSpeakFrame("My apologies, I'm still setting up a few things. I'll respond as soon as I'm ready."))
            return

        # Use MLflow's native span creation if trace manager is available
        if self._trace_manager:
            messages = context.get_messages()
            with mlflow.start_span(name="llm_generation", span_type="LLM") as span:
                # Set inputs
                span.set_inputs({
                    "messages": [{"role": m["role"], "content": str(m["content"])[:200]} for m in messages[-3:]],
                    "num_messages": len(messages),
                    "model": self.model_name,
                })

                # Process LLM
                total_content, token_usage, latency_metrics = await self._process_llm_streaming(context)

                # Set outputs
                span.set_outputs({
                    "response": total_content[:500],  # Truncate for logging
                    "response_length": len(total_content),
                })

                # Log metrics directly to MLflow
                metrics_dict = latency_metrics.to_dict()
                for key, value in metrics_dict.items():
                    mlflow.log_metric(f"llm.{key}", value)

                # Log token metrics if available
                if token_usage:
                    token_metrics = TokenMetrics.from_usage({
                        "prompt_tokens": token_usage.prompt_tokens,
                        "completion_tokens": token_usage.completion_tokens,
                        "total_tokens": token_usage.total_tokens,
                    })
                    for key, value in token_metrics.to_dict().items():
                        mlflow.log_metric(f"llm.{key}", value)

                # Record response for judge evaluation
                self._trace_manager.record_turn_data(assistant_response=total_content)
        else:
            # No tracing, just process
            await self._process_llm_streaming(context)

    async def _process_llm_streaming(self, context: OpenAILLMContext):
        """Process LLM streaming and return metrics."""
        # Initialize metrics tracking
        latency_metrics = LatencyMetrics()

        await self.start_ttfb_metrics()

        # Reset the JSON parser for this context
        self.json_parser.reset()

        messages = context.get_messages()
        edited_messages = []
        for msg in messages[:-1]:
          if msg['role'] in ["assistant", "system"]:
            edited_messages.append(msg)
        edited_messages.append(messages[-1])
        new_context = OpenAILLMContext(messages=edited_messages)

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions_specific_context(
            new_context
        )

        # Track token usage and response
        total_content = ""
        token_usage = None

        async for chunk in chunk_stream:

            if chunk.choices is None or len(chunk.choices) == 0:
                continue

            if not chunk.choices[0].delta:
                continue

            if chunk.choices[0].delta.content:
                await self.stop_ttfb_metrics()

                # Track first token
                if latency_metrics.first_token_time is None:
                    latency_metrics.record_first_token()

                # Track each token
                latency_metrics.record_token()

                # Accumulate content
                total_content += chunk.choices[0].delta.content

                await self.json_parser.process_chunk(chunk.choices[0].delta.content)

            # Capture token usage if available
            if hasattr(chunk, "usage") and chunk.usage:
                token_usage = chunk.usage

        # Record end time
        latency_metrics.record_end()

        return total_content, token_usage, latency_metrics


