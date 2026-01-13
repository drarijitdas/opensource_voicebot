"""
Custom Metrics for Voice Bot Tracing.

This module defines custom metrics for tracking performance and quality
of the voice bot pipeline.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LatencyMetrics:
    """Latency metrics for a pipeline stage."""

    start_time: float = field(default_factory=time.perf_counter)
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    token_times: List[float] = field(default_factory=list)

    def record_first_token(self):
        """Record the time of the first token."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def record_token(self):
        """Record the time of a token."""
        self.token_times.append(time.perf_counter())

    def record_end(self):
        """Record the end time."""
        self.end_time = time.perf_counter()

    def get_ttft_ms(self) -> Optional[float]:
        """
        Get Time To First Token in milliseconds.

        Returns:
            float: TTFT in milliseconds, or None if not available
        """
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000

    def get_total_latency_ms(self) -> Optional[float]:
        """
        Get total latency in milliseconds.

        Returns:
            float: Total latency in milliseconds, or None if not available
        """
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def get_inter_token_latency_ms(self) -> Optional[float]:
        """
        Get average inter-token latency in milliseconds.

        Returns:
            float: Average inter-token latency in milliseconds, or None if not available
        """
        if len(self.token_times) < 2:
            return None

        inter_token_latencies = [
            (self.token_times[i] - self.token_times[i - 1]) * 1000
            for i in range(1, len(self.token_times))
        ]

        return sum(inter_token_latencies) / len(inter_token_latencies)

    def get_tokens_per_second(self) -> Optional[float]:
        """
        Get tokens per second.

        Returns:
            float: Tokens per second, or None if not available
        """
        if self.end_time is None or len(self.token_times) == 0:
            return None

        total_time = self.end_time - self.start_time
        if total_time == 0:
            return None

        return len(self.token_times) / total_time

    def to_dict(self) -> dict:
        """
        Convert metrics to dictionary for logging.

        Returns:
            dict: Metrics dictionary
        """
        metrics = {}

        ttft = self.get_ttft_ms()
        if ttft is not None:
            metrics["ttft_ms"] = ttft

        total_latency = self.get_total_latency_ms()
        if total_latency is not None:
            metrics["total_latency_ms"] = total_latency

        inter_token_latency = self.get_inter_token_latency_ms()
        if inter_token_latency is not None:
            metrics["inter_token_latency_ms"] = inter_token_latency

        tokens_per_second = self.get_tokens_per_second()
        if tokens_per_second is not None:
            metrics["tokens_per_second"] = tokens_per_second

        if self.token_times:
            metrics["token_count"] = len(self.token_times)

        return metrics


@dataclass
class TokenMetrics:
    """Token usage metrics for LLM."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_usage(cls, usage: dict) -> "TokenMetrics":
        """
        Create from OpenAI usage dict.

        Args:
            usage: Usage dict from OpenAI API response

        Returns:
            TokenMetrics: Token metrics
        """
        return cls(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def to_dict(self) -> dict:
        """
        Convert to dictionary for logging.

        Returns:
            dict: Metrics dictionary
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class RAGMetrics:
    """RAG retrieval metrics."""

    query: str
    num_docs_retrieved: int = 0
    retrieval_time_ms: float = 0.0
    doc_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Convert to dictionary for logging.

        Returns:
            dict: Metrics dictionary
        """
        metrics = {
            "num_docs_retrieved": self.num_docs_retrieved,
            "retrieval_time_ms": self.retrieval_time_ms,
        }

        if self.doc_scores:
            metrics["avg_doc_score"] = sum(self.doc_scores) / len(self.doc_scores)
            metrics["max_doc_score"] = max(self.doc_scores)
            metrics["min_doc_score"] = min(self.doc_scores)

        return metrics


# For standalone testing
if __name__ == "__main__":
    import asyncio

    async def test_latency_metrics():
        metrics = LatencyMetrics()

        # Simulate TTFT
        await asyncio.sleep(0.05)
        metrics.record_first_token()

        # Simulate tokens
        for _ in range(10):
            await asyncio.sleep(0.01)
            metrics.record_token()

        # End
        metrics.record_end()

        # Print metrics
        print(metrics.to_dict())

    asyncio.run(test_latency_metrics())
