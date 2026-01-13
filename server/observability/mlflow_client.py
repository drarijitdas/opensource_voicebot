"""
Simplified MLflow Client Using Native MLflow Features.

This module provides a lightweight wrapper around MLflow's native tracing API,
leveraging @mlflow.trace() decorators and mlflow.start_span() for automatic
span creation and logging.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mlflow

logger = logging.getLogger(__name__)


class MLflowAsyncClient:
    """
    Simplified async MLflow client using native MLflow tracing.

    This client configures MLflow and provides utilities for logging,
    but delegates actual tracing to MLflow's native @trace() decorator
    and start_span() context manager.

    Example:
        >>> client = MLflowAsyncClient("http://mlflow-server:5000")
        >>> await client.start()
        >>>
        >>> # Use MLflow's native tracing
        >>> with mlflow.start_span("my_span") as span:
        >>>     span.set_inputs({"input": "data"})
        >>>     # Do work...
        >>>     span.set_outputs({"output": "result"})
        >>>
        >>> await client.stop()
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = "voice-bot",
    ):
        """
        Initialize the MLflow client.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create/get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow experiment: {e}")
            self.experiment_id = None

        logger.info(f"MLflow client initialized (tracking_uri={tracking_uri})")

    async def start(self):
        """
        Start the MLflow client and begin a run.

        This starts an active MLflow run that will capture all traces.
        """
        try:
            # Start a new run if not already active
            if mlflow.active_run() is None:
                mlflow.start_run(experiment_id=self.experiment_id)
                logger.info("Started MLflow run")
            else:
                logger.info("MLflow run already active")
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")

    async def stop(self):
        """
        Stop the MLflow client and end the run.
        """
        try:
            # End the run if active
            if mlflow.active_run() is not None:
                mlflow.end_run()
                logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a metric to the current run.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metric '{key}': {e}")

    def log_param(self, key: str, value: str):
        """
        Log a parameter to the current run.

        Args:
            key: Parameter name
            value: Parameter value
        """
        try:
            mlflow.log_param(key, str(value))
        except Exception as e:
            logger.error(f"Failed to log param '{key}': {e}")

    def set_tag(self, key: str, value: str):
        """
        Set a tag on the current run.

        Args:
            key: Tag name
            value: Tag value
        """
        try:
            mlflow.set_tag(key, str(value))
        except Exception as e:
            logger.error(f"Failed to set tag '{key}': {e}")


class MLflowTracingDisabled:
    """
    Dummy client that disables tracing (for testing or fallback).

    This provides the same interface as MLflowAsyncClient but does nothing.
    """

    async def start(self):
        pass

    async def stop(self):
        pass

    def log_metric(self, *args, **kwargs):
        """No-op log_metric (synchronous to match MLflowAsyncClient)."""
        pass

    def log_param(self, *args, **kwargs):
        """No-op log_param (synchronous to match MLflowAsyncClient)."""
        pass

    def set_tag(self, *args, **kwargs):
        """No-op set_tag (synchronous to match MLflowAsyncClient)."""
        pass


# For standalone testing
if __name__ == "__main__":
    async def test_client():
        # Create client
        client = MLflowAsyncClient("http://localhost:5000")
        await client.start()

        # Create a span
        async with client.create_span(
            "test_span",
            inputs={"input_text": "Hello, world!"}
        ) as span_id:
            # Simulate some work
            await asyncio.sleep(0.1)

            # Log metrics
            await client.log_metric(span_id, "latency_ms", 100.0)
            await client.log_output(span_id, {"output_text": "Hi there!"})

        # Stop client
        await client.stop()

    # Run test
    asyncio.run(test_client())
