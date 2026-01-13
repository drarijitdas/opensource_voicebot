"""
LLM-as-a-Judge Evaluator for Voice Bot Quality Assessment.

This module implements background evaluation workers that use GPT-4 to assess
the quality of bot responses. Evaluations run asynchronously to avoid adding
latency to user interactions.
"""

import asyncio
import logging
from typing import Any, Dict

import modal
from openai import AsyncOpenAI

from server.common.const import SERVICE_REGIONS

logger = logging.getLogger(__name__)

APP_NAME = "mlflow-judge-evaluator"

# Queue for evaluation tasks
eval_queue = modal.Queue.from_name(
    "mlflow-evaluation-queue",
    create_if_missing=True
)

# Docker image with MLflow and OpenAI
judge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "mlflow==2.18.0",
        "openai>=1.0.0",
    )
)

app = modal.App(APP_NAME)


# Judge prompts for each metric
JUDGE_PROMPTS = {
    "answer_relevancy": """You are an expert evaluator assessing whether an AI assistant's response addresses the user's question.

User Question: {user_query}
Assistant Response: {assistant_response}

Evaluate whether the response directly addresses the question on a scale of 1-5:
1 - Completely irrelevant, does not address the question at all
2 - Mostly irrelevant, touches on the topic but misses the main point
3 - Somewhat relevant, addresses parts of the question but misses key aspects
4 - Mostly relevant, addresses the main question with minor gaps
5 - Highly relevant, directly and completely addresses the question

Provide your score as a JSON object: {{"score": <1-5>, "justification": "<brief explanation>"}}""",

    "retrieval_relevancy": """You are an expert evaluator assessing whether retrieved context documents are relevant to a user's query.

User Query: {user_query}
Retrieved Context: {retrieved_context}

Evaluate whether the retrieved documents are relevant to answering the query on a scale of 1-5:
1 - Completely irrelevant, documents have no relation to the query
2 - Mostly irrelevant, documents are tangentially related but not useful
3 - Somewhat relevant, documents contain some useful information
4 - Mostly relevant, documents contain information needed to answer the query
5 - Highly relevant, documents directly contain the information needed

Provide your score as a JSON object: {{"score": <1-5>, "justification": "<brief explanation>"}}""",

    "faithfulness": """You are an expert evaluator assessing whether an AI assistant's response is faithful to the provided context (no hallucinations).

Retrieved Context: {retrieved_context}
Assistant Response: {assistant_response}

Evaluate whether the response is faithful to the context on a scale of 1-5:
1 - Completely unfaithful, contains false information contradicting the context
2 - Mostly unfaithful, contains significant hallucinations or unsupported claims
3 - Somewhat faithful, mostly grounded but with minor unsupported details
4 - Mostly faithful, well-grounded with only trivial deviations
5 - Completely faithful, all claims are supported by the context

Provide your score as a JSON object: {{"score": <1-5>, "justification": "<brief explanation>"}}""",

    "tone": """You are an expert evaluator assessing the conversational tone of an AI assistant's response.

User Question: {user_query}
Assistant Response: {assistant_response}

Evaluate whether the tone is appropriate for a helpful voice assistant on a scale of 1-5:
1 - Very inappropriate (rude, overly formal, or off-putting)
2 - Mostly inappropriate (awkward or mismatched to context)
3 - Acceptable (neutral, but could be more engaging)
4 - Good (friendly and appropriate)
5 - Excellent (warm, helpful, and perfectly suited to voice interaction)

Provide your score as a JSON object: {{"score": <1-5>, "justification": "<brief explanation>"}}""",

    "coherence": """You are an expert evaluator assessing the coherence and structure of an AI assistant's response.

Assistant Response: {assistant_response}

Evaluate whether the response is well-structured and coherent on a scale of 1-5:
1 - Incoherent, disorganized, difficult to follow
2 - Mostly incoherent, jumps between topics or lacks structure
3 - Somewhat coherent, understandable but could be better organized
4 - Mostly coherent, well-structured with minor issues
5 - Highly coherent, clear, logical, and easy to follow

Provide your score as a JSON object: {{"score": <1-5>, "justification": "<brief explanation>"}}""",
}


async def evaluate_with_gpt4(
    client: AsyncOpenAI,
    metric_name: str,
    user_query: str,
    assistant_response: str,
    retrieved_context: str,
) -> Dict[str, Any]:
    """
    Evaluate a single metric using GPT-4.

    Args:
        client: OpenAI async client
        metric_name: Name of the metric (e.g., "answer_relevancy")
        user_query: User's query
        assistant_response: Assistant's response
        retrieved_context: Retrieved RAG context

    Returns:
        dict: Evaluation result with score and justification
    """
    prompt = JUDGE_PROMPTS[metric_name].format(
        user_query=user_query,
        assistant_response=assistant_response,
        retrieved_context=retrieved_context,
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        import json
        result = json.loads(content)

        return {
            "metric": metric_name,
            "score": result.get("score", 0),
            "justification": result.get("justification", ""),
        }

    except Exception as e:
        logger.error(f"Failed to evaluate {metric_name}: {e}")
        return {
            "metric": metric_name,
            "score": 0,
            "justification": f"Evaluation failed: {str(e)}",
        }


@app.function(
    image=judge_image,
    timeout=10 * 60,  # 10 minutes
    secrets=[modal.Secret.from_name("openai-secret")],
    region=SERVICE_REGIONS,
)
async def run_judge_evaluations():
    """
    Background worker for LLM-as-a-judge evaluations.

    This function consumes evaluation tasks from a Modal Queue and runs
    GPT-4 evaluations for each task. Results are logged back to MLflow.
    """
    import os
    import mlflow

    logger.info("Starting judge evaluation worker...")

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return

    client = AsyncOpenAI(api_key=api_key)

    # Get MLflow tracking URI from environment (will be set by bot)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    logger.info(f"Judge evaluator initialized (mlflow_uri={mlflow_uri})")

    # Process evaluation queue
    while True:
        try:
            # Get evaluation task from queue (non-blocking with timeout)
            try:
                eval_task = await asyncio.wait_for(
                    eval_queue.get.aio(),
                    timeout=60.0  # Check for new tasks every 60s
                )
            except asyncio.TimeoutError:
                # No tasks, continue waiting
                continue

            logger.info(f"Processing evaluation for trace_id={eval_task.get('trace_id')}")

            # Extract task data
            trace_id = eval_task.get("trace_id")
            session_id = eval_task.get("session_id")
            user_query = eval_task.get("user_query", "")
            assistant_response = eval_task.get("assistant_response", "")
            retrieved_context = eval_task.get("retrieved_context", "")

            # Skip if missing data
            if not user_query or not assistant_response:
                logger.warning(f"Skipping evaluation: missing data for trace_id={trace_id}")
                continue

            # Run all evaluations in parallel
            evaluation_tasks = [
                evaluate_with_gpt4(
                    client,
                    "answer_relevancy",
                    user_query,
                    assistant_response,
                    retrieved_context,
                ),
                evaluate_with_gpt4(
                    client,
                    "retrieval_relevancy",
                    user_query,
                    assistant_response,
                    retrieved_context,
                ),
                evaluate_with_gpt4(
                    client,
                    "faithfulness",
                    user_query,
                    assistant_response,
                    retrieved_context,
                ),
                evaluate_with_gpt4(
                    client,
                    "tone",
                    user_query,
                    assistant_response,
                    retrieved_context,
                ),
                evaluate_with_gpt4(
                    client,
                    "coherence",
                    user_query,
                    assistant_response,
                    retrieved_context,
                ),
            ]

            results = await asyncio.gather(*evaluation_tasks)

            # Log results to MLflow
            for result in results:
                metric_name = result["metric"]
                score = result["score"]
                justification = result["justification"]

                try:
                    # Log metric with trace reference
                    mlflow.log_metric(
                        f"judge.{metric_name}",
                        score,
                        step=0,  # Could use turn number here
                    )
                    # Log justification as text artifact
                    mlflow.log_text(
                        justification,
                        f"judge_{metric_name}_{trace_id}.txt",
                    )
                    logger.info(f"Logged {metric_name}={score} for trace_id={trace_id}")
                except Exception as e:
                    logger.error(f"Failed to log metric to MLflow: {e}")

            logger.info(f"Completed evaluation for trace_id={trace_id}")

        except asyncio.CancelledError:
            logger.info("Judge evaluator cancelled")
            break
        except Exception as e:
            logger.error(f"Judge evaluation error: {e}", exc_info=True)
            await asyncio.sleep(5)  # Back off on error


@app.function(image=judge_image)
def start_judge_worker():
    """
    Start the judge evaluation worker in the background.

    This should be called once when the system starts up.
    """
    logger.info("Starting judge worker...")
    modal.Function.lookup(APP_NAME, "run_judge_evaluations").spawn()
    logger.info("Judge worker spawned")


# For standalone testing
if __name__ == "__main__":
    # Test evaluation
    async def test_evaluation():
        import os
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Set OPENAI_API_KEY environment variable")
            return

        client = AsyncOpenAI(api_key=api_key)

        result = await evaluate_with_gpt4(
            client,
            "answer_relevancy",
            user_query="What is Modal?",
            assistant_response="Modal is a serverless cloud platform for deploying AI applications.",
            retrieved_context="Modal is a cloud platform...",
        )

        print(f"Result: {result}")

    # Run test
    asyncio.run(test_evaluation())
