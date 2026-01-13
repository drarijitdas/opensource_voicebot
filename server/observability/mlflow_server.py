"""
MLflow Tracking Server with SQLite Backend on Modal.

This module deploys a simple, reliable MLflow tracking server using SQLite
as the backend store and Modal Volume for persistent storage. The MLflow UI
and tracking API are exposed via a web endpoint.

This approach is ideal for development and research workloads with low to
moderate traffic. SQLite can handle thousands of runs efficiently without
the complexity of managing PostgreSQL.
"""

import subprocess

import modal

from server.common.const import SERVICE_REGIONS

APP_NAME = "mlflow-tracking"
MLFLOW_PORT = 5000

# Modal Volume for persistent MLflow data (SQLite DB + artifacts)
mlflow_volume = modal.Volume.from_name("mlflow-data", create_if_missing=True)

# Docker image with MLflow
mlflow_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("mlflow==2.18.0")
)

app = modal.App(APP_NAME)


@app.cls(
    image=mlflow_image,
    volumes={"/mlflow-data": mlflow_volume},
    cpu=2,
    memory=4096,  # 4GB
    timeout=24 * 60 * 60,  # 24 hours
    region=SERVICE_REGIONS,
)
class MLflowServer:
    """MLflow tracking server with SQLite backend."""

    @modal.web_server(MLFLOW_PORT, startup_timeout=120.0)
    def serve(self):
        """
        Serve MLflow UI and tracking API via web server.

        The MLflow server stores metadata in SQLite and artifacts on Modal Volume.
        Both are persistent across container restarts.

        Access the UI at:
        https://<workspace>--mlflow-tracking-mlflowserver-serve.modal.run
        """
        # Start MLflow server
        # SQLite backend: sqlite:////mlflow-data/mlflow.db (4 slashes = absolute path)
        # Artifacts: /mlflow-data/artifacts (on Modal Volume)
        subprocess.Popen(
            [
                "mlflow",
                "server",
                "--backend-store-uri",
                "sqlite:////mlflow-data/mlflow.db",
                "--default-artifact-root",
                "/mlflow-data/artifacts",
                "--host",
                "0.0.0.0",
                "--port",
                str(MLFLOW_PORT),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


# For standalone testing
if __name__ == "__main__":
    print("Deploying MLflow server with SQLite backend...")
    print(f"UI will be available at: https://<workspace>--{APP_NAME}-mlflowserver-serve.modal.run")
    print("\nTo get your workspace name, run: modal profile current")
