"""
MLflow Tracking Server with PostgreSQL Backend on Modal.

This module deploys a PostgreSQL database with Volume persistence and an MLflow
tracking server that uses PostgreSQL as its backend store. The MLflow UI is
exposed via a web endpoint, and the tracking API is accessible via tunnel.
"""

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

import modal

from server.common.const import SERVICE_REGIONS

logger = logging.getLogger(__name__)

APP_NAME = "mlflow-tracking"
MLFLOW_PORT = 5000
POSTGRES_PORT = 5432
POSTGRES_USER = "mlflow"
POSTGRES_PASSWORD = "mlflow"
POSTGRES_DB = "mlflow"
POSTGRES_VOLUME_MOUNT = "/var/lib/postgresql/data"
POSTGRES_DATA_DIR = "/var/lib/postgresql/data/pgdata"  # Subdirectory within Volume

# PostgreSQL binary paths (Debian/Ubuntu)
PG_BIN_DIR = "/usr/lib/postgresql/15/bin"
PG_INITDB = f"{PG_BIN_DIR}/initdb"
PG_POSTGRES = f"{PG_BIN_DIR}/postgres"
PG_CREATEDB = f"{PG_BIN_DIR}/createdb"
PG_CREATEUSER = f"{PG_BIN_DIR}/createuser"
PG_PSQL = f"{PG_BIN_DIR}/psql"
PG_ISREADY = f"{PG_BIN_DIR}/pg_isready"

# Modal Volume for PostgreSQL data persistence
postgres_volume = modal.Volume.from_name(
    "mlflow-postgres-data",
    create_if_missing=True
)

# Docker image with PostgreSQL and MLflow
mlflow_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "postgresql",
        "postgresql-contrib",
        "curl",
    )
    .pip_install(
        "mlflow==2.18.0",
        "psycopg2-binary==2.9.10",
    )
)

app = modal.App(APP_NAME)


@app.cls(
    image=mlflow_image,
    volumes={POSTGRES_VOLUME_MOUNT: postgres_volume},
    cpu=4,
    memory=8192,  # 8GB
    timeout=24 * 60 * 60,  # 24 hours
    region=SERVICE_REGIONS,
)
class MLflowServer:
    """MLflow tracking server with PostgreSQL backend."""

    @modal.enter()
    def start_services(self):
        """Initialize and start PostgreSQL and MLflow services."""
        logger.info("=" * 80)
        logger.info("Starting MLflow server with PostgreSQL backend...")
        logger.info("=" * 80)

        try:
            # Initialize PostgreSQL if needed
            logger.info("Step 1: Initializing PostgreSQL...")
            self._init_postgres()
            logger.info("✓ PostgreSQL initialization complete")

            # Start PostgreSQL server
            logger.info("Step 2: Starting PostgreSQL server...")
            self._start_postgres()
            logger.info("✓ PostgreSQL server started")

            # Wait for PostgreSQL to be ready
            logger.info("Step 3: Waiting for PostgreSQL to be ready...")
            self._wait_for_postgres()
            logger.info("✓ PostgreSQL is ready")

            # Create MLflow database if not exists
            logger.info("Step 4: Creating MLflow database...")
            self._create_mlflow_database()
            logger.info("✓ MLflow database created")

            # Start MLflow server in background
            logger.info("Step 5: Starting MLflow server...")
            self._start_mlflow_server()
            logger.info("✓ MLflow server started")

            logger.info("=" * 80)
            logger.info("MLflow server startup complete!")
            logger.info("=" * 80)
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"ERROR during startup: {type(e).__name__}: {e}")
            logger.error("=" * 80)
            import traceback
            traceback.print_exc()
            raise

    def _init_postgres(self):
        """Initialize PostgreSQL database if not already initialized."""
        pg_data_path = Path(POSTGRES_DATA_DIR)
        pg_version_file = pg_data_path / "PG_VERSION"

        if pg_version_file.exists():
            logger.info(f"PostgreSQL already initialized at {POSTGRES_DATA_DIR}")
            # Ensure postgres user owns the data directory
            subprocess.run(
                ["chown", "-R", "postgres:postgres", POSTGRES_DATA_DIR],
                check=True
            )
            return

        logger.info(f"Initializing PostgreSQL at {POSTGRES_DATA_DIR}...")

        # Ensure directory exists and postgres user owns it
        pg_data_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["chown", "-R", "postgres:postgres", POSTGRES_DATA_DIR],
            check=True
        )

        # Initialize as postgres user
        subprocess.run(
            [
                "su", "postgres", "-c",
                f"{PG_INITDB} -D {POSTGRES_DATA_DIR} --encoding=UTF8 --locale=C"
            ],
            check=True
        )

        logger.info("PostgreSQL initialized successfully")

    def _start_postgres(self):
        """Start PostgreSQL server as background process."""
        logger.info("Starting PostgreSQL server...")

        # Start PostgreSQL
        self.postgres_process = subprocess.Popen(
            [
                "su", "-", "postgres", "-c",
                f"{PG_POSTGRES} -D {POSTGRES_DATA_DIR} -p {POSTGRES_PORT}"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give PostgreSQL time to start
        time.sleep(3)

        if self.postgres_process.poll() is not None:
            # Process died, capture error output
            stdout, stderr = self.postgres_process.communicate()
            logger.error(f"PostgreSQL stdout: {stdout.decode()}")
            logger.error(f"PostgreSQL stderr: {stderr.decode()}")
            raise RuntimeError("PostgreSQL failed to start")

        logger.info(f"PostgreSQL started on port {POSTGRES_PORT}")

    def _wait_for_postgres(self, max_retries=30, delay=1):
        """Wait for PostgreSQL to be ready to accept connections."""
        logger.info("Waiting for PostgreSQL to be ready...")

        for i in range(max_retries):
            result = subprocess.run(
                [
                    "su", "-", "postgres", "-c",
                    f"{PG_ISREADY} -p {POSTGRES_PORT}"
                ],
                capture_output=True
            )
            if result.returncode == 0:
                logger.info("PostgreSQL is ready")
                return

            time.sleep(delay)

        raise RuntimeError(f"PostgreSQL not ready after {max_retries} retries")

    def _create_mlflow_database(self):
        """Create MLflow database and user if they don't exist."""
        logger.info(f"Creating MLflow database '{POSTGRES_DB}'...")

        # Create user
        subprocess.run(
            [
                "su", "-", "postgres", "-c",
                f"{PG_PSQL} -p {POSTGRES_PORT} -c \"CREATE USER {POSTGRES_USER} WITH PASSWORD '{POSTGRES_PASSWORD}';\""
            ],
            capture_output=True  # Ignore error if user exists
        )

        # Create database
        subprocess.run(
            [
                "su", "-", "postgres", "-c",
                f"{PG_PSQL} -p {POSTGRES_PORT} -c \"CREATE DATABASE {POSTGRES_DB} OWNER {POSTGRES_USER};\""
            ],
            capture_output=True  # Ignore error if database exists
        )

        # Grant privileges
        subprocess.run(
            [
                "su", "-", "postgres", "-c",
                f"{PG_PSQL} -p {POSTGRES_PORT} -c \"GRANT ALL PRIVILEGES ON DATABASE {POSTGRES_DB} TO {POSTGRES_USER};\""
            ],
            check=True
        )

        logger.info("MLflow database created successfully")

    def _start_mlflow_server(self):
        """Start MLflow tracking server with PostgreSQL backend."""
        logger.info("Starting MLflow tracking server...")

        # Construct PostgreSQL connection string
        db_uri = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"
        logger.info(f"Database URI: postgresql://{POSTGRES_USER}:***@localhost:{POSTGRES_PORT}/{POSTGRES_DB}")

        # Start MLflow server in background
        self.mlflow_process = subprocess.Popen(
            [
                "mlflow",
                "server",
                "--backend-store-uri", db_uri,
                "--default-artifact-root", "/tmp/mlflow-artifacts",
                "--host", "0.0.0.0",
                "--port", str(MLFLOW_PORT),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give MLflow time to start
        time.sleep(5)

        if self.mlflow_process.poll() is not None:
            # Process died, capture error output
            stdout, stderr = self.mlflow_process.communicate()
            logger.error(f"MLflow stdout: {stdout.decode()}")
            logger.error(f"MLflow stderr: {stderr.decode()}")
            raise RuntimeError("MLflow server failed to start")

        # Health check
        self._health_check()

        logger.info(f"MLflow server started on port {MLFLOW_PORT}")

    def _health_check(self, max_retries=30, delay=1):
        """Check if MLflow server is responding."""
        logger.info("Performing MLflow health check...")

        for i in range(max_retries):
            result = subprocess.run(
                ["curl", "-f", f"http://localhost:{MLFLOW_PORT}/health"],
                capture_output=True
            )
            if result.returncode == 0:
                logger.info("MLflow health check passed")
                return

            time.sleep(delay)

        raise RuntimeError(f"MLflow health check failed after {max_retries} retries")

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        """
        Create a tunnel to the MLflow server and share the URL via Modal Dict.

        This follows the same pattern as other services (STT, TTS, LLM) for
        service discovery.

        Args:
            d: Modal ephemeral Dict to share the tunnel URL
        """
        from modal.experimental import forward

        logger.info("Creating tunnel to MLflow server...")

        # Forward MLflow port
        with forward(MLFLOW_PORT) as tunnel:
            # Share tunnel URL
            url = tunnel.url
            logger.info(f"MLflow server tunnel URL: {url}")
            await d.put.aio("url", url)

            # Keep tunnel alive
            try:
                while True:
                    await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("Tunnel client cancelled")
                raise

    @modal.web_server(MLFLOW_PORT, startup_timeout=120.0)
    def mlflow_ui(self):
        """
        Expose MLflow UI via web server.

        Access the UI at: https://<workspace>--mlflow-tracking-mlflow-ui.modal.run

        This web server forwards requests to the MLflow server running on port 5000.
        """
        # MLflow server is already running from start_services()
        # This decorator just exposes it to the internet
        import time
        while True:
            time.sleep(60)  # Keep the server alive

    @modal.exit()
    def shutdown(self):
        """Gracefully shutdown PostgreSQL and MLflow."""
        logger.info("Shutting down services...")

        # Terminate MLflow
        if hasattr(self, 'mlflow_process') and self.mlflow_process:
            self.mlflow_process.terminate()
            self.mlflow_process.wait(timeout=10)

        # Stop PostgreSQL gracefully
        if hasattr(self, 'postgres_process') and self.postgres_process:
            subprocess.run(
                ["su", "-", "postgres", "-c", "pg_ctl stop -D /var/lib/postgresql/data"],
                timeout=30
            )

        logger.info("Services shut down successfully")


class ModalTunnelManager:
    """
    Manager for spawning MLflow server and obtaining its tunnel URL.

    This follows the same pattern as ModalTunnelManager in modal_services.py.
    """

    def __init__(self, app_name: str = APP_NAME, cls_name: str = "MLflowServer"):
        self.app_name = app_name
        self.cls_name = cls_name
        self.connection_urls_dict = modal.Dict.ephemeral()

    async def get_url(self) -> str:
        """
        Spawn the MLflow server (if not running) and get its tunnel URL.

        Returns:
            str: MLflow server tunnel URL (e.g., "https://...modal.run")
        """
        # Spawn the MLflow server
        cls = modal.Cls.lookup(self.app_name, self.cls_name)
        cls.spawn_sandbox.remote(
            "run_tunnel_client",
            self.connection_urls_dict
        )

        # Wait for URL to be available
        max_retries = 60
        for i in range(max_retries):
            try:
                url = await self.connection_urls_dict.get.aio("url")
                if url:
                    return url
            except Exception:
                pass
            await asyncio.sleep(1)

        raise RuntimeError(f"Failed to get MLflow server URL after {max_retries} seconds")


# For standalone testing
if __name__ == "__main__":
    # Test MLflow server deployment
    with modal.enable_output():
        # Deploy the server
        print("Deploying MLflow server...")
        modal.run(app)

        # Create tunnel manager and get URL
        tunnel_mgr = ModalTunnelManager()
        print("Getting MLflow server URL...")
        url = asyncio.run(tunnel_mgr.get_url())
        print(f"MLflow server URL: {url}")
        print("MLflow UI should be accessible shortly")
