# Docker Neo4j Enterprise Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-manage a Docker container running Neo4j Enterprise + APOC when the MCP server starts, with resilient two-phase startup and clean shutdown.

**Architecture:** A new `_docker.py` module provides `Neo4jDockerManager` (async context manager) that probes for Neo4j, starts a Docker container if needed, waits for readiness, and stops it on exit. It integrates into the existing FastMCP lifespan in `server.py`. A retry loop wraps `MemoryClient` creation to handle the bolt-open-but-not-ready race condition.

**Tech Stack:** Python `docker` SDK (>=7.0.0), asyncio, Neo4j Enterprise 5.x Docker image, APOC plugin

---

### Task 1: Add docker dependency to pyproject.toml

**Files:**
- Modify: `pyproject.toml:6-10`

**Step 1: Add the docker package to dependencies**

```toml
dependencies = [
    "neo4j-agent-memory[mcp,openai]",
    "fastmcp>=2.0.0,<3",
    "baml-py>=0.70.0",
    "docker>=7.0.0",
]
```

**Step 2: Install updated dependencies**

Run: `uv sync`
Expected: Resolves and installs `docker` package successfully

**Step 3: Verify import works**

Run: `uv run python -c "import docker; print(docker.__version__)"`
Expected: Prints version >=7.0.0

---

### Task 2: Create _docker.py — Neo4jDockerManager core class

**Files:**
- Create: `src/neo4j_agent_memory/mcp/_docker.py`
- Test: `tests/test_docker_manager.py`

**Step 1: Write failing test for TCP probe utility**

Create `tests/test_docker_manager.py`:

```python
"""Tests for Neo4jDockerManager."""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


@pytest.fixture
def docker_config():
    """Standard Docker manager config for tests."""
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "testpassword",
        "docker_auto": True,
        "docker_image": "neo4j:5-enterprise",
        "startup_timeout": 10,
    }


class TestProbeNeo4j:
    """Tests for the _probe_bolt helper."""

    async def test_probe_returns_true_when_reachable(self):
        from neo4j_agent_memory.mcp._docker import _probe_bolt

        # Use a mock that simulates successful connection
        with patch("neo4j_agent_memory.mcp._docker.asyncio.open_connection") as mock_conn:
            mock_reader = MagicMock()
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            result = await _probe_bolt("localhost", 7687)
            assert result is True

    async def test_probe_returns_false_when_unreachable(self):
        from neo4j_agent_memory.mcp._docker import _probe_bolt

        with patch(
            "neo4j_agent_memory.mcp._docker.asyncio.open_connection",
            side_effect=OSError("Connection refused"),
        ):
            result = await _probe_bolt("localhost", 7687)
            assert result is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docker_manager.py -v`
Expected: FAIL — `_docker` module does not exist

**Step 3: Write the _docker.py module with TCP probe and class skeleton**

Create `src/neo4j_agent_memory/mcp/_docker.py`:

```python
"""Docker container management for Neo4j Enterprise.

Provides Neo4jDockerManager, an async context manager that ensures
a Neo4j Enterprise container with APOC is running before the MCP
server connects. Stops the container on exit.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_NAME = "neo4j-agent-memory"
DEFAULT_IMAGE = "neo4j:5-enterprise"
DEFAULT_STARTUP_TIMEOUT = 60
VOLUME_NAME = "neo4j-agent-memory-data"

NEO4J_CONTAINER_ENV = {
    "NEO4J_ACCEPT_LICENSE_AGREEMENT": "yes",
    "NEO4J_PLUGINS": '["apoc"]',
    "NEO4J_dbms_security_procedures_unrestricted": "apoc.*",
    "NEO4J_dbms_security_procedures_allowlist": "apoc.*",
    "NEO4J_server_memory_heap_initial__size": "2g",
    "NEO4J_server_memory_heap_max__size": "4g",
    "NEO4J_server_memory_pagecache_size": "2g",
    "NEO4J_dbms_memory_transaction_total_max": "1g",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _probe_bolt(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to *host*:*port* succeeds."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


def _parse_bolt_uri(uri: str) -> tuple[str, int]:
    """Extract host and port from a Neo4j bolt URI."""
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    return host, port


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


@dataclass
class Neo4jDockerManager:
    """Async context manager that ensures a Neo4j Docker container is running.

    Usage::

        async with Neo4jDockerManager(uri=uri, password=pw) as mgr:
            # Neo4j is reachable at uri
            ...
        # Container stopped (if we started it)
    """

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    image: str = DEFAULT_IMAGE
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT
    docker_auto: bool = True

    # Private state
    _container: object | None = field(default=None, init=False, repr=False)
    _we_started: bool = field(default=False, init=False, repr=False)

    async def __aenter__(self) -> Neo4jDockerManager:
        if not self.docker_auto:
            logger.info("Docker auto-management disabled, skipping")
            return self

        host, port = _parse_bolt_uri(self.uri)

        # Phase 1a: check if Neo4j is already reachable
        if await _probe_bolt(host, port):
            logger.info("Neo4j already reachable at %s:%d, skipping Docker", host, port)
            return self

        # Phase 1b: start or reuse Docker container
        await self._ensure_container(host, port)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._container and self._we_started:
            await self._stop_container()

    # ----- container lifecycle -----

    async def _ensure_container(self, host: str, port: int) -> None:
        """Start or reuse a Neo4j Docker container."""
        try:
            import docker
            import docker.errors
        except ImportError:
            logger.warning(
                "docker package not installed — cannot auto-start Neo4j. "
                "Install with: pip install docker"
            )
            return

        try:
            client = docker.from_env()
            client.ping()
        except docker.errors.DockerException as exc:
            logger.warning(
                "Docker not available (%s) — assuming Neo4j is managed externally", exc
            )
            return

        # Check for existing container
        try:
            container = client.containers.get(CONTAINER_NAME)
            if container.status == "running":
                logger.info(
                    "Container '%s' already running — reusing (will not stop on exit)",
                    CONTAINER_NAME,
                )
                self._container = container
                self._we_started = False
                await self._wait_for_bolt(host, port)
                return
            else:
                logger.info(
                    "Container '%s' exists but status=%s — starting it",
                    CONTAINER_NAME,
                    container.status,
                )
                container.start()
                self._container = container
                self._we_started = True
                await self._wait_for_bolt(host, port)
                return
        except docker.errors.NotFound:
            pass  # Will create below

        # Create new container
        logger.info(
            "Creating Neo4j container '%s' from image '%s'",
            CONTAINER_NAME,
            self.image,
        )
        env = {
            **NEO4J_CONTAINER_ENV,
            "NEO4J_AUTH": f"{self.user}/{self.password}" if self.password else "none",
        }
        try:
            container = client.containers.run(
                self.image,
                name=CONTAINER_NAME,
                detach=True,
                ports={"7687/tcp": port, "7474/tcp": 7474},
                environment=env,
                volumes={VOLUME_NAME: {"bind": "/data", "mode": "rw"}},
                mem_limit="8g",
            )
            self._container = container
            self._we_started = True
            logger.info("Container '%s' started", CONTAINER_NAME)
        except docker.errors.APIError as exc:
            _msg = f"Failed to start Neo4j container: {exc}"
            if "port is already allocated" in str(exc).lower():
                _msg += f" — port {port} is in use. Is another Neo4j running?"
            logger.error(_msg)
            raise RuntimeError(_msg) from exc
        except docker.errors.ImageNotFound:
            _msg = (
                f"Docker image '{self.image}' not found. "
                "Run: docker pull neo4j:5-enterprise"
            )
            logger.error(_msg)
            raise RuntimeError(_msg)

        await self._wait_for_bolt(host, port)

    async def _wait_for_bolt(self, host: str, port: int) -> None:
        """Poll bolt port until ready or timeout."""
        logger.info(
            "Waiting for Neo4j bolt at %s:%d (timeout=%ds)...",
            host,
            port,
            self.startup_timeout,
        )
        elapsed = 0.0
        interval = 0.5
        while elapsed < self.startup_timeout:
            if await _probe_bolt(host, port):
                logger.info("Neo4j bolt ready at %s:%d (%.1fs)", host, port, elapsed)
                return
            await asyncio.sleep(interval)
            elapsed += interval
            # Exponential backoff capped at 4s
            interval = min(interval * 2, 4.0)

        # Timeout — clean up
        if self._container and self._we_started:
            await self._stop_container()
        raise RuntimeError(
            f"Neo4j failed to become ready at {host}:{port} "
            f"within {self.startup_timeout}s"
        )

    async def _stop_container(self) -> None:
        """Stop and remove the container. Best-effort, never raises."""
        if not self._container:
            return
        name = CONTAINER_NAME
        try:
            self._container.stop(timeout=10)
            self._container.remove()
            logger.info("Container '%s' stopped and removed", name)
        except Exception as exc:
            logger.warning("Failed to stop container '%s': %s", name, exc)
        finally:
            self._container = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_docker_manager.py -v`
Expected: Both probe tests PASS

**Step 5: Commit**

```bash
git add src/neo4j_agent_memory/mcp/_docker.py tests/test_docker_manager.py pyproject.toml
git commit -m "feat: add Neo4jDockerManager for automatic container lifecycle"
```

---

### Task 3: Add tests for container lifecycle logic

**Files:**
- Modify: `tests/test_docker_manager.py`

**Step 1: Write tests for manager enter/exit behavior**

Append to `tests/test_docker_manager.py`:

```python
class TestNeo4jDockerManager:
    """Tests for the manager context lifecycle."""

    async def test_skips_docker_when_neo4j_reachable(self, docker_config):
        """If Neo4j is already up, don't touch Docker at all."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        mgr = Neo4jDockerManager(**docker_config)

        with patch("neo4j_agent_memory.mcp._docker._probe_bolt", return_value=True):
            async with mgr:
                assert mgr._container is None
                assert mgr._we_started is False

    async def test_skips_when_docker_auto_false(self, docker_config):
        """docker_auto=False disables all container management."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        docker_config["docker_auto"] = False
        mgr = Neo4jDockerManager(**docker_config)

        async with mgr:
            assert mgr._container is None

    async def test_skips_when_docker_not_available(self, docker_config):
        """Gracefully skip if docker package not installed."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        mgr = Neo4jDockerManager(**docker_config)

        with (
            patch("neo4j_agent_memory.mcp._docker._probe_bolt", return_value=False),
            patch.dict("sys.modules", {"docker": None, "docker.errors": None}),
        ):
            async with mgr:
                assert mgr._container is None

    async def test_creates_container_when_neo4j_unreachable(self, docker_config):
        """Creates and starts container when Neo4j not reachable."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.stop = MagicMock()
        mock_container.remove = MagicMock()

        mock_docker_client = MagicMock()
        mock_docker_client.ping.return_value = True
        mock_docker_client.containers.get.side_effect = Exception("NotFound")
        mock_docker_client.containers.run.return_value = mock_container

        # First probe fails (Neo4j not up), then succeeds after container start
        probe_results = [False, True]

        mgr = Neo4jDockerManager(**docker_config)

        with (
            patch(
                "neo4j_agent_memory.mcp._docker._probe_bolt",
                side_effect=probe_results,
            ),
            patch("docker.from_env", return_value=mock_docker_client),
            patch("docker.errors.DockerException", Exception),
            patch("docker.errors.NotFound", Exception),
            patch("docker.errors.APIError", Exception),
            patch("docker.errors.ImageNotFound", Exception),
        ):
            async with mgr:
                assert mgr._we_started is True

            # After exit, container should be stopped
            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()

    async def test_reuses_running_container(self, docker_config):
        """Reuses existing running container without stopping on exit."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        mock_container = MagicMock()
        mock_container.status = "running"

        mock_docker_client = MagicMock()
        mock_docker_client.ping.return_value = True
        mock_docker_client.containers.get.return_value = mock_container

        mgr = Neo4jDockerManager(**docker_config)

        with (
            patch("neo4j_agent_memory.mcp._docker._probe_bolt", side_effect=[False, True]),
            patch("docker.from_env", return_value=mock_docker_client),
            patch("docker.errors.DockerException", Exception),
            patch("docker.errors.NotFound", Exception),
        ):
            async with mgr:
                assert mgr._we_started is False

            # Should NOT stop — we didn't start it
            mock_container.stop.assert_not_called()
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_docker_manager.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_docker_manager.py
git commit -m "test: add container lifecycle tests for Neo4jDockerManager"
```

---

### Task 4: Add MemoryClient retry loop helper

**Files:**
- Modify: `src/neo4j_agent_memory/mcp/_docker.py`
- Test: `tests/test_docker_manager.py`

**Step 1: Write failing test for retry helper**

Append to `tests/test_docker_manager.py`:

```python
class TestConnectWithRetry:
    """Tests for connect_with_retry helper."""

    async def test_succeeds_on_first_try(self):
        from neo4j_agent_memory.mcp._docker import connect_with_retry

        mock_client = MagicMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        factory = MagicMock(return_value=mock_cm)

        client, cm = await connect_with_retry(factory, max_attempts=3, delay=0.01)
        assert client is mock_client
        assert factory.call_count == 1

    async def test_retries_on_failure_then_succeeds(self):
        from neo4j_agent_memory.mcp._docker import connect_with_retry

        mock_client = MagicMock()
        mock_cm_good = AsyncMock()
        mock_cm_good.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm_good.__aexit__ = AsyncMock(return_value=False)

        mock_cm_bad = AsyncMock()
        mock_cm_bad.__aenter__ = AsyncMock(side_effect=Exception("not ready"))
        mock_cm_bad.__aexit__ = AsyncMock(return_value=False)

        calls = [mock_cm_bad, mock_cm_bad, mock_cm_good]
        factory = MagicMock(side_effect=calls)

        client, cm = await connect_with_retry(factory, max_attempts=5, delay=0.01)
        assert client is mock_client
        assert factory.call_count == 3

    async def test_raises_after_all_retries_exhausted(self):
        from neo4j_agent_memory.mcp._docker import connect_with_retry

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=Exception("not ready"))
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        factory = MagicMock(return_value=mock_cm)

        with pytest.raises(RuntimeError, match="Could not connect to Neo4j"):
            await connect_with_retry(factory, max_attempts=3, delay=0.01)
        assert factory.call_count == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_docker_manager.py::TestConnectWithRetry -v`
Expected: FAIL — `connect_with_retry` does not exist

**Step 3: Add connect_with_retry to _docker.py**

Append before the `Neo4jDockerManager` class in `_docker.py`:

```python
async def connect_with_retry(
    client_factory,
    *,
    max_attempts: int = 5,
    delay: float = 2.0,
):
    """Create a MemoryClient with retries for Neo4j warmup.

    Args:
        client_factory: Callable returning an async context manager
            (e.g., ``lambda: MemoryClient(settings)``).
        max_attempts: Max connection attempts.
        delay: Seconds between attempts.

    Returns:
        Tuple of (client, context_manager) — caller must ``await cm.__aexit__()``
        when done.

    Raises:
        RuntimeError: After all attempts exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        cm = client_factory()
        try:
            client = await cm.__aenter__()
            return client, cm
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Neo4j not ready yet, retrying in %.0fs (attempt %d/%d): %s",
                delay,
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                await asyncio.sleep(delay)

    raise RuntimeError(
        f"Could not connect to Neo4j after {max_attempts} attempts: {last_exc}"
    ) from last_exc
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_docker_manager.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/neo4j_agent_memory/mcp/_docker.py tests/test_docker_manager.py
git commit -m "feat: add connect_with_retry for resilient Neo4j connection"
```

---

### Task 5: Integrate Neo4jDockerManager into server.py lifespan

**Files:**
- Modify: `src/neo4j_agent_memory/mcp/server.py:48-83` (lifespan function)
- Modify: `src/neo4j_agent_memory/mcp/server.py:167-210` (run_server function)

**Step 1: Modify the lifespan in create_mcp_server**

Replace the lifespan function body (lines 51-83) with:

```python
            @asynccontextmanager
            async def lifespan(server: FastMCP):  # noqa: E303
                """Manage Docker container and MemoryClient lifecycle."""
                import os

                from neo4j_agent_memory import MemoryClient as _MemoryClient
                from neo4j_agent_memory.mcp._docker import (
                    Neo4jDockerManager,
                    connect_with_retry,
                )

                # Patch factory to support BAML extraction [RFI-R1]
                import neo4j_agent_memory.extraction.factory as _factory_mod
                from neo4j_agent_memory.extraction.factory_ext import (
                    create_extractor as _ext_create_extractor,
                )

                _factory_mod.create_extractor = _ext_create_extractor
                logger.info("Extraction factory patched with BAML support")

                # Phase 1: Ensure Neo4j container is running
                docker_cfg = getattr(settings, "_docker_config", {})
                neo4j_cfg = settings.neo4j
                docker_mgr = Neo4jDockerManager(
                    uri=str(neo4j_cfg.uri),
                    user=neo4j_cfg.username,
                    password=neo4j_cfg.password.get_secret_value()
                    if neo4j_cfg.password
                    else "",
                    docker_auto=docker_cfg.get("docker_auto", True),
                    image=docker_cfg.get("docker_image", "neo4j:5-enterprise"),
                    startup_timeout=docker_cfg.get("startup_timeout", 60),
                )

                async with docker_mgr:
                    # Phase 2: Connect MemoryClient with retries
                    client, client_cm = await connect_with_retry(
                        lambda: _MemoryClient(settings),
                        max_attempts=5,
                        delay=2.0,
                    )
                    try:
                        # Verify BAML patch took effect [RFI-R1]
                        baml_enabled = os.environ.get(
                            "NAM_EXTRACTION__BAML_ENABLED", ""
                        ).lower() in ("true", "1", "yes")
                        if baml_enabled:
                            _ext = getattr(client, "_extractor", None)
                            _ext_name = getattr(_ext, "name", str(type(_ext)))
                            if _ext and "Baml" in str(_ext_name):
                                logger.info(
                                    "BAML extraction active: %s", _ext_name
                                )
                            else:
                                logger.error(
                                    "BAML enabled but extractor is %s "
                                    "— patch may have failed",
                                    _ext_name,
                                )

                        yield {"client": client}
                    finally:
                        await client_cm.__aexit__(None, None, None)
```

**Step 2: Modify run_server to accept and pass Docker config**

Replace `run_server` function (lines 167-210) with:

```python
    async def run_server(
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        transport: str = "stdio",
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        docker_auto: bool = True,
        docker_image: str = "neo4j:5-enterprise",
        docker_startup_timeout: int = 60,
    ) -> None:
        """Run the MCP server with Neo4j connection.

        Args:
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            neo4j_database: Neo4j database name.
            transport: Transport type (stdio, sse, or http).
            host: Host for network transports.
            port: Port for network transports.
            docker_auto: Enable automatic Docker container management.
            docker_image: Neo4j Docker image to use.
            docker_startup_timeout: Max seconds to wait for Neo4j startup.
        """
        from pydantic import SecretStr

        from neo4j_agent_memory import MemorySettings
        from neo4j_agent_memory.config.settings import Neo4jConfig

        settings = MemorySettings(
            neo4j=Neo4jConfig(
                uri=neo4j_uri,
                username=neo4j_user,
                password=SecretStr(neo4j_password),
                database=neo4j_database,
            )
        )

        # Attach Docker config as private attr for lifespan to read
        settings._docker_config = {
            "docker_auto": docker_auto,
            "docker_image": docker_image,
            "startup_timeout": docker_startup_timeout,
        }

        server = create_mcp_server(settings, server_name="neo4j-agent-memory")

        if transport == "sse":
            await server.run_async(transport="sse", host=host, port=port)
        elif transport == "http":
            await server.run_async(transport="http", host=host, port=port)
        else:
            await server.run_async(transport="stdio")
```

**Step 3: Verify existing tests still pass**

Run: `uv run pytest tests/ -v`
Expected: All existing tests PASS

**Step 4: Commit**

```bash
git add src/neo4j_agent_memory/mcp/server.py
git commit -m "feat: integrate Neo4jDockerManager into MCP server lifespan"
```

---

### Task 6: Add Docker CLI flags to main()

**Files:**
- Modify: `src/neo4j_agent_memory/mcp/server.py:228-284` (main function)

**Step 1: Add new CLI arguments after the existing --port argument**

Insert after line 270 (`help="Port for network transports"`):

```python
    parser.add_argument(
        "--neo4j-docker-auto",
        default=os.environ.get("NEO4J_DOCKER_AUTO", "true").lower()
        in ("true", "1", "yes"),
        action=argparse.BooleanOptionalAction,
        help="Enable automatic Docker container management (default: true)",
    )
    parser.add_argument(
        "--neo4j-docker-image",
        default=os.environ.get("NEO4J_DOCKER_IMAGE", "neo4j:5-enterprise"),
        help="Neo4j Docker image to use",
    )
    parser.add_argument(
        "--neo4j-docker-startup-timeout",
        type=int,
        default=int(os.environ.get("NEO4J_DOCKER_STARTUP_TIMEOUT", "60")),
        help="Max seconds to wait for Neo4j startup (default: 60)",
    )
```

**Step 2: Pass new args to run_server call**

Update the `asyncio.run(run_server(...))` block to include:

```python
    asyncio.run(
        run_server(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            transport=args.transport,
            host=args.host,
            port=args.port,
            docker_auto=args.neo4j_docker_auto,
            docker_image=args.neo4j_docker_image,
            docker_startup_timeout=args.neo4j_docker_startup_timeout,
        )
    )
```

**Step 3: Test CLI help output**

Run: `uv run python -m neo4j_agent_memory.mcp.server --help`
Expected: Shows `--neo4j-docker-auto`, `--neo4j-docker-image`, `--neo4j-docker-startup-timeout` in help

**Step 4: Commit**

```bash
git add src/neo4j_agent_memory/mcp/server.py
git commit -m "feat: add Docker management CLI flags to MCP server"
```

---

### Task 7: Add test for URI parser edge cases

**Files:**
- Modify: `tests/test_docker_manager.py`

**Step 1: Write tests for _parse_bolt_uri**

Append to `tests/test_docker_manager.py`:

```python
class TestParseBoltUri:
    """Tests for bolt URI parsing."""

    def test_standard_bolt_uri(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("bolt://localhost:7687")
        assert host == "localhost"
        assert port == 7687

    def test_custom_port(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("bolt://myhost:9999")
        assert host == "myhost"
        assert port == 9999

    def test_default_port_when_missing(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("bolt://myhost")
        assert host == "myhost"
        assert port == 7687

    def test_neo4j_scheme(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("neo4j://db.example.com:7687")
        assert host == "db.example.com"
        assert port == 7687

    def test_bolt_plus_s_scheme(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("bolt+s://secure.example.com:7687")
        assert host == "secure.example.com"
        assert port == 7687
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_docker_manager.py::TestParseBoltUri -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_docker_manager.py
git commit -m "test: add URI parser edge case tests"
```

---

### Task 8: Final integration verification

**Step 1: Run the complete test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify CLI works end-to-end with Docker disabled**

Run: `uv run python -c "from neo4j_agent_memory.mcp._docker import Neo4jDockerManager; print('import OK')"`
Expected: `import OK`

**Step 3: Verify --help shows all new flags**

Run: `uv run python run_server.py --help`
Expected: All Docker flags visible

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Docker Neo4j Enterprise integration with APOC

Automatic container lifecycle management for Neo4j Enterprise
with APOC plugin. Two-phase resilient startup with retry loop.
8GB RAM allocation with tuned heap/cache settings.

Closes: docker-neo4j-integration"
```
