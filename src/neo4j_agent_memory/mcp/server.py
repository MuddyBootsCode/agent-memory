"""MCP Server implementation for Neo4j Agent Memory.

Provides a Model Context Protocol server using FastMCP that exposes
memory capabilities as tools, resources, and prompts for AI platforms.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient

logger = logging.getLogger(__name__)

try:
    from fastmcp import FastMCP

    def create_mcp_server(
        settings: Any = None,
        *,
        server_name: str = "neo4j-agent-memory",
    ) -> FastMCP:
        """Create a configured FastMCP server.

        The server uses a lifespan to manage the async MemoryClient lifecycle.
        Tools, resources, and prompts are registered on the returned server.

        Args:
            settings: MemorySettings for Neo4j connection. If None, the server
                is created without a lifespan (useful for testing).
            server_name: Server name for MCP registration.

        Returns:
            Configured FastMCP server instance.

        Example:
            from neo4j_agent_memory import MemorySettings
            from neo4j_agent_memory.mcp import create_mcp_server

            settings = MemorySettings(...)
            server = create_mcp_server(settings)
            server.run()
        """
        lifespan = None
        if settings is not None:

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
                    docker_auto=docker_cfg.get("docker_auto", True),
                    startup_timeout=docker_cfg.get("startup_timeout", 60),
                    compose_file=docker_cfg.get("compose_file"),
                )

                async with docker_mgr:
                    # Phase 2: Connect MemoryClient with retries
                    try:
                        client, client_cm = await connect_with_retry(
                            lambda: _MemoryClient(settings),
                            max_attempts=5,
                            delay=2.0,
                        )
                    except RuntimeError as exc:
                        logger.error(
                            "Neo4j unavailable — server will start but "
                            "tools will return errors until Neo4j is "
                            "reachable: %s",
                            exc,
                        )
                        yield {"client": None}
                        return

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

        mcp = FastMCP(
            server_name,
            lifespan=lifespan,
        )

        from neo4j_agent_memory.mcp._prompts import register_prompts
        from neo4j_agent_memory.mcp._resources import register_resources
        from neo4j_agent_memory.mcp._tools import register_tools

        register_tools(mcp)
        register_resources(mcp)
        register_prompts(mcp)

        return mcp

    class Neo4jMemoryMCPServer:
        """MCP server exposing Neo4j Agent Memory capabilities.

        Backward-compatible wrapper that accepts a pre-connected MemoryClient.
        For new code, prefer ``create_mcp_server(settings)`` instead.

        Example:
            from neo4j_agent_memory import MemoryClient, MemorySettings
            from neo4j_agent_memory.mcp import Neo4jMemoryMCPServer

            settings = MemorySettings(...)
            async with MemoryClient(settings) as client:
                server = Neo4jMemoryMCPServer(client)
                await server.run()

        Tools:
            - memory_search: Hybrid vector + graph search
            - memory_store: Store messages, facts, preferences
            - entity_lookup: Get entity with relationships
            - conversation_history: Get conversation for session
            - graph_query: Execute read-only Cypher queries
        """

        def __init__(
            self,
            memory_client: MemoryClient,
            *,
            server_name: str = "neo4j-agent-memory",
        ):
            """Initialize the MCP server with a pre-connected client.

            Args:
                memory_client: Connected MemoryClient instance.
                server_name: Server name for MCP registration.
            """
            self._client = memory_client

            @asynccontextmanager
            async def _preconnected_lifespan(server: FastMCP):
                yield {"client": memory_client}

            self._mcp = FastMCP(
                server_name,
                lifespan=_preconnected_lifespan,
            )

            from neo4j_agent_memory.mcp._prompts import register_prompts
            from neo4j_agent_memory.mcp._resources import register_resources
            from neo4j_agent_memory.mcp._tools import register_tools

            register_tools(self._mcp)
            register_resources(self._mcp)
            register_prompts(self._mcp)

        async def run(self) -> None:
            """Run the MCP server using stdio transport."""
            await self._mcp.run_async(transport="stdio")

        async def run_sse(self, host: str = "127.0.0.1", port: int = 8080) -> None:
            """Run the MCP server using SSE transport.

            Args:
                host: Host to bind to.
                port: Port to listen on.
            """
            await self._mcp.run_async(transport="sse", host=host, port=port)

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
        docker_startup_timeout: int = 60,
        compose_file: str | None = None,
    ) -> None:
        """Run the MCP server with Neo4j connection.

        Convenience function for CLI usage.

        Args:
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            neo4j_database: Neo4j database name.
            transport: Transport type (stdio, sse, or http).
            host: Host for network transports.
            port: Port for network transports.
            docker_auto: Enable automatic Docker container management.
            docker_startup_timeout: Max seconds to wait for Neo4j startup.
            compose_file: Path to docker-compose.yml (auto-detected if None).
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
            "startup_timeout": docker_startup_timeout,
            "compose_file": compose_file,
        }

        server = create_mcp_server(settings, server_name="neo4j-agent-memory")

        if transport == "sse":
            await server.run_async(transport="sse", host=host, port=port)
        elif transport == "http":
            await server.run_async(transport="http", host=host, port=port)
        else:
            await server.run_async(transport="stdio")

except ImportError:
    # FastMCP not installed
    class Neo4jMemoryMCPServer:  # type: ignore[no-redef]
        """Placeholder when FastMCP is not installed."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "FastMCP not installed. Install with: pip install neo4j-agent-memory[mcp]"
            )

    def create_mcp_server(*args: Any, **kwargs: Any) -> Neo4jMemoryMCPServer:  # type: ignore[misc]
        raise ImportError(
            "FastMCP not installed. Install with: pip install neo4j-agent-memory[mcp]"
        )


def main() -> None:
    """CLI entry point for running the MCP server."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Neo4j Agent Memory MCP Server")
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="MCP transport type",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for network transports (use 0.0.0.0 to expose on all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for network transports",
    )
    parser.add_argument(
        "--neo4j-docker-auto",
        default=os.environ.get("NEO4J_DOCKER_AUTO", "true").lower()
        in ("true", "1", "yes"),
        action=argparse.BooleanOptionalAction,
        help="Enable automatic Docker container management (default: true)",
    )
    parser.add_argument(
        "--compose-file",
        default=os.environ.get("NEO4J_COMPOSE_FILE"),
        help="Path to docker-compose.yml (auto-detected if not specified)",
    )
    parser.add_argument(
        "--neo4j-docker-startup-timeout",
        type=int,
        default=int(os.environ.get("NEO4J_DOCKER_STARTUP_TIMEOUT", "60")),
        help="Max seconds to wait for Neo4j startup (default: 60)",
    )

    args = parser.parse_args()

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
            docker_startup_timeout=args.neo4j_docker_startup_timeout,
            compose_file=args.compose_file,
        )
    )


if __name__ == "__main__":
    main()
