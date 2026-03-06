"""Shared utilities for MCP tool, resource, and prompt modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import Context

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient


def get_client(ctx: Context) -> MemoryClient:
    """Get MemoryClient from lifespan context.

    Args:
        ctx: FastMCP context with lifespan data.

    Returns:
        The MemoryClient instance.

    Raises:
        RuntimeError: If Neo4j was not available at startup.
    """
    client = ctx.request_context.lifespan_context["client"]
    if client is None:
        raise RuntimeError(
            "Neo4j is not connected. Please ensure Docker Desktop is running "
            "and restart Claude Desktop, or start Neo4j manually on "
            "bolt://localhost:7687."
        )
    return client
