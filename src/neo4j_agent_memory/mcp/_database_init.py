"""Database initialization for vertical databases."""

from __future__ import annotations

import logging
import os

from neo4j_agent_memory.verticals import get_default_vertical_names

logger = logging.getLogger(__name__)


def get_configured_verticals() -> list[str]:
    """Get list of vertical databases from env or defaults."""
    env_val = os.environ.get("NAM_VERTICALS", "")
    if env_val.strip():
        return [v.strip() for v in env_val.split(",") if v.strip()]
    return get_default_vertical_names()


async def ensure_databases_exist(driver) -> list[str]:
    """Create vertical databases if they don't exist.

    Must be called with a driver connected to the Neo4j instance.
    Database creation commands run against the 'system' database.

    Returns:
        List of database names that were created or already existed.
    """
    verticals = get_configured_verticals()
    created = []

    async with driver.session(database="system") as session:
        for db_name in verticals:
            try:
                await session.run(
                    f"CREATE DATABASE {db_name} IF NOT EXISTS"
                )
                created.append(db_name)
                logger.info("Database '%s' ready", db_name)
            except Exception as e:
                logger.error(
                    "Failed to create database '%s': %s", db_name, e
                )

    return created
