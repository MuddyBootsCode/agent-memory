# Multi-Database Verticals Implementation Plan

## Overview

Expand the neo4j-agent-memory-mcp system to support dedicated Neo4j databases for different work verticals (meetings, projects, research) with vertical-specific ontologies, a BAML-powered query router that classifies both reads and writes, and cross-database reference nodes in the general database. Every MCP tool gains the ability to fan out across multiple databases and merge results.

The system includes a confidence gate that detects vague or ambiguous queries and returns disambiguation options for the calling LLM to present to the user, plus a post-retrieval re-ranker that scores results against the original query to filter noise — especially important for broad fan-out queries.

## Current State Analysis

- **Single Neo4j database** (`neo4j` default) with one `MemoryClient` via FastMCP lifespan
- **Single ontology**: POLE+O model (Person, Organization, Location, Event, Object)
- **8 MCP tools** all route through `get_client(ctx)` returning a single client
- **BAML functions**: `ExtractEntities` (POLE+O), `ExtractReasoning`, `SynthesizeExplanation`
- **Neo4j Enterprise** already in Docker — natively supports multiple databases
- **No routing logic** — all queries hit one database, no classification

### Key Discoveries:
- `MemoryClient` takes a single `Neo4jConfig.database` param — need one client per DB ([server.py:238-245](src/neo4j_agent_memory/mcp/server.py#L238-L245))
- All tools use `get_client(ctx)` from `_common.py` — single point to refactor ([_common.py:13-32](src/neo4j_agent_memory/mcp/_common.py#L13-L32))
- Neo4j Python driver supports `driver.session(database="meetings")` per-transaction
- `CREATE DATABASE IF NOT EXISTS` must run against `system` database
- Cross-DB queries require application-level fan-out (no native cross-DB JOINs)
- Neo4j Docker image runs `.cypher` files from `/docker-entrypoint-initdb.d/` on first startup
- Each database has independent indexes and constraints

## Desired End State

A system where:
1. Neo4j hosts 4 databases: `neo4j` (general), `meetings`, `projects`, `research`
2. Each vertical DB has its own ontology with domain-specific entity types and relationships
3. Every MCP tool call is classified by a BAML router that determines target database(s)
4. Queries can fan out across multiple DBs in parallel and return merged results
5. Writes are routed to the appropriate vertical DB by the BAML router
6. The general DB holds proxy nodes referencing entities in vertical DBs
7. New verticals can be added by defining a BAML ontology + config entry
8. Vague queries trigger a disambiguation response so the calling LLM can ask the user
9. All fan-out results pass through a BAML re-ranker that scores relevance and filters noise

### Verification:
- `SHOW DATABASES` returns all 4 databases in `online` state
- `memory_search("sprint planning meeting notes")` routes to meetings + projects DBs
- `memory_store` of a meeting note creates nodes in the meetings DB
- `entity_lookup` resolves cross-DB references via proxy nodes
- All existing tests continue to pass (general DB unchanged)

## What We're NOT Doing

- Composite databases (adds complexity, application-level fan-out is sufficient)
- Schema migration tooling (databases are created fresh with init scripts)
- Per-vertical MCP tools (tools remain generic, routing is transparent)
- Custom Neo4j plugins or procedures
- Authentication per-database (same Neo4j user for all DBs)
- Real-time sync between vertical DBs (cross-references are created at write time)

## Implementation Approach

The approach is layered: infrastructure first (databases + clients), then routing intelligence (BAML), then ontologies (per-vertical schemas), then tool refactoring, then cross-DB references. Each phase produces a testable increment.

---

## Phase 1: Docker & Database Infrastructure

### Overview
Create vertical databases at container startup via init script, and as a fallback in the application lifespan. Manage multiple `MemoryClient` instances in a `ClientRegistry`.

### Changes Required:

#### 1. Docker Init Script
**File**: `neo4j-init/init-databases.cypher` (new)
**Changes**: Cypher script run automatically on fresh container startup

```cypher
// Creates vertical databases on first container initialization
// Safe to re-run: IF NOT EXISTS makes it idempotent
CREATE DATABASE meetings IF NOT EXISTS;
CREATE DATABASE projects IF NOT EXISTS;
CREATE DATABASE research IF NOT EXISTS;
```

#### 2. Docker Compose Update
**File**: `docker-compose.yml`
**Changes**: Mount init script directory into container

```yaml
services:
  neo4j:
    image: neo4j:5-enterprise
    container_name: neo4j-agent-memory
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: ${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD:-graphmemory}
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: "apoc.*"
      NEO4J_dbms_security_procedures_allowlist: "apoc.*"
      NEO4J_server_memory_heap_initial__size: "2g"
      NEO4J_server_memory_heap_max__size: "4g"
      NEO4J_server_memory_pagecache_size: "2g"
      NEO4J_dbms_memory_transaction_total_max: "1g"
    volumes:
      - neo4j-data:/data
      - ./neo4j-init:/docker-entrypoint-initdb.d
    healthcheck:
      test: cypher-shell -u ${NEO4J_USER:-neo4j} -p ${NEO4J_PASSWORD:-graphmemory} "RETURN 1"
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    deploy:
      resources:
        limits:
          memory: 8g

volumes:
  neo4j-data:
```

#### 3. Client Registry
**File**: `src/neo4j_agent_memory/mcp/_registry.py` (new)
**Changes**: New module managing multiple MemoryClient instances

```python
"""Registry managing MemoryClient instances for multiple databases."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient, MemorySettings

logger = logging.getLogger(__name__)

# Default vertical databases
DEFAULT_VERTICALS = ["meetings", "projects", "research"]


class ClientRegistry:
    """Manages MemoryClient instances for multiple Neo4j databases.

    Holds a dict of {database_name: MemoryClient} and provides
    lookup, iteration, and parallel query execution.
    """

    def __init__(self) -> None:
        self._clients: dict[str, MemoryClient] = {}
        self._context_managers: dict[str, Any] = {}

    @property
    def databases(self) -> list[str]:
        """List of registered database names."""
        return list(self._clients.keys())

    def get(self, database: str) -> MemoryClient:
        """Get client for a specific database."""
        if database not in self._clients:
            raise KeyError(
                f"No client registered for database '{database}'. "
                f"Available: {self.databases}"
            )
        return self._clients[database]

    @property
    def general(self) -> MemoryClient:
        """Get the general (default) database client."""
        # Try 'neo4j' first, then first registered client
        for name in ("neo4j", "general"):
            if name in self._clients:
                return self._clients[name]
        raise RuntimeError("No general database client registered")

    def register(
        self, database: str, client: MemoryClient, context_manager: Any = None
    ) -> None:
        """Register a client for a database."""
        self._clients[database] = client
        if context_manager is not None:
            self._context_managers[database] = context_manager

    async def close_all(self) -> None:
        """Close all registered clients."""
        for name, cm in self._context_managers.items():
            try:
                await cm.__aexit__(None, None, None)
                logger.info("Closed client for database '%s'", name)
            except Exception as e:
                logger.warning("Error closing client for '%s': %s", name, e)
        self._clients.clear()
        self._context_managers.clear()

    async def query_multiple(
        self,
        databases: list[str],
        query_fn,
    ) -> dict[str, Any]:
        """Execute a query function against multiple databases in parallel.

        Args:
            databases: List of database names to query.
            query_fn: Async callable(client, db_name) -> result.

        Returns:
            Dict of {database_name: result}.
        """
        async def _run(db_name: str):
            client = self.get(db_name)
            try:
                return db_name, await query_fn(client, db_name)
            except Exception as e:
                logger.warning("Query failed on '%s': %s", db_name, e)
                return db_name, {"error": str(e)}

        results = await asyncio.gather(
            *[_run(db) for db in databases],
            return_exceptions=False,
        )
        return dict(results)
```

#### 4. Database Initialization in Lifespan
**File**: `src/neo4j_agent_memory/mcp/_database_init.py` (new)
**Changes**: Ensures databases exist (fallback for existing volumes where Docker init didn't run)

```python
"""Database initialization for vertical databases."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_configured_verticals() -> list[str]:
    """Get list of vertical databases from env or defaults."""
    env_val = os.environ.get("NAM_VERTICALS", "")
    if env_val.strip():
        return [v.strip() for v in env_val.split(",") if v.strip()]
    return ["meetings", "projects", "research"]


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
```

#### 5. Updated _common.py
**File**: `src/neo4j_agent_memory/mcp/_common.py`
**Changes**: Support both single client (backward compat) and registry access

```python
"""Shared utilities for MCP tool, resource, and prompt modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import Context

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient
    from neo4j_agent_memory.mcp._registry import ClientRegistry


def get_client(ctx: Context) -> MemoryClient:
    """Get the general MemoryClient from lifespan context.

    Backward-compatible: works with both single-client and
    registry-based setups.
    """
    lifespan = ctx.request_context.lifespan_context

    # New registry-based access
    registry = lifespan.get("registry")
    if registry is not None:
        return registry.general

    # Legacy single-client access
    client = lifespan.get("client")
    if client is None:
        raise RuntimeError(
            "Neo4j is not connected. Please ensure Docker Desktop is running "
            "and restart Claude Desktop, or start Neo4j manually on "
            "bolt://localhost:7687."
        )
    return client


def get_registry(ctx: Context) -> ClientRegistry:
    """Get the ClientRegistry from lifespan context."""
    registry = ctx.request_context.lifespan_context.get("registry")
    if registry is None:
        raise RuntimeError(
            "ClientRegistry not available. Multi-database support "
            "requires NAM_VERTICALS to be configured."
        )
    return registry
```

#### 6. Updated Server Lifespan
**File**: `src/neo4j_agent_memory/mcp/server.py`
**Changes**: Create multiple MemoryClients in lifespan, yield registry in context

The lifespan function (lines 51-120) needs to be updated to:
1. Connect the general client as before
2. Call `ensure_databases_exist()` using the driver from the general client
3. Create a `MemoryClient` for each vertical database
4. Register all clients in a `ClientRegistry`
5. Yield `{"client": general_client, "registry": registry}`

```python
@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage Docker container and multi-database MemoryClient lifecycle."""
    import os

    from neo4j_agent_memory import MemoryClient as _MemoryClient
    from neo4j_agent_memory.config.settings import Neo4jConfig
    from neo4j_agent_memory.mcp._database_init import (
        ensure_databases_exist,
        get_configured_verticals,
    )
    from neo4j_agent_memory.mcp._docker import (
        Neo4jDockerManager,
        connect_with_retry,
    )
    from neo4j_agent_memory.mcp._registry import ClientRegistry

    # Patch factory to support BAML extraction [RFI-R1]
    import neo4j_agent_memory.extraction.factory as _factory_mod
    from neo4j_agent_memory.extraction.factory_ext import (
        create_extractor as _ext_create_extractor,
    )
    _factory_mod.create_extractor = _ext_create_extractor
    logger.info("Extraction factory patched with BAML support")

    docker_cfg = getattr(settings, "_docker_config", {})
    neo4j_cfg = settings.neo4j
    docker_mgr = Neo4jDockerManager(
        uri=str(neo4j_cfg.uri),
        docker_auto=docker_cfg.get("docker_auto", True),
        startup_timeout=docker_cfg.get("startup_timeout", 60),
        compose_file=docker_cfg.get("compose_file"),
    )

    registry = ClientRegistry()

    async with docker_mgr:
        # Phase 1: Connect general client
        try:
            client, client_cm = await connect_with_retry(
                lambda: _MemoryClient(settings),
                max_attempts=5,
                delay=2.0,
            )
        except RuntimeError as exc:
            logger.error("Neo4j unavailable: %s", exc)
            yield {"client": None, "registry": None}
            return

        registry.register("neo4j", client, client_cm)

        # Phase 2: Ensure vertical databases exist
        try:
            driver = client.graph._driver  # Access underlying driver
            verticals = await ensure_databases_exist(driver)
        except Exception as e:
            logger.warning(
                "Could not create vertical databases: %s. "
                "Continuing with general DB only.", e
            )
            verticals = []

        # Phase 3: Create clients for each vertical
        for db_name in verticals:
            try:
                from pydantic import SecretStr

                vertical_settings = MemorySettings(
                    neo4j=Neo4jConfig(
                        uri=neo4j_cfg.uri,
                        username=neo4j_cfg.username,
                        password=neo4j_cfg.password,
                        database=db_name,
                    )
                )
                v_client, v_cm = await connect_with_retry(
                    lambda s=vertical_settings: _MemoryClient(s),
                    max_attempts=3,
                    delay=2.0,
                )
                registry.register(db_name, v_client, v_cm)
                logger.info("Client ready for database '%s'", db_name)
            except Exception as e:
                logger.warning(
                    "Failed to connect to '%s': %s", db_name, e
                )

        logger.info(
            "ClientRegistry ready with databases: %s",
            registry.databases,
        )

        try:
            yield {"client": client, "registry": registry}
        finally:
            await registry.close_all()
```

### Success Criteria:

#### Automated Verification:
- [ ] `docker compose up -d` creates container with init script mounted
- [ ] `cypher-shell -d system "SHOW DATABASES"` lists neo4j, meetings, projects, research
- [ ] Server starts with logs showing "Client ready for database 'meetings'" etc.
- [ ] Existing tests pass (backward-compatible `get_client()` still works)

#### Manual Verification:
- [ ] Fresh volume: databases created by Docker init script
- [ ] Existing volume: databases created by lifespan fallback
- [ ] Neo4j Browser at :7474 shows all databases in dropdown

---

## Phase 2: BAML Query Router

### Overview
Create a BAML function that classifies queries and storage requests into target vertical(s) with confidence scores. Add a Python wrapper that integrates with the tool layer.

### Changes Required:

#### 1. BAML Routing Definitions
**File**: `baml_src/routing.baml` (new)
**Changes**: Define the routing function and types

```baml
// Database vertical classification for multi-DB routing

enum QueryVertical {
  MEETINGS @alias("k1")
    @description("Meetings, calendars, scheduling, attendees, agendas, standups, syncs, 1:1s, action items from meetings, meeting notes")
  PROJECTS @alias("k2")
    @description("Projects, tasks, milestones, deliverables, sprints, backlogs, dependencies, team assignments, project status, deadlines")
  RESEARCH @alias("k3")
    @description("Research notes, papers, findings, knowledge base, citations, experiments, hypotheses, literature, sources, analysis results")
  GENERAL @alias("k4")
    @description("General memory, personal preferences, facts, people, organizations, locations, conversation history, reasoning traces, anything that doesn't clearly fit other categories")
}

class RoutingTarget {
  vertical QueryVertical @description("The database vertical to query")
  confidence float @description("Confidence this vertical is relevant, 0.0 to 1.0")
  reasoning string @description("Brief explanation of why this vertical is relevant")
}

class RoutingDecision {
  targets RoutingTarget[] @description("Verticals to query, ordered by confidence descending. Only include verticals with confidence > 0.3")
  primary_vertical QueryVertical @description("The single most relevant vertical")
  requires_fanout bool @description("True if multiple verticals must be queried to fully answer the query")
  ambiguous bool @description("True if the query is too vague to route confidently. Set true when no vertical has confidence > 0.6 or the query lacks specificity")
  disambiguation_options string[]? @description("If ambiguous, provide 2-3 concrete interpretations of what the user might mean, each mentioning which vertical it would search. Only populate when ambiguous is true.")
}

function RouteQuery(query: string, context: string?) -> RoutingDecision {
  client Anthropic
  prompt #"
    You are a query router for a knowledge graph with multiple database verticals.
    Analyze the user query and determine which database(s) should be searched.

    Rules:
    - Set requires_fanout to true ONLY when the query genuinely spans multiple domains
    - Confidence reflects how certain you are the vertical contains relevant data
    - Only include verticals with confidence > 0.3
    - GENERAL is the fallback when no specific vertical matches
    - Most queries should target 1-2 verticals, rarely more
    - Set ambiguous to true when the query is vague, lacks context, or could
      reasonably apply to multiple unrelated verticals (e.g., "what did we talk about?",
      "anything new?", "what did Sarah say?")
    - When ambiguous, provide 2-3 specific interpretations the user might mean,
      each mentioning the vertical it would search

    {% if context %}
    ## Recent Context
    {{ context }}
    {% endif %}

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ query }}
  "#
}

function RouteStorage(content: string, memory_type: string, context: string?) -> RoutingDecision {
  client Anthropic
  prompt #"
    You are a storage router for a knowledge graph. Determine which database
    vertical should store this content.

    For storage, pick ONE primary vertical. Only set requires_fanout to true
    if the content genuinely belongs in multiple databases (rare).

    Memory type being stored: {{ memory_type }}

    {% if context %}
    ## Context
    {{ context }}
    {% endif %}

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ content }}
  "#
}

test MeetingQuery {
  functions [RouteQuery]
  args {
    query "What was discussed in yesterday's standup?"
    context null
  }
}

test CrossDomainQuery {
  functions [RouteQuery]
  args {
    query "What research findings were presented in the project kickoff meeting?"
    context null
  }
}

test ProjectStorage {
  functions [RouteStorage]
  args {
    content "The migration project milestone 2 is delayed by a week due to API dependency"
    memory_type "message"
    context null
  }
}

test AmbiguousQuery {
  functions [RouteQuery]
  args {
    query "What did Sarah say?"
    context "Sarah presented research findings at Monday's team meeting about the migration project"
  }
}

test VagueQuery {
  functions [RouteQuery]
  args {
    query "anything new?"
    context null
  }
}
```

#### 1b. BAML Re-Ranker Definition
**File**: `baml_src/reranking.baml` (new)
**Changes**: Define a post-retrieval re-ranker that scores results against the original query

```baml
// Post-retrieval re-ranking for multi-database fan-out results

class ResultItem {
  id string @description("Result ID")
  content string @description("The result content or summary")
  source_db string @description("Which database this came from")
  result_type string @description("Type: message, entity, preference, trace")
}

class ScoredResult {
  id string @description("Result ID (pass through from input)")
  relevance float @description("Relevance to the original query, 0.0 to 1.0")
  keep bool @description("True if this result is relevant enough to show to the user")
  reasoning string @description("Brief explanation of relevance assessment")
}

class RerankOutput {
  scored_results ScoredResult[] @description("Results scored and filtered, ordered by relevance descending")
  total_input int @description("Number of results received")
  total_kept int @description("Number of results kept after filtering")
}

function RerankResults(query: string, results: ResultItem[]) -> RerankOutput {
  client Anthropic
  prompt #"
    You are a relevance judge. Score each search result for how well it answers
    the user's original query.

    Rules:
    - Score 0.0-1.0 based on direct relevance to the query
    - Set keep=true only for results with relevance >= 0.4
    - A result that partially answers the query is still relevant
    - A result that is tangentially related but not useful should score low
    - Be aggressive about filtering noise — better to show 3 great results than 10 mediocre ones

    ## Original Query
    {{ query }}

    ## Results to Score
    {% for r in results %}
    ### Result {{ loop.index }} (id: {{ r.id }}, from: {{ r.source_db }}, type: {{ r.result_type }})
    {{ r.content }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

test RerankMeetingResults {
  functions [RerankResults]
  args {
    query "What actions came out of Monday's standup?"
    results [
      {
        id "1"
        content "Monday standup: Sarah to finish API docs by Wednesday. Marcus to review PR #42."
        source_db "meetings"
        result_type "message"
      },
      {
        id "2"
        content "Sarah mentioned she prefers morning meetings over afternoon ones."
        source_db "meetings"
        result_type "preference"
      },
      {
        id "3"
        content "The API documentation project milestone is due Friday."
        source_db "projects"
        result_type "message"
      }
    ]
  }
}
```

#### 2. Python Router Wrapper
**File**: `src/neo4j_agent_memory/routing/__init__.py` (new)
**Changes**: Empty init for package

```python
"""Query routing module for multi-database vertical support."""
```

#### 3. Router Implementation
**File**: `src/neo4j_agent_memory/routing/router.py` (new)
**Changes**: Python class wrapping BAML routing functions

```python
"""BAML-powered query router for multi-database verticals."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Mapping from BAML enum values to Neo4j database names
VERTICAL_TO_DB: dict[str, str] = {
    "MEETINGS": "meetings",
    "PROJECTS": "projects",
    "RESEARCH": "research",
    "GENERAL": "neo4j",
}


class QueryRouter:
    """Routes queries and storage requests to appropriate database verticals."""

    def __init__(
        self,
        available_databases: list[str],
        confidence_threshold: float = 0.3,
    ) -> None:
        self._available_dbs = set(available_databases)
        self._threshold = confidence_threshold
        self._enabled = os.environ.get(
            "NAM_ROUTING_ENABLED", "true"
        ).lower() in ("true", "1", "yes")

    async def route_query(
        self,
        query: str,
        context: str | None = None,
    ) -> RoutingResult:
        """Route a search query to target database(s).

        Falls back to general DB if routing is disabled or fails.
        """
        if not self._enabled:
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

        try:
            from neo4j_agent_memory.baml_client.async_client import b

            decision = await b.RouteQuery(query=query, context=context)
            return self._to_result(decision)
        except Exception as e:
            logger.warning("Route failed, defaulting to general: %s", e)
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

    async def route_storage(
        self,
        content: str,
        memory_type: str,
        context: str | None = None,
    ) -> RoutingResult:
        """Route a storage request to the target database.

        Falls back to general DB if routing is disabled or fails.
        """
        if not self._enabled:
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

        try:
            from neo4j_agent_memory.baml_client.async_client import b

            decision = await b.RouteStorage(
                content=content,
                memory_type=memory_type,
                context=context,
            )
            return self._to_result(decision)
        except Exception as e:
            logger.warning("Storage route failed, defaulting to general: %s", e)
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

    def _to_result(self, decision) -> RoutingResult:
        """Convert BAML RoutingDecision to internal RoutingResult."""
        targets = []
        for target in decision.targets:
            db_name = VERTICAL_TO_DB.get(target.vertical.value, "neo4j")
            if db_name in self._available_dbs and target.confidence >= self._threshold:
                targets.append((db_name, target.confidence))

        # Ensure at least general DB
        if not targets:
            targets = [("neo4j", 1.0)]

        primary = VERTICAL_TO_DB.get(decision.primary_vertical.value, "neo4j")
        if primary not in self._available_dbs:
            primary = "neo4j"

        return RoutingResult(
            targets=targets,
            primary=primary,
            requires_fanout=decision.requires_fanout,
            ambiguous=decision.ambiguous,
            disambiguation_options=decision.disambiguation_options or [],
        )


class ResultReranker:
    """Re-ranks results from multi-database fan-out queries."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score and filter results by relevance to the original query.

        Args:
            query: The original user query.
            results: List of result dicts with id, content, _source_db, type.

        Returns:
            Filtered and re-ordered results.
        """
        if not self._enabled or not results:
            return results

        # Skip re-ranking for small result sets (not worth the LLM call)
        if len(results) <= 3:
            return results

        try:
            from neo4j_agent_memory.baml_client.async_client import b

            items = [
                {
                    "id": r.get("id", ""),
                    "content": r.get("content") or r.get("name") or r.get("task") or str(r),
                    "source_db": r.get("_source_db", "unknown"),
                    "result_type": r.get("_result_type", "unknown"),
                }
                for r in results
            ]

            reranked = await b.RerankResults(query=query, results=items)

            # Build lookup of kept results with their scores
            kept_ids = {}
            for scored in reranked.scored_results:
                if scored.keep:
                    kept_ids[scored.id] = scored.relevance

            # Filter and re-sort original results
            filtered = [r for r in results if r.get("id", "") in kept_ids]
            filtered.sort(
                key=lambda r: kept_ids.get(r.get("id", ""), 0),
                reverse=True,
            )

            logger.info(
                "Re-ranked %d results -> %d kept for query: %s",
                len(results), len(filtered), query[:80],
            )
            return filtered

        except Exception as e:
            logger.warning("Re-ranking failed, returning unfiltered: %s", e)
            return results


class RoutingResult:
    """Result of a routing decision."""

    def __init__(
        self,
        targets: list[tuple[str, float]],
        primary: str,
        requires_fanout: bool,
        ambiguous: bool = False,
        disambiguation_options: list[str] | None = None,
    ) -> None:
        self.targets = targets  # [(db_name, confidence), ...]
        self.primary = primary
        self.requires_fanout = requires_fanout
        self.ambiguous = ambiguous
        self.disambiguation_options = disambiguation_options or []

    @property
    def target_databases(self) -> list[str]:
        """List of database names to query, ordered by confidence."""
        return [db for db, _ in self.targets]

    @property
    def primary_database(self) -> str:
        """The single most relevant database."""
        return self.primary
```

#### 4. Router Initialization in Lifespan
**File**: `src/neo4j_agent_memory/mcp/server.py`
**Changes**: Create router after registry, add to lifespan context

Add after registry creation in the lifespan:

```python
from neo4j_agent_memory.routing.router import QueryRouter, ResultReranker

router = QueryRouter(available_databases=registry.databases)
reranker = ResultReranker(enabled=True)
# ...
yield {"client": client, "registry": registry, "router": router, "reranker": reranker}
```

#### 5. Router Access Helper
**File**: `src/neo4j_agent_memory/mcp/_common.py`
**Changes**: Add `get_router()` helper

```python
def get_router(ctx: Context) -> QueryRouter:
    """Get the QueryRouter from lifespan context."""
    from neo4j_agent_memory.routing.router import QueryRouter

    router = ctx.request_context.lifespan_context.get("router")
    if router is None:
        # Return a disabled router that always routes to general
        return QueryRouter(available_databases=["neo4j"])
    return router


def get_reranker(ctx: Context) -> ResultReranker:
    """Get the ResultReranker from lifespan context."""
    from neo4j_agent_memory.routing.router import ResultReranker

    reranker = ctx.request_context.lifespan_context.get("reranker")
    if reranker is None:
        return ResultReranker(enabled=False)
    return reranker
```

### Success Criteria:

#### Automated Verification:
- [ ] `baml test` passes for all 5 routing test cases + 1 reranking test
- [ ] BAML code generation succeeds: `baml-cli generate`
- [ ] Router falls back to general DB when `NAM_ROUTING_ENABLED=false`
- [ ] Router falls back to general DB on BAML call failure
- [ ] Ambiguous query returns `ambiguous: true` with disambiguation options
- [ ] Re-ranker filters irrelevant results from fan-out queries
- [ ] Re-ranker is skipped for result sets with 3 or fewer items

#### Manual Verification:
- [ ] "What was discussed in the standup?" routes to meetings
- [ ] "Sprint milestone delayed" routes to projects
- [ ] "Research findings on embeddings" routes to research
- [ ] "What research was in the kickoff meeting?" fans out to meetings + research
- [ ] "anything new?" returns disambiguation options, not results
- [ ] "What did Sarah say?" with no context returns disambiguation
- [ ] "What did Sarah say?" with context about a meeting routes correctly
- [ ] Fan-out results are visibly cleaner after re-ranking

---

## Phase 3: Vertical Ontologies

### Overview
Define domain-specific BAML extraction schemas for each vertical. Each vertical gets its own entity types, relationship types, and extraction function. The existing POLE+O extraction remains for the general database.

### Changes Required:

#### 1. Meetings Ontology
**File**: `baml_src/ontology_meetings.baml` (new)

```baml
// Meetings vertical ontology

enum MeetingEntityType {
  MEETING @description("A meeting event: standup, sync, 1:1, kickoff, retrospective, review")
  ATTENDEE @description("A person who attended or was invited to a meeting")
  AGENDA_ITEM @description("A topic or item on the meeting agenda")
  ACTION_ITEM @description("A task or follow-up assigned during a meeting")
  DECISION @description("A decision made during a meeting")
}

class MeetingEntity {
  name string @description("Entity name as it appears in text")
  type MeetingEntityType
  date string? @description("Date/time if mentioned, ISO format preferred")
  status string? @description("Status: scheduled, completed, cancelled, recurring")
  confidence float @description("Extraction confidence 0.0 to 1.0")
}

class MeetingRelation {
  source string @description("Source entity name")
  target string @description("Target entity name")
  relation_type string @description("Relationship: ATTENDED, PRESENTED, ASSIGNED_TO, DISCUSSED, FOLLOW_UP, DECIDED_IN, SCHEDULED_FOR, BLOCKED_BY")
  confidence float
}

class MeetingExtractionOutput {
  entities MeetingEntity[]
  relations MeetingRelation[]
}

function ExtractMeetingEntities(text: string) -> MeetingExtractionOutput {
  client Anthropic
  prompt #"
    Extract meeting-related entities and relationships from the text.

    Focus on: meetings, attendees, agenda items, action items, and decisions.

    For relations, use types like:
    ATTENDED, PRESENTED, ASSIGNED_TO, DISCUSSED, FOLLOW_UP,
    DECIDED_IN, SCHEDULED_FOR, BLOCKED_BY

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ text }}
  "#
}
```

#### 2. Projects Ontology
**File**: `baml_src/ontology_projects.baml` (new)

```baml
// Projects vertical ontology

enum ProjectEntityType {
  PROJECT @description("A project, initiative, or workstream")
  TASK @description("A task, ticket, or work item within a project")
  MILESTONE @description("A project milestone, deadline, or checkpoint")
  DELIVERABLE @description("A deliverable, artifact, or output of project work")
  TEAM @description("A team or group working on a project")
}

class ProjectEntity {
  name string @description("Entity name as it appears in text")
  type ProjectEntityType
  status string? @description("Status: active, completed, blocked, planned, cancelled")
  priority string? @description("Priority: critical, high, medium, low")
  confidence float @description("Extraction confidence 0.0 to 1.0")
}

class ProjectRelation {
  source string @description("Source entity name")
  target string @description("Target entity name")
  relation_type string @description("Relationship: DEPENDS_ON, ASSIGNED_TO, BLOCKED_BY, DELIVERS, PART_OF, OWNS, CONTRIBUTES_TO, TRACKS")
  confidence float
}

class ProjectExtractionOutput {
  entities ProjectEntity[]
  relations ProjectRelation[]
}

function ExtractProjectEntities(text: string) -> ProjectExtractionOutput {
  client Anthropic
  prompt #"
    Extract project-related entities and relationships from the text.

    Focus on: projects, tasks, milestones, deliverables, and teams.

    For relations, use types like:
    DEPENDS_ON, ASSIGNED_TO, BLOCKED_BY, DELIVERS,
    PART_OF, OWNS, CONTRIBUTES_TO, TRACKS

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ text }}
  "#
}
```

#### 3. Research Ontology
**File**: `baml_src/ontology_research.baml` (new)

```baml
// Research vertical ontology

enum ResearchEntityType {
  NOTE @description("A research note, observation, or write-up")
  FINDING @description("A research finding, result, or conclusion")
  SOURCE @description("A source: paper, article, book, dataset, URL")
  TOPIC @description("A research topic, theme, or area of investigation")
  EXPERIMENT @description("An experiment, test, or trial")
}

class ResearchEntity {
  name string @description("Entity name as it appears in text")
  type ResearchEntityType
  status string? @description("Status: draft, validated, refuted, in_progress")
  confidence float @description("Extraction confidence 0.0 to 1.0")
}

class ResearchRelation {
  source string @description("Source entity name")
  target string @description("Target entity name")
  relation_type string @description("Relationship: CITES, SUPPORTS, CONTRADICTS, BUILDS_ON, EXPLORES, PRODUCED_BY, RELATED_TOPIC, VALIDATES")
  confidence float
}

class ResearchExtractionOutput {
  entities ResearchEntity[]
  relations ResearchRelation[]
}

function ExtractResearchEntities(text: string) -> ResearchExtractionOutput {
  client Anthropic
  prompt #"
    Extract research-related entities and relationships from the text.

    Focus on: research notes, findings, sources, topics, and experiments.

    For relations, use types like:
    CITES, SUPPORTS, CONTRADICTS, BUILDS_ON,
    EXPLORES, PRODUCED_BY, RELATED_TOPIC, VALIDATES

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ text }}
  "#
}
```

#### 4. Vertical Extractor Dispatcher
**File**: `src/neo4j_agent_memory/extraction/vertical_extractor.py` (new)
**Changes**: Routes extraction to the correct ontology-specific BAML function based on target database

```python
"""Vertical-aware entity extraction dispatcher."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maps database name to BAML extraction function name
VERTICAL_EXTRACTORS = {
    "meetings": "ExtractMeetingEntities",
    "projects": "ExtractProjectEntities",
    "research": "ExtractResearchEntities",
}


async def extract_for_vertical(
    text: str,
    database: str,
) -> dict[str, Any] | None:
    """Extract entities using the vertical-specific ontology.

    Args:
        text: Text to extract from.
        database: Target database name.

    Returns:
        Extraction result dict or None if no vertical extractor exists.
    """
    if database not in VERTICAL_EXTRACTORS:
        return None  # Use default POLE+O extraction

    try:
        from neo4j_agent_memory.baml_client.async_client import b

        func_name = VERTICAL_EXTRACTORS[database]
        extract_fn = getattr(b, func_name)
        result = await extract_fn(text=text)

        return {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type.value if hasattr(e.type, "value") else str(e.type),
                    "confidence": e.confidence,
                }
                for e in result.entities
            ],
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "confidence": r.confidence,
                }
                for r in result.relations
            ],
        }
    except Exception as e:
        logger.warning(
            "Vertical extraction failed for '%s': %s", database, e
        )
        return None
```

### Success Criteria:

#### Automated Verification:
- [ ] `baml-cli generate` succeeds with all 3 new ontology files
- [ ] `baml test` passes for ontology extraction functions
- [ ] `extract_for_vertical("standup notes...", "meetings")` returns MeetingEntity types
- [ ] `extract_for_vertical("anything", "neo4j")` returns None (falls back to POLE+O)

#### Manual Verification:
- [ ] Meeting text extracts MEETING, ATTENDEE, ACTION_ITEM entities
- [ ] Project text extracts PROJECT, TASK, MILESTONE entities
- [ ] Research text extracts NOTE, FINDING, SOURCE entities
- [ ] Extraction quality is comparable to existing POLE+O extraction

---

## Phase 4: Tool Layer Refactor

### Overview
Update all 8 MCP tools to support multi-database routing. Each tool call goes through the BAML router, fans out to target databases in parallel, and merges results. Tools also gain an optional `database` parameter for explicit targeting.

### Changes Required:

#### 1. Result Merger Utility
**File**: `src/neo4j_agent_memory/mcp/_merge.py` (new)
**Changes**: Utility for merging and deduplicating results from multiple databases

```python
"""Result merging utilities for multi-database queries."""

from __future__ import annotations

from typing import Any


def merge_search_results(
    per_db_results: dict[str, dict[str, list]],
) -> dict[str, list]:
    """Merge search results from multiple databases.

    Combines lists by memory type, annotates each result with its
    source database, and deduplicates by ID.
    """
    merged: dict[str, list] = {}
    seen_ids: set[str] = set()

    for db_name, results in per_db_results.items():
        for memory_type, items in results.items():
            if memory_type not in merged:
                merged[memory_type] = []
            for item in items:
                item_id = item.get("id", "")
                if item_id and item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                item["_source_db"] = db_name
                merged[memory_type].append(item)

    # Sort each list by similarity/confidence if available
    for memory_type in merged:
        merged[memory_type].sort(
            key=lambda x: x.get("similarity") or x.get("confidence") or 0,
            reverse=True,
        )

    return merged


def merge_entity_results(
    per_db_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge entity lookup results from multiple databases."""
    found_entities = []
    all_neighbors = []

    for db_name, result in per_db_results.items():
        if result.get("found"):
            entity = result.get("entity", {})
            entity["_source_db"] = db_name
            found_entities.append(entity)
            for neighbor in result.get("neighbors", []):
                neighbor["_source_db"] = db_name
                all_neighbors.append(neighbor)

    if not found_entities:
        return {"found": False}

    return {
        "found": True,
        "entities": found_entities,
        "neighbors": all_neighbors,
        "databases_searched": list(per_db_results.keys()),
    }
```

#### 2. Updated Tools
**File**: `src/neo4j_agent_memory/mcp/_tools.py`
**Changes**: Each tool gains routing awareness. The pattern for each tool is:

1. Accept optional `database: str | None` parameter
2. If `database` is explicit, use that single DB
3. Otherwise, call the BAML router to get target DB(s)
4. Fan out queries via `registry.query_multiple()`
5. Merge results

Example for `memory_search` (the other tools follow the same pattern):

```python
@mcp.tool()
async def memory_search(
    ctx: Context,
    query: str,
    limit: int = 10,
    memory_types: list[str] | None = None,
    session_id: str | None = None,
    threshold: float = 0.7,
    database: str | None = None,
) -> str:
    """Search across all memory types using hybrid vector + graph search.

    Automatically routes to the most relevant database(s) using AI classification.
    Set database explicitly to target a specific vertical.

    If the query is ambiguous, returns disambiguation options instead of results
    so the calling LLM can ask the user for clarification.
    """
    from neo4j_agent_memory.mcp._merge import merge_search_results

    registry = get_registry(ctx)
    router = get_router(ctx)

    # Step 1: Determine target databases
    if database:
        target_dbs = [database]
        route = None
    else:
        route = await router.route_query(query)

        # Step 2: Confidence gate — if ambiguous, ask for clarification
        if route.ambiguous:
            return json.dumps({
                "ambiguous": True,
                "message": "This query could apply to several areas. Can you clarify?",
                "disambiguation_options": route.disambiguation_options,
                "suggestion": "You can also pass database='meetings' (or projects, research, neo4j) to target a specific vertical.",
            })

        target_dbs = route.target_databases

    if memory_types is None:
        memory_types = ["messages", "entities", "preferences", "traces"]

    # Step 3: Fan out search to target databases
    async def _search(client, db_name):
        results = {}
        # ... (existing search logic per memory_type, same as current)
        return results

    per_db = await registry.query_multiple(target_dbs, _search)
    merged = merge_search_results(per_db)

    # Step 4: Re-rank results if this was a fan-out query
    if route and route.requires_fanout and len(target_dbs) > 1:
        reranker = get_reranker(ctx)
        all_results = []
        for memory_type, items in merged.items():
            for item in items:
                item["_result_type"] = memory_type
            all_results.extend(items)

        reranked = await reranker.rerank(query, all_results)

        # Rebuild merged dict from reranked results
        merged = {}
        for item in reranked:
            rt = item.pop("_result_type", "unknown")
            if rt not in merged:
                merged[rt] = []
            merged[rt].append(item)

    return json.dumps({
        "results": merged,
        "query": query,
        "databases_searched": target_dbs,
        "reranked": route.requires_fanout if route else False,
    }, default=str)
```

The flow for each search tool call is:

```
Query + optional database param
    │
    ├── database explicit? → Skip routing, query that DB directly
    │
    └── database=None? → BAML RouteQuery
            │
            ├── ambiguous=true → Return disambiguation options (no DB query)
            │                     Calling LLM asks user, re-calls with database=
            │
            ├── high confidence, single DB → Query that DB → Return results
            │
            └── fan-out, multiple DBs → Query all targets in parallel
                                          → Merge results
                                          → BAML RerankResults (filter noise)
                                          → Return scored results
```

For `memory_store`, the pattern routes via `RouteStorage`:

```python
@mcp.tool()
async def memory_store(
    ctx: Context,
    memory_type: str,
    content: str,
    # ... existing params ...
    database: str | None = None,
) -> str:
    """Store a memory, automatically routed to the appropriate database vertical."""
    registry = get_registry(ctx)
    router = get_router(ctx)

    # Route to target database
    if database:
        target_db = database
    else:
        route = await router.route_storage(content, memory_type)
        target_db = route.primary_database

    client = registry.get(target_db)

    # ... existing storage logic using client ...
    # Add target_db to response
    result["database"] = target_db
```

The same pattern applies to all 8 tools:
- **memory_search**: Fan-out query, merge results
- **memory_store**: Route to single DB for write
- **entity_lookup**: Fan-out search, merge entities
- **conversation_history**: Route to single DB (sessions are DB-specific)
- **graph_query**: Route to single DB (Cypher is DB-specific)
- **add_reasoning_trace**: Route to single DB for write
- **explain_reasoning**: Fan-out search for traces
- **extract_reasoning**: Route to single DB for write

### Success Criteria:

#### Automated Verification:
- [ ] All tools accept optional `database` parameter
- [ ] `memory_search` with explicit `database="meetings"` only hits meetings DB
- [ ] `memory_search` without `database` calls BAML router
- [ ] Response JSON includes `databases_searched` and `reranked` fields
- [ ] Existing tool signatures remain backward-compatible (database defaults to None)
- [ ] Ambiguous queries return disambiguation JSON instead of empty results
- [ ] Re-ranking runs on fan-out queries with >3 results
- [ ] Re-ranking is skipped when `database` is explicit (no fan-out)

#### Manual Verification:
- [ ] "What was in the standup?" searches meetings DB
- [ ] Storing a meeting note creates nodes in meetings DB
- [ ] Cross-vertical query merges results from multiple DBs
- [ ] Explicit `database` override works for all tools
- [ ] Vague query → disambiguation → user clarifies → correct results
- [ ] Fan-out query returns re-ranked results with noise filtered out

---

## Phase 5: Cross-Database References

### Overview
When data is stored in a vertical database, create lightweight proxy nodes in the general database that reference the vertical data. This allows the general DB to serve as a unified index.

### Changes Required:

#### 1. Proxy Node Manager
**File**: `src/neo4j_agent_memory/mcp/_proxy.py` (new)
**Changes**: Creates and resolves cross-database proxy references

```python
"""Cross-database proxy node management."""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


async def create_proxy_reference(
    general_client,
    source_db: str,
    node_id: str,
    node_type: str,
    node_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a proxy node in the general DB referencing a vertical DB node.

    Args:
        general_client: MemoryClient for the general (neo4j) database.
        source_db: Name of the vertical database (e.g., "meetings").
        node_id: ID of the node in the vertical database.
        node_type: Type/label of the referenced node.
        node_name: Display name for the proxy.
        metadata: Optional additional metadata.

    Returns:
        ID of the created proxy node.
    """
    proxy_id = str(uuid.uuid4())

    await general_client.graph.execute_write(
        """
        CREATE (p:ProxyRef {
            id: $proxy_id,
            source_database: $source_db,
            external_id: $node_id,
            external_type: $node_type,
            name: $node_name,
            created_at: datetime()
        })
        """,
        {
            "proxy_id": proxy_id,
            "source_db": source_db,
            "node_id": node_id,
            "node_type": node_type,
            "node_name": node_name,
        },
    )

    logger.debug(
        "Created proxy ref %s -> %s:%s", proxy_id, source_db, node_id
    )
    return proxy_id


async def resolve_proxy_references(
    general_client,
    registry,
    entity_id: str,
) -> list[dict[str, Any]]:
    """Resolve proxy references for an entity, fetching data from vertical DBs.

    Args:
        general_client: MemoryClient for the general database.
        registry: ClientRegistry for accessing vertical clients.
        entity_id: Entity ID in the general database.

    Returns:
        List of resolved cross-database references.
    """
    # Find proxy refs linked to this entity
    rows = await general_client.graph.execute_read(
        """
        MATCH (e:Entity {id: $entity_id})-[:HAS_REFERENCE]->(p:ProxyRef)
        RETURN p.source_database AS db, p.external_id AS ext_id,
               p.external_type AS ext_type, p.name AS name
        """,
        {"entity_id": entity_id},
    )

    resolved = []
    for row in rows:
        db_name = row["db"]
        try:
            client = registry.get(db_name)
            # Look up the actual node in the vertical DB
            records = await client.graph.execute_read(
                """
                MATCH (n {id: $node_id})
                RETURN properties(n) AS props, labels(n) AS labels
                """,
                {"node_id": row["ext_id"]},
            )
            if records:
                resolved.append({
                    "source_database": db_name,
                    "external_id": row["ext_id"],
                    "type": row["ext_type"],
                    "name": row["name"],
                    "resolved": True,
                    "data": records[0]["props"],
                    "labels": records[0]["labels"],
                })
            else:
                resolved.append({
                    "source_database": db_name,
                    "external_id": row["ext_id"],
                    "type": row["ext_type"],
                    "name": row["name"],
                    "resolved": False,
                })
        except Exception as e:
            logger.warning("Failed to resolve proxy for %s:%s: %s", db_name, row["ext_id"], e)
            resolved.append({
                "source_database": db_name,
                "external_id": row["ext_id"],
                "resolved": False,
                "error": str(e),
            })

    return resolved
```

#### 2. Integrate Proxy Creation into memory_store
**File**: `src/neo4j_agent_memory/mcp/_tools.py`
**Changes**: After storing to a vertical DB, create proxy reference in general DB

When `memory_store` writes to a vertical DB (not general), also:
1. Extract a summary entity name from the stored content
2. Call `create_proxy_reference()` in the general DB
3. Optionally link the proxy to related entities in the general DB

#### 3. Integrate Proxy Resolution into entity_lookup
**File**: `src/neo4j_agent_memory/mcp/_tools.py`
**Changes**: After looking up an entity, check for and resolve proxy references

Add to the entity_lookup result:
```python
# After getting the entity, resolve cross-DB references
cross_refs = await resolve_proxy_references(
    general_client=registry.general,
    registry=registry,
    entity_id=entity_id,
)
if cross_refs:
    result["cross_references"] = cross_refs
```

### Success Criteria:

#### Automated Verification:
- [ ] Storing a meeting note creates a ProxyRef node in the general DB
- [ ] `MATCH (p:ProxyRef) RETURN p` in general DB shows proxy nodes
- [ ] `entity_lookup` returns `cross_references` when they exist
- [ ] Proxy resolution handles missing/offline vertical DBs gracefully

#### Manual Verification:
- [ ] Store a meeting → verify ProxyRef in general DB via Neo4j Browser
- [ ] Look up a person entity → see their meetings/projects as cross-references
- [ ] Cross-references show resolved data from vertical DBs

---

## Phase 6: Configuration & Testing

### Overview
Finalize configuration, environment variables, regenerate BAML client, and create integration tests.

### Changes Required:

#### 1. Environment Variable Updates
**File**: `.env.example`
**Changes**: Document all new env vars

```bash
# Existing
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphmemory
NEO4J_DATABASE=neo4j

# Multi-database verticals (comma-separated)
NAM_VERTICALS=meetings,projects,research

# Routing
NAM_ROUTING_ENABLED=true

# Extraction
NAM_EXTRACTION__BAML_ENABLED=true
NAM_EXTRACTION__BAML_CLIENT=Anthropic
```

#### 2. BAML Client Regeneration
**Command**: `baml-cli generate`
**Changes**: Regenerate Python client code after adding routing.baml and ontology files

#### 3. Integration Tests
**File**: `tests/test_multi_db.py` (new)
**Changes**: Tests for the multi-database pipeline

```python
"""Integration tests for multi-database vertical support."""

import pytest


class TestClientRegistry:
    """Tests for ClientRegistry."""

    def test_register_and_get(self):
        """Registry stores and retrieves clients by name."""

    def test_get_missing_raises(self):
        """Accessing unregistered database raises KeyError."""

    def test_general_property(self):
        """General property returns the neo4j client."""

    async def test_query_multiple(self):
        """Parallel query across multiple databases."""


class TestQueryRouter:
    """Tests for BAML query routing."""

    async def test_route_meeting_query(self):
        """Meeting query routes to meetings DB."""

    async def test_route_cross_domain(self):
        """Cross-domain query fans out to multiple DBs."""

    async def test_route_fallback_on_failure(self):
        """Router falls back to general on BAML failure."""

    async def test_route_disabled(self):
        """Disabled router always returns general."""

    async def test_route_ambiguous_query(self):
        """Vague query returns ambiguous=True with options."""

    async def test_route_ambiguous_with_context(self):
        """Vague query with good context routes correctly (not ambiguous)."""


class TestResultReranker:
    """Tests for post-retrieval re-ranking."""

    async def test_rerank_filters_irrelevant(self):
        """Irrelevant results removed after re-ranking."""

    async def test_rerank_preserves_relevant(self):
        """Highly relevant results kept and sorted."""

    async def test_rerank_skips_small_sets(self):
        """Re-ranking skipped for <= 3 results."""

    async def test_rerank_fallback_on_failure(self):
        """Returns unfiltered results if BAML call fails."""


class TestResultMerger:
    """Tests for multi-database result merging."""

    def test_merge_deduplicates_by_id(self):
        """Same entity from multiple DBs appears once."""

    def test_merge_annotates_source_db(self):
        """Each result has _source_db field."""

    def test_merge_sorts_by_similarity(self):
        """Results sorted by similarity score descending."""


class TestDisambiguation:
    """Tests for the disambiguation flow end-to-end."""

    async def test_ambiguous_returns_options(self):
        """Tool returns disambiguation JSON for vague queries."""

    async def test_explicit_database_bypasses_routing(self):
        """Explicit database param skips router entirely."""

    async def test_disambiguation_then_explicit(self):
        """After disambiguation, re-call with database= works."""


class TestProxyReferences:
    """Tests for cross-database proxy nodes."""

    async def test_create_proxy(self):
        """Proxy node created in general DB."""

    async def test_resolve_proxy(self):
        """Proxy resolves to data in vertical DB."""

    async def test_resolve_missing_graceful(self):
        """Missing vertical node returns resolved=False."""
```

#### 4. Docker Compose Memory Tuning
**File**: `docker-compose.yml`
**Changes**: Increase page cache for multi-database overhead

```yaml
environment:
  # Increased for 4 databases
  NEO4J_server_memory_pagecache_size: "3g"
deploy:
  resources:
    limits:
      memory: 10g
```

### Success Criteria:

#### Automated Verification:
- [ ] `baml-cli generate` succeeds with all new BAML files
- [ ] `baml test` passes for all routing and ontology tests
- [ ] `pytest tests/test_multi_db.py` passes
- [ ] Server starts with all 4 databases connected
- [ ] Existing tests continue to pass

#### Manual Verification:
- [ ] Full end-to-end: store meeting note → routed to meetings DB → proxy in general → search finds it
- [ ] Cross-domain query returns merged results from multiple DBs
- [ ] Neo4j Browser shows all databases with correct node types
- [ ] Performance acceptable (routing adds < 500ms latency)

---

## Testing Strategy

### Unit Tests:
- ClientRegistry: register, get, general, query_multiple
- QueryRouter: route_query, route_storage, fallback behavior
- ResultMerger: deduplication, annotation, sorting
- ProxyManager: create, resolve, error handling

### Integration Tests:
- Full pipeline: query → route → fan-out → merge → response
- Storage pipeline: content → route → store → proxy creation
- Cross-DB entity resolution

### Manual Testing Steps:
1. Start fresh: `docker compose down -v && docker compose up -d`
2. Verify databases: connect to :7474, check all 4 DBs exist
3. Store a meeting note via `memory_store` — verify it lands in meetings DB
4. Store a project update — verify it lands in projects DB
5. Search "standup yesterday" — verify routing hits meetings DB
6. Search "research from the planning meeting" — verify fan-out to meetings + research
7. Look up a person entity — verify cross-references to vertical DBs

## Performance Considerations

- **BAML routing adds latency**: ~200-500ms per LLM call. Consider caching recent routing decisions by query hash.
- **Re-ranking adds latency on fan-out**: ~300-500ms additional for the re-ranker LLM call. Only triggered on multi-DB fan-out queries with >3 results, so single-DB queries pay no penalty.
- **Disambiguation saves latency on vague queries**: Returning disambiguation options immediately is faster than querying all DBs and returning noisy results. Net positive for user experience.
- **Multiple MemoryClients**: Each holds a Neo4j connection pool. Default pool size of 100 × 4 databases = 400 connections. May need to reduce per-client pool size.
- **Page cache pressure**: 4 databases sharing page cache. Increased to 3GB but monitor hit ratios.
- **Startup time**: Creating 4 MemoryClients sequentially adds ~8-10s. Consider parallel initialization.
- **LLM cost per query**: Worst case (fan-out + rerank) = 3 LLM calls (route + query extraction + rerank). Best case (explicit DB) = 0 LLM calls for routing. Average case = 1 LLM call (route only). Disambiguation = 1 LLM call, no DB query.

## Migration Notes

- **Existing data untouched**: The general (`neo4j`) database keeps all current data
- **No data migration needed**: Verticals start empty, new data gets routed
- **Backward compatible**: `get_client(ctx)` still returns the general client
- **Gradual adoption**: Set `NAM_ROUTING_ENABLED=false` to disable routing entirely
- **Stale volume caveat**: Docker init script only runs on fresh volumes; lifespan fallback handles existing deployments

## References

- Architecture flowchart: `docs/multi-db-flow.mmd`
- Neo4j 5 multi-database docs: https://neo4j.com/docs/operations-manual/5/database-administration/
- BAML classification patterns: https://docs.boundaryml.com/examples/prompt-engineering/classification.mdx
- Prior gap analysis: `docs/2026-02-26-extraction-dedup-gap-analysis.md`
- Existing BAML extraction: `baml_src/extraction.baml`
