"""MCP tool implementations for Neo4j Agent Memory.

Defines the 8 core tools as FastMCP @mcp.tool decorated functions:
- memory_search: Hybrid vector + graph search
- memory_store: Store memories (messages, facts, preferences)
- entity_lookup: Get entity with relationships
- conversation_history: Get conversation for session
- graph_query: Execute read-only Cypher queries
- add_reasoning_trace: Store procedural memory with real reasoning
- explain_reasoning: Retrieve and explain past reasoning chains
- extract_reasoning: Extract reasoning from conversation text
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from fastmcp import Context

from neo4j_agent_memory.mcp._common import get_client, get_reranker, get_registry, get_router

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


async def _backfill_entity_embeddings(client: Any, message_id: str) -> int:
    """Generate embeddings for entities linked to a message that lack them.

    After add_message extracts entities, this finds any without embeddings
    and generates them using the client's embedder.

    Returns:
        Number of entities that received embeddings.
    """
    embedder = getattr(client.short_term, "_embedder", None)
    if embedder is None:
        logger.warning("No embedder available — skipping entity embedding backfill")
        return 0

    # Find entities linked to this message that have no embedding
    rows = await client.graph.execute_read(
        """
        MATCH (m:Message {id: $message_id})-[:MENTIONS]->(e:Entity)
        WHERE e.embedding IS NULL
        RETURN e.id AS id, e.name AS name
        """,
        {"message_id": message_id},
    )

    count = 0
    for row in rows:
        try:
            embedding = await embedder.embed(row["name"])
            if embedding:
                await client.graph.execute_write(
                    "MATCH (e:Entity {id: $id}) SET e.embedding = $embedding",
                    {"id": row["id"], "embedding": embedding},
                )
                count += 1
        except Exception as e:
            logger.warning("Failed to embed entity %s: %s", row["name"], e)

    if count > 0:
        logger.info("Backfilled embeddings for %d entities from message %s", count, message_id)
    return count


# Patterns for detecting write operations in Cypher (matched against uppercased query).
# Note: CALL db.* and CALL apoc.* are allowed since many procedures are read-only
# (e.g., db.index.vector.queryNodes, apoc.meta.data). The database itself will
# reject writes when executed via execute_read().
WRITE_PATTERNS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\s+DELETE\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bLOAD\s+CSV\b",
    r"\bFOREACH\b",
    r"\bCALL\s+\{",
    r"\bIN\s+TRANSACTIONS\b",
]


def _is_read_only_query(query: str) -> bool:
    """Check if a Cypher query is read-only.

    Args:
        query: The Cypher query to check.

    Returns:
        True if the query contains no write operations.
    """
    query_upper = query.upper()
    return all(not re.search(pattern, query_upper) for pattern in WRITE_PATTERNS)


def register_tools(mcp: FastMCP) -> None:
    """Register all MCP tools on the server.

    Args:
        mcp: FastMCP server instance.
    """

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
        Set database explicitly to target a specific vertical (meetings, projects,
        research, or neo4j for general).

        If the query is ambiguous, returns disambiguation options instead of results
        so the calling LLM can ask the user for clarification.

        Note: Entity relationships in the graph use a generic RELATED_TO edge type
        with the semantic relationship stored as a property. To find specific
        relationship types, use graph_query with:
            MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
            WHERE r.relation_type = "WORKS_AT"
            RETURN a.name, r.relation_type, b.name
        """
        from neo4j_agent_memory.mcp._merge import merge_search_results

        try:
            registry = get_registry(ctx)
        except RuntimeError:
            # Fall back to single-client mode
            registry = None

        router = get_router(ctx)

        # Step 1: Determine target databases
        route = None
        if database:
            target_dbs = [database]
        elif registry is None:
            # Single-client mode — no routing
            target_dbs = ["neo4j"]
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

        # Step 3: Search function to run per-database
        async def _search_db(client, db_name):
            results: dict[str, list[dict[str, Any]]] = {}

            if "messages" in memory_types:
                messages = await client.short_term.search_messages(
                    query=query,
                    session_id=session_id,
                    limit=limit,
                    threshold=threshold,
                )
                results["messages"] = [
                    {
                        "id": str(msg.id),
                        "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                        "similarity": msg.metadata.get("similarity") if msg.metadata else None,
                    }
                    for msg in messages
                ]

            if "entities" in memory_types:
                entities = await client.long_term.search_entities(
                    query=query,
                    limit=limit,
                )
                results["entities"] = [
                    {
                        "id": str(entity.id),
                        "name": entity.display_name,
                        "type": (
                            entity.type.value if hasattr(entity.type, "value") else str(entity.type)
                        ),
                        "description": entity.description,
                    }
                    for entity in entities
                ]

            if "preferences" in memory_types:
                preferences = await client.long_term.search_preferences(
                    query=query,
                    limit=limit,
                )
                results["preferences"] = [
                    {
                        "id": str(pref.id),
                        "category": pref.category,
                        "preference": pref.preference,
                        "context": pref.context,
                    }
                    for pref in preferences
                ]

            if "traces" in memory_types:
                traces = await client.reasoning.get_similar_traces(
                    task=query,
                    limit=limit,
                )
                results["traces"] = [
                    {
                        "id": str(trace.id),
                        "task": trace.task,
                        "outcome": trace.outcome,
                        "success": trace.success,
                        "session_id": trace.session_id,
                        "started_at": trace.started_at.isoformat() if trace.started_at else None,
                    }
                    for trace in traces
                ]

            return results

        try:
            # Fan out search to target databases
            if registry and len(target_dbs) > 0:
                per_db = await registry.query_multiple(target_dbs, _search_db)
                merged = merge_search_results(per_db)
            else:
                # Single-client fallback
                client = get_client(ctx)
                merged = await _search_db(client, "neo4j")

            # Step 4: Re-rank results if this was a fan-out query
            reranked = False
            if route and route.requires_fanout and len(target_dbs) > 1:
                reranker = get_reranker(ctx)
                all_results = []
                for memory_type, items in merged.items():
                    for item in items:
                        item["_result_type"] = memory_type
                    all_results.extend(items)

                reranked_results = await reranker.rerank(query, all_results)

                # Rebuild merged dict from reranked results
                merged = {}
                for item in reranked_results:
                    rt = item.pop("_result_type", "unknown")
                    if rt not in merged:
                        merged[rt] = []
                    merged[rt].append(item)
                reranked = True

        except Exception as e:
            logger.error(f"Error in memory_search: {e}")
            return json.dumps({"error": str(e)})

        response = {
            "results": merged,
            "query": query,
            "databases_searched": target_dbs,
            "reranked": reranked,
        }
        if route:
            response["routing"] = route.to_metadata()

        return json.dumps(response, default=str)

    @mcp.tool()
    async def memory_store(
        ctx: Context,
        memory_type: str,
        content: str,
        session_id: str | None = None,
        role: str = "user",
        category: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        metadata: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> str:
        """Store a memory in the knowledge graph.

        Automatically routes to the most relevant database vertical using AI
        classification. Set database explicitly to target a specific vertical
        (meetings, projects, research, or neo4j for general).

        Supports messages, facts (SPO triples), and user preferences.
        Automatically extracts entities from message content.

        Args:
            memory_type: Type of memory - 'message', 'fact', or 'preference'.
            content: The content to store.
            session_id: Session ID (required for message type).
            role: Message role - 'user', 'assistant', or 'system' (default: 'user').
            category: Preference category (required for preference type).
            subject: Fact subject (required for fact type).
            predicate: Fact predicate/relationship (required for fact type).
            object_value: Fact object (required for fact type).
            metadata: Optional metadata to attach.
            database: Target database vertical. If None, auto-routed by AI.
        """
        router = get_router(ctx)

        # Route to target database
        route = None
        if database:
            target_db = database
        else:
            route = await router.route_storage(content, memory_type)
            target_db = route.primary_database

        # Get the appropriate client
        try:
            registry = get_registry(ctx)
            client = registry.get(target_db)
        except (RuntimeError, KeyError):
            client = get_client(ctx)
            target_db = "neo4j"

        try:
            result_data: dict[str, Any] | None = None
            stored_id: str | None = None
            stored_name: str | None = None

            if memory_type == "message":
                if not session_id:
                    return json.dumps({"error": "session_id required for message storage"})

                message = await client.short_term.add_message(
                    session_id=session_id,
                    role=role,
                    content=content,
                    metadata=metadata,
                    extract_entities=True,
                    generate_embedding=True,
                )

                # Backfill embeddings for extracted entities
                embedded_count = await _backfill_entity_embeddings(
                    client, str(message.id)
                )

                # Run vertical-specific extraction for domain databases
                vertical_counts = None
                if target_db != "neo4j":
                    try:
                        from neo4j_agent_memory.extraction.vertical_extractor import (
                            extract_for_vertical,
                            persist_vertical_entities,
                        )

                        vertical_extraction = await extract_for_vertical(content, target_db)
                        if vertical_extraction:
                            vertical_counts = await persist_vertical_entities(
                                client=client,
                                message_id=str(message.id),
                                extraction=vertical_extraction,
                                database=target_db,
                            )
                    except Exception as vert_err:
                        logger.warning(
                            "Vertical extraction failed for %s: %s", target_db, vert_err
                        )

                stored_id = str(message.id)
                stored_name = content[:80]
                result_data = {
                    "stored": True,
                    "type": "message",
                    "id": stored_id,
                    "session_id": session_id,
                    "entities_embedded": embedded_count,
                    "database": target_db,
                }

                if vertical_counts:
                    result_data["vertical_entities"] = vertical_counts["entities"]
                    result_data["vertical_relations"] = vertical_counts["relations"]
                    result_data["vertical_ontology"] = target_db

            elif memory_type == "preference":
                if not category:
                    return json.dumps({"error": "category required for preference storage"})

                preference = await client.long_term.add_preference(
                    category=category,
                    preference=content,
                    generate_embedding=True,
                )
                stored_id = str(preference.id)
                stored_name = f"{category}: {content[:60]}"
                result_data = {
                    "stored": True,
                    "type": "preference",
                    "id": stored_id,
                    "category": category,
                    "database": target_db,
                }

            elif memory_type == "fact":
                if not all([subject, predicate, object_value]):
                    return json.dumps(
                        {"error": "subject, predicate, and object_value required for fact storage"}
                    )

                fact = await client.long_term.add_fact(
                    subject=subject,
                    predicate=predicate,
                    obj=object_value,
                )
                stored_id = str(fact.id) if hasattr(fact, "id") else None
                stored_name = f"{subject} -> {predicate} -> {object_value}"
                result_data = {
                    "stored": True,
                    "type": "fact",
                    "id": stored_id,
                    "triple": stored_name,
                    "database": target_db,
                }

            else:
                return json.dumps({"error": f"Unknown memory type: {memory_type}"})

            # Create proxy reference in general DB for vertical storage
            if target_db != "neo4j" and stored_id and result_data:
                try:
                    from neo4j_agent_memory.mcp._proxy import create_proxy_reference

                    general_client = registry.general
                    proxy_id = await create_proxy_reference(
                        general_client=general_client,
                        source_db=target_db,
                        node_id=stored_id,
                        node_type=memory_type,
                        node_name=stored_name or content[:80],
                    )
                    result_data["proxy_id"] = proxy_id
                except Exception as proxy_err:
                    logger.warning("Proxy creation failed: %s", proxy_err)

            if route and result_data:
                result_data["routing"] = route.to_metadata()

            return json.dumps(result_data)

        except Exception as e:
            logger.error(f"Error in memory_store: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def entity_lookup(
        ctx: Context,
        name: str,
        entity_type: str | None = None,
        include_neighbors: bool = True,
        max_hops: int = 1,
        database: str | None = None,
    ) -> str:
        """Look up an entity and retrieve its relationships and neighbors.

        Searches the knowledge graph for entities by name, with optional
        graph traversal to find related entities. Neighbors include the
        semantic relationship type (e.g., WORKS_AT, CREATES, REQUIRES)
        and direction of the connection.

        Automatically searches across relevant database verticals.
        Set database explicitly to search a specific vertical.
        """
        # Determine which databases to search
        route = None
        try:
            registry = get_registry(ctx)
            if database:
                target_dbs = [database]
            else:
                router = get_router(ctx)
                route = await router.route_query(f"entity: {name}")
                target_dbs = route.target_databases
        except RuntimeError:
            registry = None
            target_dbs = ["neo4j"]

        client = get_client(ctx) if registry is None else registry.get(target_dbs[0])

        try:
            # Try vector search first (requires entity embeddings)
            entities = await client.long_term.search_entities(
                query=name,
                entity_types=[entity_type] if entity_type else None,
                limit=1,
            )

            # Fallback to Cypher name match if vector search returns nothing
            if not entities:
                type_filter = ""
                params: dict[str, Any] = {"name": name}
                if entity_type:
                    type_filter = f" AND e:{entity_type}"

                cypher_results = await client.graph.execute_read(
                    f"""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name){type_filter}
                    RETURN e.id AS id, e.name AS name, e.type AS type,
                           e.description AS description, labels(e) AS labels
                    LIMIT 1
                    """,
                    params,
                )

                if not cypher_results:
                    return json.dumps({"found": False, "name": name})

                match = cypher_results[0]
                result: dict[str, Any] = {
                    "found": True,
                    "match_method": "name",
                    "entity": {
                        "id": match["id"],
                        "name": match["name"],
                        "type": match["type"],
                        "description": match["description"],
                        "labels": match["labels"],
                    },
                }

                if include_neighbors:
                    neighbors = await _get_entity_neighbors(
                        client, match["id"], max_hops
                    )
                    result["neighbors"] = neighbors

                # Resolve cross-database proxy references
                if registry:
                    try:
                        from neo4j_agent_memory.mcp._proxy import resolve_proxy_references
                        cross_refs = await resolve_proxy_references(
                            general_client=registry.general,
                            registry=registry,
                            entity_id=match["id"],
                        )
                        if cross_refs:
                            result["cross_references"] = cross_refs
                    except Exception as proxy_err:
                        logger.debug("Proxy resolution skipped: %s", proxy_err)

                if route:
                    result["routing"] = route.to_metadata()

                return json.dumps(result, default=str)

            entity = entities[0]
            result = {
                "found": True,
                "match_method": "vector",
                "entity": {
                    "id": str(entity.id),
                    "name": entity.display_name,
                    "type": (
                        entity.type.value if hasattr(entity.type, "value") else str(entity.type)
                    ),
                    "description": entity.description,
                    "aliases": entity.aliases if hasattr(entity, "aliases") else [],
                },
            }

            if include_neighbors:
                neighbors = await _get_entity_neighbors(client, str(entity.id), max_hops)
                result["neighbors"] = neighbors

            # Resolve cross-database proxy references
            if registry:
                try:
                    from neo4j_agent_memory.mcp._proxy import resolve_proxy_references
                    cross_refs = await resolve_proxy_references(
                        general_client=registry.general,
                        registry=registry,
                        entity_id=str(entity.id),
                    )
                    if cross_refs:
                        result["cross_references"] = cross_refs
                except Exception as proxy_err:
                    logger.debug("Proxy resolution skipped: %s", proxy_err)

            if route:
                result["routing"] = route.to_metadata()

            return json.dumps(result, default=str)

        except Exception as e:
            logger.error(f"Error in entity_lookup: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def conversation_history(
        ctx: Context,
        session_id: str,
        limit: int = 50,
        before: str | None = None,
        include_metadata: bool = True,
        database: str | None = None,
    ) -> str:
        """Retrieve conversation history for a session.

        Returns messages in chronological order with role, content, and metadata.
        Set database to target a specific vertical.
        """
        try:
            registry = get_registry(ctx)
            client = registry.get(database) if database else registry.general
        except (RuntimeError, KeyError):
            client = get_client(ctx)

        try:
            conversation = await client.short_term.get_conversation(
                session_id=session_id,
                limit=limit,
            )

            messages = []
            for msg in conversation.messages:
                msg_data: dict[str, Any] = {
                    "id": str(msg.id),
                    "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                }
                if include_metadata and msg.metadata:
                    msg_data["metadata"] = msg.metadata
                messages.append(msg_data)

            return json.dumps(
                {
                    "session_id": session_id,
                    "message_count": len(messages),
                    "messages": messages,
                },
                default=str,
            )

        except Exception as e:
            logger.error(f"Error in conversation_history: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def graph_query(
        ctx: Context,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> str:
        """Execute a read-only Cypher query against the knowledge graph.

        Only MATCH/RETURN queries are allowed. Write operations
        (CREATE, MERGE, DELETE, SET, REMOVE) are blocked for safety.

        Set database to target a specific vertical (meetings, projects,
        research, or neo4j for general).

        Schema notes for writing effective queries:
        - Entity nodes have labels like :Entity:Person:Individual, :Entity:Organization:Company,
          :Entity:Object:Product, :Entity:Location:City, :Entity:Event
        - Entity relationships all use the edge type :RELATED_TO with the semantic
          relationship stored in the `relation_type` property (e.g., "WORKS_AT", "CREATES",
          "REQUIRES", "LIVES_IN", "OWNS"). Filter with: WHERE r.relation_type = "WORKS_AT"
        - Messages link to entities via :MENTIONS relationships
        - Conversations link to messages via :HAS_MESSAGE, :FIRST_MESSAGE, :NEXT_MESSAGE
        - Facts are stored as :Fact nodes with subject, predicate, object properties
        - Preferences are :Preference nodes with category and preference properties
        - ReasoningTrace nodes link to :ReasoningStep via :HAS_STEP, steps link to :ToolCall
        """
        if not _is_read_only_query(query):
            return json.dumps(
                {
                    "error": "Only read-only queries are allowed. "
                    "Write operations (CREATE, MERGE, DELETE, SET, REMOVE) are not permitted."
                }
            )

        try:
            registry = get_registry(ctx)
            client = registry.get(database) if database else registry.general
        except (RuntimeError, KeyError):
            client = get_client(ctx)

        try:
            records = await client.graph.execute_read(query, parameters or {})
            return json.dumps(
                {
                    "success": True,
                    "row_count": len(records),
                    "rows": records,
                },
                default=str,
            )

        except Exception as e:
            logger.error(f"Error in graph_query: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def add_reasoning_trace(
        ctx: Context,
        session_id: str,
        task: str,
        steps: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        outcome: str | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> str:
        """Store a reasoning trace capturing HOW and WHY a task was solved.

        Records the task, reasoning steps with thought/action/observation,
        tool calls, and final outcome. Enables later retrieval via
        explain_reasoning to understand the agent's decision process.

        Set database to target a specific vertical.

        Args:
            session_id: Session ID for the reasoning trace.
            task: Description of the task being solved.
            steps: List of reasoning steps. Each step should include:
                - thought: WHY this action was chosen (hypothesis, reasoning)
                - tool_name: What tool/action was taken
                - observation: WHAT was learned from the result
                - alternatives_considered: Other approaches weighed (optional)
                - arguments: Tool arguments (optional)
                - result: Raw result (optional)
                - success: Whether this step succeeded (optional)
            tool_calls: Deprecated alias for steps (backward compatibility).
            outcome: Final outcome or result of the task.
            success: Whether the task was completed successfully.
            metadata: Optional metadata (model, latency, etc.).
            database: Target database vertical. If None, uses general.
        """
        from neo4j_agent_memory.memory.reasoning import ToolCallStatus

        try:
            registry = get_registry(ctx)
            client = registry.get(database) if database else registry.general
        except (RuntimeError, KeyError):
            client = get_client(ctx)
        # Use steps if provided, fall back to tool_calls for backward compat
        step_list = steps or tool_calls or []

        try:
            trace = await client.reasoning.start_trace(
                session_id=session_id,
                task=task,
                metadata=metadata or {},
            )

            for tc in step_list:
                tool_name = tc.get("tool_name", "unknown")

                # Use caller-provided thought, or generate a default
                thought = tc.get("thought") or f"Calling {tool_name}"

                # Use caller-provided observation, or fall back to raw result
                observation = tc.get("observation") or tc.get("result")
                if observation and not isinstance(observation, str):
                    observation = str(observation)

                # Include alternatives in thought if provided
                alternatives = tc.get("alternatives_considered")
                if alternatives:
                    thought = f"{thought}\n\nAlternatives considered: {alternatives}"

                step = await client.reasoning.add_step(
                    trace_id=trace.id,
                    thought=thought,
                    action=tool_name,
                    observation=observation,
                )

                # Map success field to ToolCallStatus
                tc_success = tc.get("success")
                if tc_success is False:
                    status = ToolCallStatus.FAILURE
                elif tc_success is True:
                    status = ToolCallStatus.SUCCESS
                else:
                    status = ToolCallStatus.SUCCESS

                await client.reasoning.record_tool_call(
                    step_id=step.id,
                    tool_name=tool_name,
                    arguments=tc.get("arguments", {}),
                    result=tc.get("result"),
                    status=status,
                )

            await client.reasoning.complete_trace(
                trace_id=trace.id,
                outcome=outcome,
                success=success,
            )

            return json.dumps({
                "success": True,
                "stored": True,
                "trace_id": str(trace.id),
                "session_id": session_id,
                "task": task,
                "step_count": len(step_list),
            })

        except Exception as e:
            logger.error(f"Error in add_reasoning_trace: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def explain_reasoning(
        ctx: Context,
        trace_id: str | None = None,
        query: str | None = None,
        session_id: str | None = None,
        synthesize: bool = False,
        database: str | None = None,
    ) -> str:
        """Retrieve and explain the reasoning behind a past decision or answer.

        Use this to answer "Why did you come up with this answer?" by retrieving
        the full reasoning chain (thought -> action -> observation for each step).

        Provide ONE of:
        - trace_id: Get reasoning for a specific trace
        - query: Find the most relevant trace by semantic similarity
        - session_id: Get the most recent trace from a session

        Set synthesize=true for a natural-language explanation powered by LLM.
        Default returns the raw reasoning chain.
        Set database to search a specific vertical.

        Args:
            trace_id: Specific trace ID to explain.
            query: Semantic search query to find relevant trace.
            session_id: Session ID to get most recent trace from.
            synthesize: If true, produce LLM-synthesized explanation (slower).
            database: Target database vertical. If None, uses general.
        """
        try:
            registry = get_registry(ctx)
            client = registry.get(database) if database else registry.general
        except (RuntimeError, KeyError):
            client = get_client(ctx)

        try:
            trace = None

            # Route 1: Direct trace lookup
            if trace_id:
                trace = await client.reasoning.get_trace_with_steps(trace_id)
                if not trace:
                    return json.dumps({"error": f"Trace not found: {trace_id}"})

            # Route 2: Semantic search for most relevant trace
            elif query:
                similar = await client.reasoning.get_similar_traces(
                    task=query, limit=1, success_only=False,
                )
                if not similar:
                    return json.dumps({
                        "found": False,
                        "message": "No reasoning traces found matching query",
                        "query": query,
                    })
                trace = await client.reasoning.get_trace_with_steps(similar[0].id)

            # Route 3: Most recent trace from session
            elif session_id:
                traces = await client.reasoning.list_traces(
                    session_id=session_id, limit=1,
                )
                if not traces:
                    return json.dumps({
                        "found": False,
                        "message": f"No traces found for session: {session_id}",
                    })
                trace = await client.reasoning.get_trace_with_steps(traces[0].id)

            else:
                return json.dumps({
                    "error": "Provide trace_id, query, or session_id"
                })

            if not trace:
                return json.dumps({"error": "Failed to retrieve trace details"})

            # Build the reasoning chain
            chain = {
                "trace_id": str(trace.id),
                "task": trace.task,
                "session_id": trace.session_id,
                "success": trace.success,
                "outcome": trace.outcome,
                "started_at": trace.started_at.isoformat() if trace.started_at else None,
                "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
                "steps": [
                    {
                        "step_number": step.step_number,
                        "thought": step.thought,
                        "action": step.action,
                        "observation": step.observation,
                        "tool_calls": [
                            {
                                "tool_name": tc.tool_name,
                                "arguments": tc.arguments,
                                "status": tc.status.value if hasattr(tc.status, "value") else str(tc.status),
                            }
                            for tc in (step.tool_calls or [])
                        ],
                    }
                    for step in (trace.steps or [])
                ],
            }

            # Optional: synthesize a natural-language explanation
            if synthesize and trace.steps:
                try:
                    from neo4j_agent_memory.extraction.reasoning_extractor import (
                        BamlReasoningExtractor,
                    )

                    extractor = BamlReasoningExtractor()
                    explanation = await extractor.synthesize_explanation(
                        task=trace.task,
                        steps=[
                            {
                                "thought": s.thought or "",
                                "action": s.action or "",
                                "observation": s.observation or "",
                            }
                            for s in trace.steps
                        ],
                        outcome=trace.outcome or "",
                    )
                    chain["synthesized_explanation"] = explanation
                except Exception as synth_err:
                    logger.warning(f"Synthesis failed, returning raw chain: {synth_err}")
                    chain["synthesis_error"] = str(synth_err)

            return json.dumps(chain, default=str)

        except Exception as e:
            logger.error(f"Error in explain_reasoning: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def extract_reasoning(
        ctx: Context,
        text: str,
        session_id: str,
        store: bool = True,
        database: str | None = None,
    ) -> str:
        """Extract structured reasoning from conversation text using LLM analysis.

        Analyzes a conversation transcript to identify the reasoning chain:
        hypotheses formed, evidence gathered, decisions made, and conclusions drawn.

        Optionally stores the extracted reasoning as a trace in memory.
        Set database to target a specific vertical.

        Args:
            text: Conversation transcript or text to analyze.
            session_id: Session ID to associate with the extracted trace.
            store: If true (default), store extracted reasoning as a trace.
            database: Target database vertical. If None, uses general.
        """
        try:
            registry = get_registry(ctx)
            client = registry.get(database) if database else registry.general
        except (RuntimeError, KeyError):
            client = get_client(ctx)

        try:
            from neo4j_agent_memory.extraction.reasoning_extractor import (
                BamlReasoningExtractor,
            )

            extractor = BamlReasoningExtractor()
            extracted = await extractor.extract_reasoning(text)

            result = {
                "extracted": True,
                "task": extracted["task"],
                "step_count": len(extracted["steps"]),
                "conclusion": extracted["final_conclusion"],
                "success": extracted["success"],
                "steps": extracted["steps"],
            }

            # Store as a reasoning trace if requested
            if store and extracted["steps"]:
                trace = await client.reasoning.start_trace(
                    session_id=session_id,
                    task=extracted["task"],
                    metadata={"source": "conversation_extraction"},
                )

                for ext_step in extracted["steps"]:
                    await client.reasoning.add_step(
                        trace_id=trace.id,
                        thought=ext_step["thought"],
                        action=ext_step["action"],
                        observation=ext_step["observation"],
                    )

                await client.reasoning.complete_trace(
                    trace_id=trace.id,
                    outcome=extracted["final_conclusion"],
                    success=extracted["success"],
                )

                result["stored"] = True
                result["trace_id"] = str(trace.id)

            return json.dumps(result, default=str)

        except Exception as e:
            logger.error(f"Error in extract_reasoning: {e}")
            return json.dumps({"error": str(e)})


async def _get_entity_neighbors(
    client,
    entity_id: str,
    max_hops: int = 1,
) -> list[dict[str, Any]]:
    """Get neighboring entities via graph traversal with relationship details.

    Args:
        client: MemoryClient instance.
        entity_id: Starting entity ID.
        max_hops: Maximum relationship depth (clamped to 1-3).

    Returns:
        List of neighboring entities with relationship type and direction.
    """
    max_hops = min(max(max_hops, 1), 3)
    query = f"""
    MATCH (e:Entity {{id: $entity_id}})-[r:RELATED_TO]-(neighbor:Entity)
    WHERE neighbor.id <> $entity_id
    WITH DISTINCT neighbor, r,
         CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END AS direction
    RETURN neighbor.id AS id,
           neighbor.name AS name,
           neighbor.type AS type,
           neighbor.description AS description,
           coalesce(r.relation_type, r.type, 'RELATED_TO') AS relation_type,
           r.confidence AS confidence,
           direction
    ORDER BY r.confidence DESC
    LIMIT 20
    """

    try:
        records = await client.graph.execute_read(
            query,
            {"entity_id": entity_id},
        )
        neighbors = [
            {
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "relationship": r["relation_type"],
                "direction": r["direction"],
                "confidence": r["confidence"],
            }
            for r in records
        ]

        # For multi-hop, also get second-degree connections
        if max_hops >= 2:
            query_2hop = """
            MATCH (e:Entity {id: $entity_id})-[:RELATED_TO]-(:Entity)
                  -[r2:RELATED_TO]-(hop2:Entity)
            WHERE hop2.id <> $entity_id
              AND NOT (e)-[:RELATED_TO]-(hop2)
            WITH DISTINCT hop2, r2,
                 CASE WHEN startNode(r2) = hop2 THEN 'incoming' ELSE 'outgoing' END AS direction
            RETURN hop2.id AS id,
                   hop2.name AS name,
                   hop2.type AS type,
                   hop2.description AS description,
                   coalesce(r2.relation_type, r2.type, 'RELATED_TO') AS relation_type,
                   r2.confidence AS confidence,
                   direction
            LIMIT 10
            """
            try:
                hop2_records = await client.graph.execute_read(
                    query_2hop,
                    {"entity_id": entity_id},
                )
                for r in hop2_records:
                    neighbors.append(
                        {
                            "id": r["id"],
                            "name": r["name"],
                            "type": r["type"],
                            "description": r["description"],
                            "relationship": r["relation_type"],
                            "direction": r["direction"],
                            "confidence": r["confidence"],
                            "hop": 2,
                        }
                    )
            except Exception:
                pass  # Second hop is best-effort

        return neighbors
    except Exception as e:
        logger.debug(f"Error getting neighbors: {e}")
        return []
