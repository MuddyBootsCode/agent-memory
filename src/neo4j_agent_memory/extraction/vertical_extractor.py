"""Vertical-aware entity extraction dispatcher and persistence."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

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

        entities = []
        for e in result.entities:
            entity_dict: dict[str, Any] = {
                "name": e.name,
                "type": e.type.value if hasattr(e.type, "value") else str(e.type),
                "confidence": e.confidence,
            }
            # Capture domain-specific fields when present
            if hasattr(e, "date") and e.date:
                entity_dict["date"] = e.date
            if hasattr(e, "status") and e.status:
                entity_dict["status"] = e.status
            if hasattr(e, "priority") and e.priority:
                entity_dict["priority"] = e.priority
            entities.append(entity_dict)

        return {
            "entities": entities,
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


async def persist_vertical_entities(
    client: Any,
    message_id: str,
    extraction: dict[str, Any],
    database: str,
) -> dict[str, int]:
    """Persist vertical-extracted entities and relations to the graph.

    Creates Entity nodes with domain-specific types (e.g., ACTION_ITEM,
    MILESTONE) and links them to the source message via :MENTIONS edges.
    Relations use :RELATED_TO edges with vertical-specific relation_type
    properties (e.g., ASSIGNED_TO, DEPENDS_ON).

    Args:
        client: MemoryClient for the target vertical database.
        message_id: ID of the message these entities were extracted from.
        extraction: Dict from extract_for_vertical with 'entities' and 'relations'.
        database: Name of the vertical database (for metadata).

    Returns:
        Dict with 'entities' and 'relations' counts.
    """
    from neo4j_agent_memory.graph.query_builder import build_create_entity_query

    entities_stored = 0
    relations_stored = 0
    entity_name_to_id: dict[str, str] = {}

    for entity in extraction.get("entities", []):
        entity_id = str(uuid4())
        entity_type = entity["type"]
        entity_name = entity["name"]

        try:
            create_query = build_create_entity_query(entity_type, None)
            await client.graph.execute_write(
                create_query,
                {
                    "id": entity_id,
                    "name": entity_name,
                    "type": entity_type,
                    "subtype": None,
                    "canonical_name": entity_name,
                    "description": None,
                    "embedding": None,
                    "confidence": max(0.0, min(1.0, entity.get("confidence", 0.8))),
                    "metadata": None,
                    "location": None,
                },
            )

            # Set domain-specific properties
            domain_props = {}
            if entity.get("status"):
                domain_props["status"] = entity["status"]
            if entity.get("priority"):
                domain_props["priority"] = entity["priority"]
            if entity.get("date"):
                domain_props["date"] = entity["date"]
            domain_props["vertical_source"] = database

            if domain_props:
                set_clauses = ", ".join(
                    f"e.{k} = ${k}" for k in domain_props
                )
                await client.graph.execute_write(
                    f"MATCH (e:Entity {{name: $name, type: $type}}) SET {set_clauses}",
                    {"name": entity_name, "type": entity_type, **domain_props},
                )

            # Link to source message
            await client.graph.execute_write(
                """
                MATCH (m:Message {id: $message_id})
                MATCH (e:Entity {name: $name, type: $type})
                MERGE (m)-[r:MENTIONS]->(e)
                ON CREATE SET r.confidence = $confidence
                """,
                {
                    "message_id": message_id,
                    "name": entity_name,
                    "type": entity_type,
                    "confidence": entity.get("confidence", 0.8),
                },
            )

            entity_name_to_id[entity_name.lower().strip()] = entity_id
            entities_stored += 1
        except Exception as e:
            logger.warning(
                "Failed to persist vertical entity '%s': %s", entity_name, e
            )

    # Persist relations
    for relation in extraction.get("relations", []):
        source_name = relation["source"]
        target_name = relation["target"]
        relation_type = relation["relation_type"]
        confidence = max(0.0, min(1.0, relation.get("confidence", 0.7)))

        # Try ID-based linking first, fall back to name-based
        source_id = entity_name_to_id.get(source_name.lower().strip())
        target_id = entity_name_to_id.get(target_name.lower().strip())

        try:
            if source_id and target_id:
                await client.graph.execute_write(
                    """
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    MERGE (source)-[r:RELATED_TO]->(target)
                    ON CREATE SET
                        r.relation_type = $relation_type,
                        r.confidence = $confidence,
                        r.created_at = datetime()
                    ON MATCH SET
                        r.confidence = CASE WHEN $confidence > r.confidence
                            THEN $confidence ELSE r.confidence END,
                        r.updated_at = datetime()
                    """,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": relation_type,
                        "confidence": confidence,
                    },
                )
            else:
                await client.graph.execute_write(
                    """
                    MATCH (source:Entity)
                    WHERE toLower(source.name) = toLower($source_name)
                    WITH source LIMIT 1
                    MATCH (target:Entity)
                    WHERE toLower(target.name) = toLower($target_name)
                    WITH source, target LIMIT 1
                    MERGE (source)-[r:RELATED_TO]->(target)
                    ON CREATE SET
                        r.relation_type = $relation_type,
                        r.confidence = $confidence,
                        r.created_at = datetime()
                    ON MATCH SET
                        r.confidence = CASE WHEN $confidence > r.confidence
                            THEN $confidence ELSE r.confidence END,
                        r.updated_at = datetime()
                    """,
                    {
                        "source_name": source_name,
                        "target_name": target_name,
                        "relation_type": relation_type,
                        "confidence": confidence,
                    },
                )
            relations_stored += 1
        except Exception as e:
            logger.warning(
                "Failed to persist vertical relation '%s'-[%s]->'%s': %s",
                source_name, relation_type, target_name, e,
            )

    # Backfill embeddings for vertical entities that lack them
    try:
        embedder = getattr(client.short_term, "_embedder", None)
        if embedder and entities_stored > 0:
            rows = await client.graph.execute_read(
                """
                MATCH (m:Message {id: $message_id})-[:MENTIONS]->(e:Entity)
                WHERE e.embedding IS NULL AND e.vertical_source = $database
                RETURN e.id AS id, e.name AS name
                """,
                {"message_id": message_id, "database": database},
            )
            for row in rows:
                try:
                    embedding = await embedder.embed(row["name"])
                    if embedding:
                        await client.graph.execute_write(
                            "MATCH (e:Entity {id: $id}) SET e.embedding = $embedding",
                            {"id": row["id"], "embedding": embedding},
                        )
                except Exception:
                    pass  # Embedding is best-effort
    except Exception:
        pass  # Embedding backfill is best-effort

    logger.info(
        "Vertical extraction for '%s': %d entities, %d relations persisted",
        database, entities_stored, relations_stored,
    )

    return {"entities": entities_stored, "relations": relations_stored}
