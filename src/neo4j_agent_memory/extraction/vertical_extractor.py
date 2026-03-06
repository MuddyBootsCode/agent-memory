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
