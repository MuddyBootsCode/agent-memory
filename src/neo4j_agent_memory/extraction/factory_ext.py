"""Extended factory that adds BAML extractor support.

Wraps the base package's create_extractor() and intercepts calls when
NAM_EXTRACTION__BAML_ENABLED=true. The BAML_ENABLED env var overrides
regardless of the configured extractor_type [RFI-F1].
"""

import logging
import os

from neo4j_agent_memory.extraction.base import EntityExtractor
from neo4j_agent_memory.extraction.factory import (
    create_extractor as _base_create_extractor,
)
from neo4j_agent_memory.extraction.baml_config import DEFAULT_BAML_CLIENT

logger = logging.getLogger(__name__)


def _is_baml_enabled() -> bool:
    """Check if BAML extraction is enabled via env var."""
    return os.environ.get(
        "NAM_EXTRACTION__BAML_ENABLED", ""
    ).lower() in ("true", "1", "yes")


def create_baml_extractor(
    extraction_config, schema_config=None
) -> EntityExtractor:
    """Create a BAML entity extractor from config."""
    from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

    client_name = os.environ.get(
        "NAM_EXTRACTION__BAML_CLIENT", DEFAULT_BAML_CLIENT
    )

    entity_types = extraction_config.entity_types
    if schema_config and hasattr(schema_config, "entity_types") and schema_config.entity_types:
        entity_types = schema_config.entity_types

    logger.info(
        "BAML extraction enabled (overriding extractor_type=%s, client=%s)",
        extraction_config.extractor_type,
        client_name,
    )

    return BamlEntityExtractor(
        client_name=client_name,
        entity_types=entity_types,
        extract_relations=extraction_config.extract_relations,
        extract_preferences=extraction_config.extract_preferences,
    )


def create_extractor(
    extraction_config, schema_config=None, llm_config=None
) -> EntityExtractor:
    """Extended factory — routes to BAML when enabled, else base factory."""
    if _is_baml_enabled():
        return create_baml_extractor(extraction_config, schema_config)

    return _base_create_extractor(extraction_config, schema_config, llm_config)
