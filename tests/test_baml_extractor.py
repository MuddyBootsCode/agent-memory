"""Tests for BamlEntityExtractor."""

import pytest
from unittest.mock import MagicMock

from neo4j_agent_memory.extraction.base import (
    EntityExtractor,
    ExtractionResult,
)


class TestBamlExtractorProtocol:
    """Verify BamlEntityExtractor satisfies EntityExtractor protocol."""

    def test_satisfies_protocol(self):
        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        assert isinstance(extractor, EntityExtractor)

    def test_has_name_property(self):
        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor(client_name="Anthropic")
        assert extractor.name == "BamlEntityExtractor(Anthropic)"


class TestBamlExtractorEmptyInput:
    """Verify empty/whitespace text returns empty result without calling BAML."""

    async def test_empty_string(self):
        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("")
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0

    async def test_whitespace_only(self):
        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("   \n  ")
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0


class TestBamlExtractorConversion:
    """Verify BAML types are correctly converted to base extraction types."""

    async def test_entity_conversion(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        # Set up mock BAML response
        entity = MagicMock()
        entity.name = "John Smith"
        entity.type = MagicMock()
        entity.type.value = "PERSON"
        entity.subtype = None
        entity.confidence = 0.95
        mock_result.entities = [entity]
        mock_result.relations = []
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("John Smith works at Acme Corp")

        assert len(result.entities) == 1
        assert result.entities[0].name == "John Smith"
        assert result.entities[0].type == "PERSON"
        assert result.entities[0].confidence == 0.95
        assert result.entities[0].extractor == "baml"

    async def test_relation_conversion(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        entity1 = MagicMock()
        entity1.name = "John"
        entity1.type = MagicMock(value="PERSON")
        entity1.subtype = None
        entity1.confidence = 0.9

        entity2 = MagicMock()
        entity2.name = "Acme"
        entity2.type = MagicMock(value="ORGANIZATION")
        entity2.subtype = None
        entity2.confidence = 0.85

        relation = MagicMock()
        relation.source = "John"
        relation.target = "Acme"
        relation.relation_type = "works_at"
        relation.confidence = 0.8

        mock_result.entities = [entity1, entity2]
        mock_result.relations = [relation]
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("John works at Acme")

        assert len(result.relations) == 1
        assert result.relations[0].source == "John"
        assert result.relations[0].target == "Acme"
        assert result.relations[0].relation_type == "WORKS_AT"

    async def test_relation_filtered_when_entity_missing(self, mock_baml_extract):
        """Relations referencing non-extracted entities are filtered out."""
        mock_fn, mock_result = mock_baml_extract

        entity = MagicMock()
        entity.name = "John"
        entity.type = MagicMock(value="PERSON")
        entity.subtype = None
        entity.confidence = 0.9

        relation = MagicMock()
        relation.source = "John"
        relation.target = "UnknownEntity"
        relation.relation_type = "knows"
        relation.confidence = 0.5

        mock_result.entities = [entity]
        mock_result.relations = [relation]
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("John knows someone")

        assert len(result.relations) == 0

    async def test_preference_conversion(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        pref = MagicMock()
        pref.category = "food"
        pref.preference = "likes pizza"
        pref.context = "for dinner"
        pref.confidence = 0.7

        mock_result.entities = []
        mock_result.relations = []
        mock_result.preferences = [pref]

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("I like pizza for dinner")

        assert len(result.preferences) == 1
        assert result.preferences[0].category == "food"
        assert result.preferences[0].preference == "likes pizza"

    async def test_confidence_clamped(self, mock_baml_extract):
        """Confidence values outside [0,1] are clamped."""
        mock_fn, mock_result = mock_baml_extract

        entity = MagicMock()
        entity.name = "Test"
        entity.type = MagicMock(value="PERSON")
        entity.subtype = None
        entity.confidence = 1.5  # Out of range

        mock_result.entities = [entity]
        mock_result.relations = []
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("Test entity")

        assert result.entities[0].confidence == 1.0


class TestBamlExtractorOptions:
    """Verify extract_relations and extract_preferences flags."""

    async def test_relations_disabled(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        entity = MagicMock()
        entity.name = "John"
        entity.type = MagicMock(value="PERSON")
        entity.subtype = None
        entity.confidence = 0.9

        relation = MagicMock()
        relation.source = "John"
        relation.target = "John"
        relation.relation_type = "self"
        relation.confidence = 0.5

        mock_result.entities = [entity]
        mock_result.relations = [relation]
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor(extract_relations=False)
        result = await extractor.extract("John")

        assert len(result.relations) == 0

    async def test_preferences_disabled(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        pref = MagicMock()
        pref.category = "food"
        pref.preference = "likes pizza"
        pref.context = None
        pref.confidence = 0.7

        mock_result.entities = []
        mock_result.relations = []
        mock_result.preferences = [pref]

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor(extract_preferences=False)
        result = await extractor.extract("I like pizza")

        assert len(result.preferences) == 0
