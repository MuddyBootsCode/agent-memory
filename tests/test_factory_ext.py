"""Tests for the extended factory with BAML support."""

import os
import pytest
from unittest.mock import patch


class TestFactoryExtBamlRouting:
    """Verify factory routes to BAML when BAML_ENABLED is set."""

    def test_baml_enabled_creates_baml_extractor(self):
        with patch.dict(os.environ, {
            "NAM_EXTRACTION__BAML_ENABLED": "true",
            "NAM_EXTRACTION__BAML_CLIENT": "OpenAI",
        }):
            from neo4j_agent_memory.extraction.factory_ext import create_extractor
            from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor
            from neo4j_agent_memory.config.settings import ExtractionConfig

            config = ExtractionConfig()
            extractor = create_extractor(config)
            assert isinstance(extractor, BamlEntityExtractor)

    def test_baml_disabled_falls_through(self):
        with patch.dict(os.environ, {
            "NAM_EXTRACTION__BAML_ENABLED": "false",
        }, clear=False):
            # Remove BAML_ENABLED if set
            os.environ.pop("NAM_EXTRACTION__BAML_ENABLED", None)

            from neo4j_agent_memory.extraction.factory_ext import create_extractor
            from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor
            from neo4j_agent_memory.config.settings import ExtractionConfig

            config = ExtractionConfig()
            extractor = create_extractor(config)
            assert not isinstance(extractor, BamlEntityExtractor)

    def test_baml_client_env_var_respected(self):
        with patch.dict(os.environ, {
            "NAM_EXTRACTION__BAML_ENABLED": "true",
            "NAM_EXTRACTION__BAML_CLIENT": "Anthropic",
        }):
            from neo4j_agent_memory.extraction.factory_ext import create_extractor
            from neo4j_agent_memory.config.settings import ExtractionConfig

            config = ExtractionConfig()
            extractor = create_extractor(config)
            assert "Anthropic" in extractor.name

    def test_baml_enabled_overrides_any_extractor_type(self):
        """BAML_ENABLED=true works regardless of extractor_type [RFI-F1]."""
        with patch.dict(os.environ, {
            "NAM_EXTRACTION__BAML_ENABLED": "true",
            "NAM_EXTRACTION__BAML_CLIENT": "OpenAI",
        }):
            from neo4j_agent_memory.extraction.factory_ext import create_extractor
            from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor
            from neo4j_agent_memory.config.settings import ExtractionConfig, ExtractorType

            # Even with NONE type, BAML overrides
            config = ExtractionConfig(extractor_type=ExtractorType.NONE)
            extractor = create_extractor(config)
            assert isinstance(extractor, BamlEntityExtractor)
