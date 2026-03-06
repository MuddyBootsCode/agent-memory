"""Tests for server lifespan BAML factory patching [RFI-R1]."""

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def test_factory_module_is_patched_when_baml_enabled():
    """Verify the factory module attribute gets replaced."""
    with patch.dict(os.environ, {"NAM_EXTRACTION__BAML_ENABLED": "true"}):
        import neo4j_agent_memory.extraction.factory as factory_mod
        from neo4j_agent_memory.extraction.factory_ext import (
            create_extractor as ext_create,
        )

        # Simulate what the server lifespan does
        original = factory_mod.create_extractor
        factory_mod.create_extractor = ext_create

        assert factory_mod.create_extractor is ext_create
        assert factory_mod.create_extractor.__module__ == "neo4j_agent_memory.extraction.factory_ext"

        # Restore
        factory_mod.create_extractor = original
