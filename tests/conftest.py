"""Shared test fixtures for BAML extraction tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_baml_extract(monkeypatch):
    """Patch BAML generated client to avoid live API calls [RFI-F4].

    Returns the mock function so tests can configure return values.
    """
    mock_result = MagicMock()
    mock_result.entities = []
    mock_result.relations = []
    mock_result.preferences = []

    mock_fn = AsyncMock(return_value=mock_result)
    monkeypatch.setattr(
        "neo4j_agent_memory.baml_client.async_client.b.ExtractEntities",
        mock_fn,
    )
    return mock_fn, mock_result
