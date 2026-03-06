"""Integration tests for multi-database vertical support."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo4j_agent_memory.mcp._registry import ClientRegistry


class TestClientRegistry:
    """Tests for ClientRegistry."""

    def test_register_and_get(self):
        """Registry stores and retrieves clients by name."""
        registry = ClientRegistry()
        mock_client = MagicMock()
        registry.register("meetings", mock_client)
        assert registry.get("meetings") is mock_client

    def test_get_missing_raises(self):
        """Accessing unregistered database raises KeyError."""
        registry = ClientRegistry()
        with pytest.raises(KeyError, match="No client registered"):
            registry.get("nonexistent")

    def test_general_property(self):
        """General property returns the neo4j client."""
        registry = ClientRegistry()
        mock_client = MagicMock()
        registry.register("neo4j", mock_client)
        assert registry.general is mock_client

    def test_general_property_missing_raises(self):
        """General property raises when no neo4j client registered."""
        registry = ClientRegistry()
        registry.register("meetings", MagicMock())
        with pytest.raises(RuntimeError, match="No general database"):
            _ = registry.general

    def test_databases_list(self):
        """Databases property returns all registered names."""
        registry = ClientRegistry()
        registry.register("neo4j", MagicMock())
        registry.register("meetings", MagicMock())
        registry.register("projects", MagicMock())
        assert set(registry.databases) == {"neo4j", "meetings", "projects"}

    @pytest.mark.asyncio
    async def test_query_multiple(self):
        """Parallel query across multiple databases."""
        registry = ClientRegistry()
        mock_neo4j = MagicMock()
        mock_meetings = MagicMock()
        registry.register("neo4j", mock_neo4j)
        registry.register("meetings", mock_meetings)

        async def query_fn(client, db_name):
            return {"db": db_name, "count": 5}

        results = await registry.query_multiple(
            ["neo4j", "meetings"], query_fn
        )
        assert results["neo4j"]["db"] == "neo4j"
        assert results["meetings"]["db"] == "meetings"

    @pytest.mark.asyncio
    async def test_query_multiple_handles_errors(self):
        """Failed query returns error dict instead of raising."""
        registry = ClientRegistry()
        registry.register("neo4j", MagicMock())

        async def failing_fn(client, db_name):
            raise ValueError("Connection lost")

        results = await registry.query_multiple(["neo4j"], failing_fn)
        assert "error" in results["neo4j"]

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Close all cleans up context managers."""
        registry = ClientRegistry()
        mock_cm = AsyncMock()
        registry.register("neo4j", MagicMock(), mock_cm)
        registry.register("meetings", MagicMock(), AsyncMock())

        await registry.close_all()
        assert len(registry.databases) == 0
        mock_cm.__aexit__.assert_called_once()


class TestQueryRouter:
    """Tests for BAML query routing."""

    @pytest.mark.asyncio
    async def test_route_disabled(self):
        """Disabled router always returns general."""
        from neo4j_agent_memory.routing.router import QueryRouter

        with patch.dict("os.environ", {"NAM_ROUTING_ENABLED": "false"}):
            router = QueryRouter(
                available_databases=["neo4j", "meetings", "projects"]
            )
            result = await router.route_query("standup notes")
            assert result.primary == "neo4j"
            assert result.target_databases == ["neo4j"]
            assert not result.requires_fanout

    @pytest.mark.asyncio
    async def test_route_fallback_on_failure(self):
        """Router falls back to general on BAML failure."""
        from neo4j_agent_memory.routing.router import QueryRouter

        with patch.dict("os.environ", {"NAM_ROUTING_ENABLED": "true"}):
            router = QueryRouter(
                available_databases=["neo4j", "meetings"]
            )
            # BAML client import will fail in test env
            result = await router.route_query("standup notes")
            assert result.primary == "neo4j"

    @pytest.mark.asyncio
    async def test_storage_route_disabled(self):
        """Disabled router routes storage to general."""
        from neo4j_agent_memory.routing.router import QueryRouter

        with patch.dict("os.environ", {"NAM_ROUTING_ENABLED": "false"}):
            router = QueryRouter(available_databases=["neo4j"])
            result = await router.route_storage("meeting notes", "message")
            assert result.primary_database == "neo4j"


class TestResultReranker:
    """Tests for post-retrieval re-ranking."""

    @pytest.mark.asyncio
    async def test_rerank_skips_small_sets(self):
        """Re-ranking skipped for <= 3 results."""
        from neo4j_agent_memory.routing.router import ResultReranker

        reranker = ResultReranker(enabled=True)
        results = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        output = await reranker.rerank("query", results)
        assert output == results  # Unchanged

    @pytest.mark.asyncio
    async def test_rerank_disabled(self):
        """Disabled reranker returns input unchanged."""
        from neo4j_agent_memory.routing.router import ResultReranker

        reranker = ResultReranker(enabled=False)
        results = [{"id": str(i)} for i in range(10)]
        output = await reranker.rerank("query", results)
        assert output == results

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        """Empty results returned as-is."""
        from neo4j_agent_memory.routing.router import ResultReranker

        reranker = ResultReranker(enabled=True)
        output = await reranker.rerank("query", [])
        assert output == []

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_failure(self):
        """Returns unfiltered results if BAML call fails."""
        from neo4j_agent_memory.routing.router import ResultReranker

        reranker = ResultReranker(enabled=True)
        results = [{"id": str(i)} for i in range(5)]
        # BAML client import will fail in test env
        output = await reranker.rerank("query", results)
        assert output == results


class TestResultMerger:
    """Tests for multi-database result merging."""

    def test_merge_deduplicates_by_id(self):
        """Same entity from multiple DBs appears once."""
        from neo4j_agent_memory.mcp._merge import merge_search_results

        per_db = {
            "neo4j": {"entities": [{"id": "e1", "name": "Alice"}]},
            "meetings": {"entities": [{"id": "e1", "name": "Alice"}]},
        }
        merged = merge_search_results(per_db)
        assert len(merged["entities"]) == 1

    def test_merge_annotates_source_db(self):
        """Each result has _source_db field."""
        from neo4j_agent_memory.mcp._merge import merge_search_results

        per_db = {
            "meetings": {"messages": [{"id": "m1", "content": "standup"}]},
        }
        merged = merge_search_results(per_db)
        assert merged["messages"][0]["_source_db"] == "meetings"

    def test_merge_sorts_by_similarity(self):
        """Results sorted by similarity score descending."""
        from neo4j_agent_memory.mcp._merge import merge_search_results

        per_db = {
            "neo4j": {
                "messages": [
                    {"id": "m1", "similarity": 0.5},
                    {"id": "m2", "similarity": 0.9},
                ]
            },
        }
        merged = merge_search_results(per_db)
        assert merged["messages"][0]["id"] == "m2"

    def test_merge_handles_errors(self):
        """Error results from a DB are skipped."""
        from neo4j_agent_memory.mcp._merge import merge_search_results

        per_db = {
            "neo4j": {"messages": [{"id": "m1"}]},
            "meetings": {"error": "Connection lost"},
        }
        merged = merge_search_results(per_db)
        assert len(merged["messages"]) == 1

    def test_merge_entity_results(self):
        """Entity results merged across databases."""
        from neo4j_agent_memory.mcp._merge import merge_entity_results

        per_db = {
            "neo4j": {
                "found": True,
                "entity": {"id": "e1", "name": "Alice"},
                "neighbors": [{"id": "e2", "name": "Bob"}],
            },
            "meetings": {
                "found": True,
                "entity": {"id": "e1-meet", "name": "Alice"},
                "neighbors": [],
            },
        }
        merged = merge_entity_results(per_db)
        assert merged["found"]
        assert len(merged["entities"]) == 2
        assert merged["entities"][0]["_source_db"] == "neo4j"


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_get_configured_verticals_default(self):
        """Default verticals returned when env not set."""
        from neo4j_agent_memory.mcp._database_init import get_configured_verticals

        with patch.dict("os.environ", {}, clear=True):
            verticals = get_configured_verticals()
            assert verticals == ["meetings", "projects", "research"]

    def test_get_configured_verticals_custom(self):
        """Custom verticals parsed from env."""
        from neo4j_agent_memory.mcp._database_init import get_configured_verticals

        with patch.dict("os.environ", {"NAM_VERTICALS": "hr, finance, legal"}):
            verticals = get_configured_verticals()
            assert verticals == ["hr", "finance", "legal"]


class TestRoutingResult:
    """Tests for RoutingResult data class."""

    def test_target_databases(self):
        """target_databases returns db names from targets."""
        from neo4j_agent_memory.routing.router import RoutingResult

        result = RoutingResult(
            targets=[("meetings", 0.9), ("projects", 0.6)],
            primary="meetings",
            requires_fanout=True,
        )
        assert result.target_databases == ["meetings", "projects"]

    def test_primary_database(self):
        """primary_database returns primary."""
        from neo4j_agent_memory.routing.router import RoutingResult

        result = RoutingResult(
            targets=[("neo4j", 1.0)],
            primary="neo4j",
            requires_fanout=False,
        )
        assert result.primary_database == "neo4j"

    def test_disambiguation(self):
        """Ambiguous result has options."""
        from neo4j_agent_memory.routing.router import RoutingResult

        result = RoutingResult(
            targets=[("neo4j", 0.5)],
            primary="neo4j",
            requires_fanout=False,
            ambiguous=True,
            disambiguation_options=[
                "Search meetings for standup discussions",
                "Search projects for status updates",
            ],
        )
        assert result.ambiguous
        assert len(result.disambiguation_options) == 2
