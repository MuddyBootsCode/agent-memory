"""Integration tests for multi-database vertical support."""

from __future__ import annotations

import asyncio
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


class TestVerticalExtraction:
    """Tests for vertical-specific entity extraction and persistence."""

    def test_extract_for_vertical_returns_none_for_general(self):
        """extract_for_vertical returns None for non-vertical databases."""
        import asyncio

        from neo4j_agent_memory.extraction.vertical_extractor import extract_for_vertical

        result = asyncio.get_event_loop().run_until_complete(
            extract_for_vertical("some text", "neo4j")
        )
        assert result is None

    def test_extract_for_vertical_returns_none_for_unknown(self):
        """extract_for_vertical returns None for unknown databases."""
        import asyncio

        from neo4j_agent_memory.extraction.vertical_extractor import extract_for_vertical

        result = asyncio.get_event_loop().run_until_complete(
            extract_for_vertical("some text", "unknown_db")
        )
        assert result is None

    def test_vertical_extractors_mapping(self):
        """VERTICAL_EXTRACTORS maps all three verticals."""
        from neo4j_agent_memory.extraction.vertical_extractor import VERTICAL_EXTRACTORS

        assert "meetings" in VERTICAL_EXTRACTORS
        assert "projects" in VERTICAL_EXTRACTORS
        assert "research" in VERTICAL_EXTRACTORS
        assert VERTICAL_EXTRACTORS["meetings"] == "ExtractMeetingEntities"
        assert VERTICAL_EXTRACTORS["projects"] == "ExtractProjectEntities"
        assert VERTICAL_EXTRACTORS["research"] == "ExtractResearchEntities"

    @pytest.mark.asyncio
    async def test_persist_vertical_entities_creates_nodes(self):
        """persist_vertical_entities writes Entity nodes to the graph."""
        from neo4j_agent_memory.extraction.vertical_extractor import persist_vertical_entities

        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(return_value=[{"e": {}}])
        mock_client.graph.execute_read = AsyncMock(return_value=[])
        mock_client.short_term = MagicMock()
        mock_client.short_term._embedder = None

        extraction = {
            "entities": [
                {"name": "Sprint Planning", "type": "MEETING", "confidence": 0.95, "status": "completed"},
                {"name": "Fix auth bug", "type": "ACTION_ITEM", "confidence": 0.88},
            ],
            "relations": [],
        }

        counts = await persist_vertical_entities(
            client=mock_client,
            message_id="msg-123",
            extraction=extraction,
            database="meetings",
        )

        assert counts["entities"] == 2
        assert counts["relations"] == 0

        # Verify execute_write was called for entity creation + message linking + domain props
        write_calls = mock_client.graph.execute_write.call_args_list
        assert len(write_calls) >= 4  # 2 entities * (create + link) + domain props

    @pytest.mark.asyncio
    async def test_persist_vertical_entities_creates_relations(self):
        """persist_vertical_entities creates RELATED_TO edges with domain types."""
        from neo4j_agent_memory.extraction.vertical_extractor import persist_vertical_entities

        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(return_value=[{"e": {}}])
        mock_client.graph.execute_read = AsyncMock(return_value=[])
        mock_client.short_term = MagicMock()
        mock_client.short_term._embedder = None

        extraction = {
            "entities": [
                {"name": "Alice", "type": "ATTENDEE", "confidence": 0.9},
                {"name": "Sprint Planning", "type": "MEETING", "confidence": 0.95},
            ],
            "relations": [
                {
                    "source": "Alice",
                    "target": "Sprint Planning",
                    "relation_type": "ATTENDED",
                    "confidence": 0.85,
                },
            ],
        }

        counts = await persist_vertical_entities(
            client=mock_client,
            message_id="msg-456",
            extraction=extraction,
            database="meetings",
        )

        assert counts["entities"] == 2
        assert counts["relations"] == 1

        # Find the relation write call — it should contain ATTENDED
        write_calls = mock_client.graph.execute_write.call_args_list
        relation_calls = [
            c for c in write_calls
            if "relation_type" in (c.args[1] if len(c.args) > 1 else c.kwargs.get("parameters", {}))
        ]
        assert len(relation_calls) == 1
        params = relation_calls[0].args[1] if len(relation_calls[0].args) > 1 else relation_calls[0].kwargs["parameters"]
        assert params["relation_type"] == "ATTENDED"

    @pytest.mark.asyncio
    async def test_persist_vertical_entities_handles_failures(self):
        """Entity persistence failures are logged but don't crash."""
        from neo4j_agent_memory.extraction.vertical_extractor import persist_vertical_entities

        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(side_effect=RuntimeError("DB down"))
        mock_client.graph.execute_read = AsyncMock(return_value=[])
        mock_client.short_term = MagicMock()
        mock_client.short_term._embedder = None

        extraction = {
            "entities": [
                {"name": "Sprint Planning", "type": "MEETING", "confidence": 0.9},
            ],
            "relations": [],
        }

        # Should not raise
        counts = await persist_vertical_entities(
            client=mock_client,
            message_id="msg-789",
            extraction=extraction,
            database="meetings",
        )

        assert counts["entities"] == 0
        assert counts["relations"] == 0

    @pytest.mark.asyncio
    async def test_persist_vertical_entities_sets_domain_properties(self):
        """Domain-specific fields (status, priority, date) are persisted."""
        from neo4j_agent_memory.extraction.vertical_extractor import persist_vertical_entities

        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(return_value=[{"e": {}}])
        mock_client.graph.execute_read = AsyncMock(return_value=[])
        mock_client.short_term = MagicMock()
        mock_client.short_term._embedder = None

        extraction = {
            "entities": [
                {
                    "name": "Auth Migration",
                    "type": "TASK",
                    "confidence": 0.9,
                    "status": "blocked",
                    "priority": "high",
                },
            ],
            "relations": [],
        }

        await persist_vertical_entities(
            client=mock_client,
            message_id="msg-100",
            extraction=extraction,
            database="projects",
        )

        # Find the domain-properties SET call
        write_calls = mock_client.graph.execute_write.call_args_list
        domain_calls = [
            c for c in write_calls
            if len(c.args) > 1 and "vertical_source" in (c.args[1] if isinstance(c.args[1], dict) else {})
        ]
        assert len(domain_calls) == 1
        params = domain_calls[0].args[1]
        assert params["status"] == "blocked"
        assert params["priority"] == "high"
        assert params["vertical_source"] == "projects"


class TestRoutingCache:
    """Tests for routing decision caching."""

    def _make_router(self, **env_overrides):
        """Create a router with caching enabled and routing enabled."""
        env = {
            "NAM_ROUTING_ENABLED": "true",
            "NAM_ROUTING_CACHE_ENABLED": "true",
            "NAM_ROUTING_CACHE_TTL": "300",
            "NAM_ROUTING_CACHE_SIZE": "64",
        }
        env.update(env_overrides)
        with patch.dict("os.environ", env):
            from neo4j_agent_memory.routing.router import QueryRouter

            return QueryRouter(
                available_databases=["neo4j", "meetings", "projects", "research"]
            )

    def _mock_decision(self, vertical="MEETINGS", confidence=0.95):
        """Build a mock BAML RoutingDecision."""
        target = MagicMock()
        target.vertical.value = vertical
        target.confidence = confidence
        target.reasoning = "test"

        decision = MagicMock()
        decision.targets = [target]
        decision.primary_vertical.value = vertical
        decision.requires_fanout = False
        decision.ambiguous = False
        decision.disambiguation_options = None
        return decision

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm_call(self):
        """Second identical query returns cached result without LLM call."""
        router = self._make_router()
        decision = self._mock_decision("MEETINGS")

        with patch(
            "neo4j_agent_memory.baml_client.async_client.b.RouteQuery",
            new_callable=AsyncMock,
            return_value=decision,
        ) as mock_route:
            result1 = await router.route_query("standup notes from yesterday")
            result2 = await router.route_query("standup notes from yesterday")

            assert mock_route.call_count == 1
            assert result1.primary == "meetings"
            assert result2.primary == "meetings"
            assert router.query_cache_stats.hits == 1
            assert router.query_cache_stats.misses == 1

    @pytest.mark.asyncio
    async def test_cache_miss_on_different_queries(self):
        """Different queries each trigger an LLM call."""
        router = self._make_router()
        meetings_decision = self._mock_decision("MEETINGS")
        projects_decision = self._mock_decision("PROJECTS")

        call_count = 0

        async def mock_route_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if "standup" in kwargs["query"]:
                return meetings_decision
            return projects_decision

        with patch(
            "neo4j_agent_memory.baml_client.async_client.b.RouteQuery",
            side_effect=mock_route_query,
        ):
            r1 = await router.route_query("standup notes")
            r2 = await router.route_query("project milestones")

            assert call_count == 2
            assert r1.primary == "meetings"
            assert r2.primary == "projects"
            assert router.query_cache_stats.misses == 2
            assert router.query_cache_stats.hits == 0

    @pytest.mark.asyncio
    async def test_storage_cache_hit(self):
        """Storage routing is also cached."""
        router = self._make_router()
        decision = self._mock_decision("PROJECTS")

        with patch(
            "neo4j_agent_memory.baml_client.async_client.b.RouteStorage",
            new_callable=AsyncMock,
            return_value=decision,
        ) as mock_route:
            r1 = await router.route_storage("task: fix auth bug", "fact")
            r2 = await router.route_storage("task: fix auth bug", "fact")

            assert mock_route.call_count == 1
            assert r1.primary == "projects"
            assert r2.primary == "projects"
            assert router.storage_cache_stats.hits == 1

    @pytest.mark.asyncio
    async def test_cache_disabled_via_env(self):
        """Cache can be disabled via env var."""
        router = self._make_router(NAM_ROUTING_CACHE_ENABLED="false")
        decision = self._mock_decision("MEETINGS")

        with patch(
            "neo4j_agent_memory.baml_client.async_client.b.RouteQuery",
            new_callable=AsyncMock,
            return_value=decision,
        ) as mock_route:
            await router.route_query("standup notes")
            await router.route_query("standup notes")

            assert mock_route.call_count == 2
            assert router.query_cache_stats.misses == 2

    @pytest.mark.asyncio
    async def test_stampede_prevention_coalesces_concurrent_requests(self):
        """Concurrent identical queries coalesce into one LLM call."""
        router = self._make_router()
        decision = self._mock_decision("MEETINGS")

        call_count = 0

        async def slow_route_query(**kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # simulate LLM latency
            return decision

        with patch(
            "neo4j_agent_memory.baml_client.async_client.b.RouteQuery",
            side_effect=slow_route_query,
        ):
            results = await asyncio.gather(
                router.route_query("standup notes"),
                router.route_query("standup notes"),
                router.route_query("standup notes"),
            )

            assert call_count == 1  # only one LLM call
            assert all(r.primary == "meetings" for r in results)
            assert router.query_cache_stats.coalesced == 2  # 2 coalesced
            assert router.query_cache_stats.misses == 1  # 1 actual miss

    @pytest.mark.asyncio
    async def test_cache_stats_to_dict(self):
        """CacheStats.to_dict produces correct hit rate."""
        from neo4j_agent_memory.routing.router import CacheStats

        stats = CacheStats()
        stats.hits = 3
        stats.misses = 7
        stats.coalesced = 2

        d = stats.to_dict()
        assert d["hits"] == 3
        assert d["misses"] == 7
        assert d["coalesced"] == 2
        assert d["hit_rate"] == 0.3

    @pytest.mark.asyncio
    async def test_normalize_key_handles_variations(self):
        """Key normalization collapses whitespace, strips punctuation, lowercases."""
        from neo4j_agent_memory.routing.router import _normalize_key

        k1 = _normalize_key("Standup Notes!")
        k2 = _normalize_key("standup  notes")
        k3 = _normalize_key("STANDUP NOTES??")
        assert k1 == k2 == k3

    @pytest.mark.asyncio
    async def test_fallback_on_error_still_caches_nothing(self):
        """When BAML call fails, fallback is returned but NOT cached."""
        router = self._make_router()

        with patch(
            "neo4j_agent_memory.baml_client.async_client.b.RouteQuery",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ) as mock_route:
            r1 = await router.route_query("standup notes")
            r2 = await router.route_query("standup notes")

            # Both calls hit the LLM (fallback not cached)
            assert mock_route.call_count == 2
            assert r1.primary == "neo4j"
            assert r2.primary == "neo4j"
