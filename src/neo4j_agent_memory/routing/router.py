"""BAML-powered query router for multi-database verticals."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Mapping from BAML enum values to Neo4j database names
VERTICAL_TO_DB: dict[str, str] = {
    "MEETINGS": "meetings",
    "PROJECTS": "projects",
    "RESEARCH": "research",
    "GENERAL": "neo4j",
}


class QueryRouter:
    """Routes queries and storage requests to appropriate database verticals."""

    def __init__(
        self,
        available_databases: list[str],
        confidence_threshold: float = 0.3,
    ) -> None:
        self._available_dbs = set(available_databases)
        self._threshold = confidence_threshold
        self._enabled = os.environ.get(
            "NAM_ROUTING_ENABLED", "true"
        ).lower() in ("true", "1", "yes")

    async def route_query(
        self,
        query: str,
        context: str | None = None,
    ) -> RoutingResult:
        """Route a search query to target database(s).

        Falls back to general DB if routing is disabled or fails.
        """
        if not self._enabled:
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

        try:
            from neo4j_agent_memory.baml_client.async_client import b

            decision = await b.RouteQuery(query=query, context=context)
            return self._to_result(decision)
        except Exception as e:
            logger.warning("Route failed, defaulting to general: %s", e)
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

    async def route_storage(
        self,
        content: str,
        memory_type: str,
        context: str | None = None,
    ) -> RoutingResult:
        """Route a storage request to the target database.

        Falls back to general DB if routing is disabled or fails.
        """
        if not self._enabled:
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

        try:
            from neo4j_agent_memory.baml_client.async_client import b

            decision = await b.RouteStorage(
                content=content,
                memory_type=memory_type,
                context=context,
            )
            return self._to_result(decision)
        except Exception as e:
            logger.warning("Storage route failed, defaulting to general: %s", e)
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

    def _to_result(self, decision) -> RoutingResult:
        """Convert BAML RoutingDecision to internal RoutingResult."""
        targets = []
        for target in decision.targets:
            db_name = VERTICAL_TO_DB.get(target.vertical.value, "neo4j")
            if db_name in self._available_dbs and target.confidence >= self._threshold:
                targets.append((db_name, target.confidence))

        # Ensure at least general DB
        if not targets:
            targets = [("neo4j", 1.0)]

        primary = VERTICAL_TO_DB.get(decision.primary_vertical.value, "neo4j")
        if primary not in self._available_dbs:
            primary = "neo4j"

        return RoutingResult(
            targets=targets,
            primary=primary,
            requires_fanout=decision.requires_fanout,
            ambiguous=decision.ambiguous,
            disambiguation_options=decision.disambiguation_options or [],
        )


class ResultReranker:
    """Re-ranks results from multi-database fan-out queries."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score and filter results by relevance to the original query.

        Args:
            query: The original user query.
            results: List of result dicts with id, content, _source_db, type.

        Returns:
            Filtered and re-ordered results.
        """
        if not self._enabled or not results:
            return results

        # Skip re-ranking for small result sets (not worth the LLM call)
        if len(results) <= 3:
            return results

        try:
            from neo4j_agent_memory.baml_client.async_client import b

            items = [
                {
                    "id": r.get("id", ""),
                    "content": r.get("content") or r.get("name") or r.get("task") or str(r),
                    "source_db": r.get("_source_db", "unknown"),
                    "result_type": r.get("_result_type", "unknown"),
                }
                for r in results
            ]

            reranked = await b.RerankResults(query=query, results=items)

            # Build lookup of kept results with their scores
            kept_ids = {}
            for scored in reranked.scored_results:
                if scored.keep:
                    kept_ids[scored.id] = scored.relevance

            # Filter and re-sort original results
            filtered = [r for r in results if r.get("id", "") in kept_ids]
            filtered.sort(
                key=lambda r: kept_ids.get(r.get("id", ""), 0),
                reverse=True,
            )

            logger.info(
                "Re-ranked %d results -> %d kept for query: %s",
                len(results), len(filtered), query[:80],
            )
            return filtered

        except Exception as e:
            logger.warning("Re-ranking failed, returning unfiltered: %s", e)
            return results


class RoutingResult:
    """Result of a routing decision."""

    def __init__(
        self,
        targets: list[tuple[str, float]],
        primary: str,
        requires_fanout: bool,
        ambiguous: bool = False,
        disambiguation_options: list[str] | None = None,
    ) -> None:
        self.targets = targets  # [(db_name, confidence), ...]
        self.primary = primary
        self.requires_fanout = requires_fanout
        self.ambiguous = ambiguous
        self.disambiguation_options = disambiguation_options or []

    @property
    def target_databases(self) -> list[str]:
        """List of database names to query, ordered by confidence."""
        return [db for db, _ in self.targets]

    @property
    def primary_database(self) -> str:
        """The single most relevant database."""
        return self.primary
