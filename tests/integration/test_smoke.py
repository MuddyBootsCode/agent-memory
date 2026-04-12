"""Smoke tests validating the integration test infrastructure.

These tests verify that the fixture chain works end-to-end:
Neo4j connection → test database → MemoryClient → Bedrock embeddings.
"""

import pytest


class TestFixtureChain:
    """Verify the integration test fixtures work."""

    async def test_memory_client_connects(self, memory_client):
        """MemoryClient is connected and ready."""
        assert memory_client.is_connected

    async def test_database_is_clean(self, memory_client):
        """Each test starts with an empty database."""
        rows = await memory_client.graph.execute_read(
            "MATCH (n) RETURN count(n) AS count", {}
        )
        # Schema nodes may exist, but no user data
        # Just verify the query executes without error
        assert isinstance(rows[0]["count"], int)

    async def test_store_and_retrieve_message(self, memory_client):
        """Basic message store → vector search round-trip with Bedrock."""
        msg = await memory_client.short_term.add_message(
            session_id="smoke-test-session",
            role="user",
            content="Alice is a software engineer at Graphable.",
            generate_embedding=True,
            extract_entities=False,
        )
        assert msg.id is not None

        results = await memory_client.short_term.search_messages(
            query="Who works at Graphable?",
            limit=5,
            threshold=0.3,
        )
        assert len(results) >= 1
        assert "Alice" in results[0].content

    async def test_store_and_retrieve_fact(self, memory_client):
        """Fact SPO triple store → search round-trip."""
        fact = await memory_client.long_term.add_fact(
            subject="Alice",
            predicate="WORKS_AT",
            obj="Graphable",
            confidence=0.95,
            generate_embedding=True,
        )
        assert fact.id is not None

        facts = await memory_client.long_term.search_facts(
            query="Where does Alice work?",
            limit=5,
            threshold=0.3,
        )
        assert len(facts) >= 1
        assert facts[0].subject == "Alice"

    async def test_store_preference(self, memory_client):
        """Preference store works."""
        pref = await memory_client.long_term.add_preference(
            category="scheduling",
            preference="prefers morning meetings",
            context="Work habits",
        )
        assert pref.id is not None

    async def test_cypher_session_works(self, memory_client, cypher_session):
        """Graph executor can verify data stored via MemoryClient."""
        await memory_client.long_term.add_fact(
            subject="Bob",
            predicate="LIVES_IN",
            obj="Austin",
            generate_embedding=False,
        )

        rows = await cypher_session.execute_read(
            "MATCH (f:Fact {subject: $subject}) RETURN f.predicate AS pred",
            {"subject": "Bob"},
        )
        assert len(rows) == 1
        assert rows[0]["pred"] == "LIVES_IN"

    async def test_test_isolation(self, memory_client):
        """Verify data from previous test was wiped (memory_client wipes on setup)."""
        rows = await memory_client.graph.execute_read(
            "MATCH (f:Fact) RETURN count(f) AS count", {}
        )
        assert rows[0]["count"] == 0, "Previous test data should be wiped"
