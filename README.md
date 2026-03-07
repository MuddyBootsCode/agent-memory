# neo4j-agent-memory-mcp

Standalone MCP server for Neo4j Agent Memory with BAML entity extraction, multi-database verticals, and intelligent query routing.

## Features

- **Multi-database verticals** — Separate Neo4j databases for meetings, projects, and research with domain-specific ontologies
- **BAML query routing** — LLM-powered routing that directs queries to the correct vertical database
- **Cross-database fan-out** — Parallel querying across multiple databases with result merging and deduplication
- **Result re-ranking** — Post-retrieval relevance scoring to filter noise from multi-database results
- **Disambiguation** — Ambiguous queries return clarification options instead of noisy results
- **Proxy references** — Lightweight cross-database links so entities in one vertical can reference entities in another
- **BAML entity extraction** — Multi-provider LLM extraction with vertical-specific ontologies

## Architecture

```
                    +------------------+
                    |   MCP Tools      |
                    | (memory_search,  |
                    |  memory_store,   |
                    |  entity_lookup)  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  QueryRouter     |
                    |  (BAML-powered)  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v--+  +--------v--+  +--------v--+
     |  neo4j    |  | meetings  |  | projects  |
     | (general) |  | database  |  | database  |
     +-----------+  +-----------+  +-----------+
                                        |
                                +-------v------+
                                |  research    |
                                |  database    |
                                +--------------+
```

## Quick Start

### 1. Start Neo4j

```bash
docker compose up -d
```

This starts Neo4j 5 Enterprise and automatically creates the vertical databases (`meetings`, `projects`, `research`) on first run.

### 2. Configure Environment

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

```bash
# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphmemory

# Entity extraction
OPENAI_API_KEY=sk-your-key-here
NAM_EXTRACTION__BAML_ENABLED=true
NAM_EXTRACTION__BAML_CLIENT=Anthropic  # or OpenAI, Gemini, Resilient

# Multi-database verticals (comma-separated)
NAM_VERTICALS=meetings,projects,research

# Query routing
NAM_ROUTING_ENABLED=true
```

### 3. Run the Server

```bash
uv run python run_server.py
```

## Multi-Database Verticals

Each vertical has its own Neo4j database with a domain-specific ontology optimized for that type of data.

### Meetings

Entity types: `MEETING`, `ATTENDEE`, `AGENDA_ITEM`, `ACTION_ITEM`, `DECISION`

Relationships: `ATTENDED`, `PRESENTED`, `ASSIGNED_TO`, `DISCUSSED`, `DECIDED_IN`, `FOLLOWS_UP`, `SCHEDULED_BY`

### Projects

Entity types: `PROJECT`, `TASK`, `MILESTONE`, `DELIVERABLE`, `TEAM`

Relationships: `DEPENDS_ON`, `ASSIGNED_TO`, `BLOCKED_BY`, `DELIVERS`, `PART_OF`, `OWNS`, `REVIEWS`

### Research

Entity types: `NOTE`, `FINDING`, `SOURCE`, `TOPIC`, `EXPERIMENT`

Relationships: `CITES`, `SUPPORTS`, `CONTRADICTS`, `BUILDS_ON`, `TAGGED_WITH`, `AUTHORED_BY`, `EXPLORES`

### Configuration

Set `NAM_VERTICALS` to customize which verticals are created:

```bash
NAM_VERTICALS=meetings,projects,research    # default
NAM_VERTICALS=hr,finance,legal              # custom verticals
```

## Query Routing

When `NAM_ROUTING_ENABLED=true`, the server uses a BAML-powered classifier to route queries to the appropriate database(s).

- **Single-database queries** — "What was decided in Monday's standup?" routes to `meetings`
- **Fan-out queries** — "What's the status across all projects?" queries multiple databases in parallel
- **Ambiguous queries** — "Tell me about the API" returns disambiguation options instead of noisy cross-database results
- **Storage routing** — `memory_store` calls are routed to the correct vertical based on content analysis

When routing is disabled, all queries go to the general (`neo4j`) database.

### Re-ranking

Results from multi-database fan-out queries are scored for relevance (0-1) by a BAML function. Results below 0.4 are filtered out, reducing noise when querying across verticals.

## Tool Layer

All MCP tools accept an optional `database` parameter for explicit database targeting. Without it, the router decides where to send the query.

| Tool | Routing Behavior |
|------|-----------------|
| `memory_search` | Route query -> fan-out if needed -> merge -> re-rank |
| `memory_store` | Route by content type -> store in vertical -> create proxy ref in general |
| `entity_lookup` | Route to relevant DBs -> resolve cross-database proxy references |
| `conversation_history` | Uses specified or general database |
| `graph_query` | Uses specified or general database |
| `add_reasoning_trace` | Uses specified or general database |
| `explain_reasoning` | Uses specified or general database |
| `extract_reasoning` | Uses specified or general database |

## BAML Entity Extraction

Multi-provider LLM extraction using [BAML](https://docs.boundaryml.com/).

### Available Clients

| Client | Provider | Model | Description |
|--------|----------|-------|-------------|
| `OpenAI` | OpenAI | gpt-4o-mini | Default, fast and cheap |
| `Anthropic` | Anthropic | Claude Sonnet | High quality |
| `Gemini` | Google AI | Gemini 2.5 Flash | Google alternative |
| `Resilient` | Fallback | All three | Tries OpenAI -> Anthropic -> Gemini |

### Custom Clients

Edit `baml_src/clients.baml` to add or modify providers, then regenerate:

```bash
uv run baml-cli generate
```

> **Note:** The `version` field in `baml_src/generators.baml` must match the
> installed `baml-py` version. Check with `uv run python -c "import baml_py; print(baml_py.__version__)"`.

### Without BAML

When `NAM_EXTRACTION__BAML_ENABLED` is unset or `false`, the server uses the
default extraction pipeline from the base `neo4j-agent-memory` package (spaCy + GLiNER + LLM fallback).

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Regenerating BAML Client

After modifying any `.baml` files in `baml_src/`:

```bash
uv run baml-cli generate
```

### Project Structure

```
baml_src/
  clients.baml          # LLM provider configuration
  extraction.baml       # General entity extraction
  ontology_meetings.baml    # Meetings vertical ontology
  ontology_projects.baml    # Projects vertical ontology
  ontology_research.baml    # Research vertical ontology
  routing.baml          # Query routing classifier
  reranking.baml        # Result re-ranking function

src/neo4j_agent_memory/
  mcp/
    server.py           # MCP server with multi-DB lifespan
    _tools.py           # Tool definitions with routing
    _registry.py        # ClientRegistry for multi-DB management
    _database_init.py   # Vertical database creation
    _merge.py           # Cross-database result merging
    _proxy.py           # Cross-database proxy references
    _common.py          # Shared helpers
  routing/
    router.py           # QueryRouter and ResultReranker
  extraction/
    vertical_extractor.py   # Vertical-specific entity extraction
```
