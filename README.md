# neo4j-agent-memory-mcp

Standalone MCP server for Neo4j Agent Memory with BAML entity extraction support.

## BAML Entity Extraction

Multi-provider LLM extraction using [BAML](https://docs.boundaryml.com/).

### Quick Start

Set environment variables and start the server:

```bash
export NAM_EXTRACTION__BAML_ENABLED=true
export NAM_EXTRACTION__BAML_CLIENT=OpenAI  # or Anthropic, Gemini, Resilient
export OPENAI_API_KEY=sk-...
```

### Available Clients

| Client | Provider | Model | Description |
|--------|----------|-------|-------------|
| `OpenAI` | OpenAI | gpt-4o-mini | Default, fast and cheap |
| `Anthropic` | Anthropic | Claude Sonnet | High quality |
| `Gemini` | Google AI | Gemini 2.5 Flash | Google alternative |
| `Resilient` | Fallback | All three | Tries OpenAI → Anthropic → Gemini |

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
