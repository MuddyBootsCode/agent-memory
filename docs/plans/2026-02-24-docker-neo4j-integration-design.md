# Docker Neo4j Enterprise Integration Design

**Date**: 2026-02-24
**Status**: Approved
**Approach**: B — Separate Docker manager module integrated via FastMCP lifespan

## Problem

The MCP server requires a running Neo4j instance but has no way to ensure one is available. Users must manually start Neo4j before launching the server. The most common failure mode is the MCP server starting before Neo4j is ready, causing connection errors.

## Architecture

### New Module: `_docker.py`

A new file `src/neo4j_agent_memory/mcp/_docker.py` containing `Neo4jDockerManager`, an async context manager that manages a Docker container running Neo4j Enterprise with the APOC plugin.

**Responsibilities:**
- Detect if Neo4j is already reachable (TCP probe on bolt port)
- If not reachable, start/reuse a Docker container
- Wait for Neo4j to become fully ready
- Stop and remove the container on shutdown (only if we started it)

### Container Configuration

- **Image**: `neo4j:5-enterprise` (configurable)
- **Container name**: `neo4j-agent-memory` (deterministic, avoids duplicates)
- **Ports**: 7687 (bolt) and 7474 (http browser) mapped to host
- **Data volume**: Named volume `neo4j-agent-memory-data` at `/data`
- **Memory limit**: 8GB

**Container environment variables:**
```
NEO4J_AUTH=neo4j/<password>
NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
NEO4J_PLUGINS=["apoc"]
NEO4J_dbms_security_procedures_unrestricted=apoc.*
NEO4J_dbms_security_procedures_allowlist=apoc.*
NEO4J_server_memory_heap_initial__size=2g
NEO4J_server_memory_heap_max__size=4g
NEO4J_server_memory_pagecache_size=2g
NEO4J_dbms_memory_transaction_total_max=1g
```

**Memory breakdown (8GB total):**
- 4GB max heap — JVM heap for query execution
- 2GB page cache — graph data caching for read performance
- 1GB transaction memory — concurrent transaction overhead
- ~1GB — OS overhead, APOC plugin, JVM metaspace

### Integration Point

The FastMCP lifespan in `server.py` wraps `MemoryClient` with the Docker manager:

```python
async with Neo4jDockerManager(neo4j_config) as docker_mgr:
    # Retry loop for MemoryClient connection
    async with _MemoryClient(settings) as client:
        yield {"client": client}
```

## Two-Phase Resilient Startup

### Phase 1: Ensure Container Running (Neo4jDockerManager)

1. TCP probe the bolt URI
2. If reachable → no-op, skip Docker entirely
3. If not reachable → check for existing container named `neo4j-agent-memory`
   - Exists and stopped → start it
   - Exists and running → reuse, mark as externally managed (don't stop on exit)
   - Doesn't exist → create and start new container
4. Poll bolt port at 0.5s intervals, up to 60s timeout
5. On timeout → stop container, raise RuntimeError

### Phase 2: MemoryClient Retry Loop (in lifespan)

1. Attempt `MemoryClient(settings)` connection
2. On failure → retry up to 5 times, 2s apart
3. Each retry logs: "Neo4j not ready yet, retrying in 2s (attempt 3/5)"
4. All retries exhausted → raise with clear error message

**Why both phases:** Phase 1 catches "container not started." Phase 2 catches "bolt port open but Neo4j still warming up" — the race condition where Neo4j accepts TCP before it can process Cypher.

## Error Handling

### Docker not available (not installed / daemon not running)
- Catch `docker.errors.DockerException` on Docker client init
- Log warning: "Docker not available — assuming Neo4j is managed externally"
- Skip container management, let MemoryClient attempt direct connection

### Container start failures (image pull, port conflict)
- Catch `docker.errors.APIError`, `docker.errors.ImageNotFound`
- Log specific error with actionable guidance
- Raise `RuntimeError` wrapping the Docker error

### Health check timeout
- After 60s of polling with no bolt response → stop container, raise RuntimeError

### Shutdown errors
- Catch and log, but don't raise — shutdown errors shouldn't crash the server
- Best-effort cleanup

### Existing container in unexpected state
- Running: reuse, don't stop on exit
- Stopped: start, stop on exit
- Doesn't exist: create, stop on exit

## Configuration

### New CLI flags

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--neo4j-docker-auto` | `NEO4J_DOCKER_AUTO` | `true` | Enable/disable Docker management |
| `--neo4j-docker-image` | `NEO4J_DOCKER_IMAGE` | `neo4j:5-enterprise` | Docker image to use |
| `--neo4j-docker-startup-timeout` | `NEO4J_DOCKER_STARTUP_TIMEOUT` | `60` | Max seconds to wait for bolt |

### New dependency

`docker>=7.0.0` added to `pyproject.toml`

## File Changes

### New files
- `src/neo4j_agent_memory/mcp/_docker.py` — Neo4jDockerManager

### Modified files
- `src/neo4j_agent_memory/mcp/server.py` — Lifespan integration, retry loop, new CLI flags
- `pyproject.toml` — Add docker dependency

### Unchanged
- `_tools.py`, `_resources.py`, `_prompts.py`, `_common.py`
- `run_server.py`

## Zero-Overhead When Neo4j Already Running

If Neo4j is already reachable at the configured URI, the entire Docker code path is skipped. A single TCP probe adds negligible latency. The Docker SDK is only imported when needed.
