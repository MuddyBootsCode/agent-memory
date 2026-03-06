---
date: 2026-02-26T18:52:00Z
researcher: Claude Code (Opus 4.6)
repository: neo4j-agent-memory-mcp
topic: "Gap analysis: extraction pipeline and entity deduplication vs base neo4j-agent-memory library"
tags: [research, extraction, deduplication, baml, pipeline, entity-resolution, gap-analysis]
status: complete
last_updated: 2026-02-26
last_updated_by: Claude Code (Opus 4.6)
---

# Gap Analysis: Extraction & Deduplication vs Base Library

**Date**: 2026-02-26
**Repository**: neo4j-agent-memory-mcp (overlay)
**Base Library**: neo4j-agent-memory v0.0.3 (neo4j-labs/agent-memory)

## Research Question

What would it take to make our overlay project match the base `neo4j-agent-memory` library in entity extraction and deduplication, while keeping BAML where it makes sense?

## Executive Summary

Our overlay currently uses BAML as a **complete replacement** for the base extraction pipeline. This means we bypass the multi-stage pipeline, merge strategies, and entity resolution entirely. The recommended approach is to **integrate BAML as a pipeline stage** (not a replacement) and **expose entity resolution/deduplication via new MCP tools**.

### Score Card

| Capability | Base Library | Our Overlay | Gap |
|---|---|---|---|
| Multi-stage extraction pipeline | spaCy → GLiNER → LLM | Single BAML call | **Large** |
| Merge strategies (5 types) | UNION, INTERSECTION, CONFIDENCE, CASCADE, FIRST_SUCCESS | None | **Large** |
| ExtractorBuilder fluent API | Full (`.with_spacy()`, `.with_gliner()`, etc.) | No `.with_baml()` | **Large** |
| Entity resolution (dedup) | ExactMatch → FuzzyMatch → SemanticMatch | Not exposed | **Large** |
| Streaming extraction | StreamingExtractor for long docs | Not used | Medium |
| ConditionalPipeline | Stage activation based on text properties | Not used | Medium |
| Enrichment | Wikipedia, Diffbot, Background service | Not used | Medium |
| Entity type model (POLE+O) | Full support | Full support | **None** |
| Protocol compliance | EntityExtractor protocol | BamlEntityExtractor satisfies it | **None** |
| Extraction result model | ExtractedEntity/Relation/Preference | Same types used | **None** |

---

## Detailed Findings

### 1. Current State of Our Overlay

**How BAML extraction works today** (`factory_ext.py`):

```
factory_ext.create_extractor(config)
  → if NAM_EXTRACTION__BAML_ENABLED=true:
      → BamlEntityExtractor(client_name, entity_types)  # single LLM call
  → else:
      → base create_extractor()  # spaCy/GLiNER/LLM/pipeline
```

**Problems with this approach:**

1. **Binary switch**: BAML is ON or the base pipeline is ON — never both
2. **No pipeline participation**: `BamlEntityExtractor` is returned as a standalone extractor, not wrapped in `ExtractionPipeline`
3. **No merge strategies**: When BAML is active, there's no multi-stage merging
4. **No builder integration**: `ExtractorBuilder` has no `.with_baml()` method
5. **Monkey-patch gap**: `factory_ext.create_extractor()` only intercepts when explicitly imported from `factory_ext` — the monkey-patch in `server.py:63-68` patches `neo4j_agent_memory.extraction.create_extractor` but callers importing from `factory` directly still get the base function
6. **No entity resolution**: We never call `CompositeResolver` or expose dedup tools

### 2. Base Library's Extraction Pipeline

**3-stage architecture** (cost/quality ladder):

| Stage | Speed | Cost | Quality | Library |
|---|---|---|---|---|
| spaCy | ~5ms | Free (local) | Good for common NER | en_core_web_sm |
| GLiNER2 | ~50ms | Free (local) | Better, domain-aware | GLiNER model |
| LLM | ~500ms | API cost | Best, contextual | OpenAI gpt-4o-mini |

**Pipeline orchestration** (`ExtractionPipeline`):
- Runs stages sequentially
- Each stage produces `ExtractionResult` with entities, relations, preferences
- Results are merged using one of 5 strategies
- `stop_on_success` flag allows early exit (used with FIRST_SUCCESS)
- `fallback_on_error` catches stage exceptions and continues
- Per-stage timing tracked in `StageResult` objects

**Merge strategies**:

| Strategy | When to Use |
|---|---|
| **CONFIDENCE** (default) | Keep highest-confidence entity per normalized_name::type key |
| **UNION** | Combine all unique entities across stages, prefer higher confidence on collision |
| **INTERSECTION** | Only keep entities found by 2+ stages (boost confidence by 1.1x) |
| **CASCADE** | First stage is base, later stages only add new entities |
| **FIRST_SUCCESS** | Return results from first stage that finds enough entities |

**ExtractorBuilder** fluent API:

```python
extractor = (
    ExtractorBuilder()
    .with_spacy()                    # Stage 1: fast local NER
    .with_gliner("urchade/gliner_multi-v2.1")  # Stage 2: domain-aware
    .with_llm_fallback("gpt-4o-mini")  # Stage 3: LLM when others miss
    .merge_by_confidence()            # Keep best per entity
    .extract_relations()              # Also extract relationships
    .extract_preferences()            # Also extract preferences
    .build()
)
```

### 3. Base Library's Entity Resolution (Deduplication)

**3-strategy chain** (`CompositeResolver`):

| Strategy | Mechanism | Threshold | Library |
|---|---|---|---|
| ExactMatch | Normalized string equality | 1.0 | Built-in |
| FuzzyMatch | Token-sort ratio | 0.85 | RapidFuzz |
| SemanticMatch | Embedding cosine similarity | 0.80 | Embedder interface |

**Key behaviors:**
- **Type-strict by default**: Never merges entities across types (PERSON vs ORGANIZATION)
- **Cascade resolution**: Tries exact first, stops on first match
- **Batch resolution**: Union-Find clustering for bulk operations
- **Auto-merge threshold**: 0.95 confidence → automatic merge
- **Flag threshold**: 0.85 confidence → creates `SAME_AS` relationship for human review
- Canonical name selection prefers longer, more specific names

**MemoryClient dedup methods** (currently unused by our MCP tools):
- `find_duplicate_entities()` → returns list of `(entity, match, score)` tuples
- `merge_entities(source_id, target_id)` → merges nodes + relationships
- `auto_deduplicate()` → batch find + auto-merge above threshold
- `get_dedup_candidates()` → returns candidates above flag threshold
- `get_entity_resolution_stats()` → metrics on resolution quality

### 4. Base Library's Enrichment Pipeline

**Providers:**
- `WikimediaProvider` — free, rate-limited, good for public figures/orgs/locations
- `DiffbotProvider` — API key required, structured entity data
- `CachedEnrichmentProvider` — wraps any provider with Neo4j-based cache
- `CompositeEnrichmentProvider` — chains providers with fallback

**Background service:**
- `BackgroundEnrichmentService` — async priority queue
- Automatically enriches newly stored entities
- Configurable concurrency and rate limiting

---

## Gap Closure Plan

### Phase 1: Integrate BAML as a Pipeline Stage (not a replacement)

**Goal**: Make BAML participate in `ExtractionPipeline` alongside spaCy/GLiNER.

**Changes needed:**

1. **Make `BamlEntityExtractor` satisfy `ExtractionStage` protocol**
   - Already has `name` property and `extract()` method
   - Just needs to be wrapped via `ExtractorStage` or implement `ExtractionStage` directly
   - File: `baml_extractor.py` — add `ExtractionStage` protocol compatibility

2. **Add `with_baml()` to `ExtractorBuilder`**
   - Create `ExtractorBuilderExt` that wraps or extends `ExtractorBuilder`
   - New method: `.with_baml(client_name="OpenAI", entity_types=None)`
   - This adds BAML as a pipeline stage, not a replacement
   - File: new `builder_ext.py` in overlay extraction/

3. **Update `factory_ext.create_extractor()`**
   - When `NAM_EXTRACTION__BAML_ENABLED=true`, instead of returning a bare `BamlEntityExtractor`, build a pipeline:
     ```python
     pipeline = (
         ExtractorBuilderExt()
         .with_spacy()           # fast, free first pass
         .with_baml("OpenAI")    # BAML as LLM stage (replaces base LLM)
         .merge_by_confidence()
         .extract_relations()
         .build()
     )
     ```
   - Support `NAM_EXTRACTION__BAML_PIPELINE_MODE` env var: `"baml_only"` (current behavior), `"hybrid"` (spaCy+BAML), `"full"` (spaCy+GLiNER+BAML)
   - File: `factory_ext.py`

4. **Where BAML makes most sense in the pipeline**:
   - **Replace the base LLM stage** (gpt-4o-mini via OpenAI SDK) with BAML
   - BAML provides type-safe extraction, multi-provider support (OpenAI/Anthropic/Gemini), and the Resilient fallback client
   - spaCy and GLiNER should run first as cheap/fast stages — BAML is the quality backstop
   - Merge strategy: `CONFIDENCE` or `CASCADE` (spaCy base, BAML fills gaps)

**Estimated scope**: ~200 lines of new code, ~50 lines modified

### Phase 2: Enable Entity Resolution/Deduplication

**Goal**: Expose the base library's `CompositeResolver` and dedup methods through MCP tools.

**Changes needed:**

1. **New MCP tool: `deduplicate_entities`**
   ```python
   @mcp.tool()
   async def deduplicate_entities(
       ctx: Context,
       auto_merge_threshold: float = 0.95,
       flag_threshold: float = 0.85,
       entity_type: str | None = None,
       dry_run: bool = True,
   ) -> str:
       """Find and optionally merge duplicate entities."""
   ```
   - `dry_run=True` by default — shows candidates without merging
   - Filter by entity type for targeted cleanup
   - Returns merge candidates with scores
   - File: `_tools.py`

2. **New MCP tool: `merge_entities`**
   ```python
   @mcp.tool()
   async def merge_entities(
       ctx: Context,
       source_entity: str,
       target_entity: str,
   ) -> str:
       """Merge two entities, combining all relationships."""
   ```
   - Delegates to `MemoryClient.merge_entities()`
   - File: `_tools.py`

3. **New MCP tool: `dedup_stats`**
   ```python
   @mcp.tool()
   async def dedup_stats(ctx: Context) -> str:
       """Get entity resolution statistics."""
   ```
   - Delegates to `MemoryClient.get_entity_resolution_stats()`
   - File: `_tools.py`

4. **Enable resolution in `memory_store`**
   - When storing entities (via `memory_store` type=`fact`), run `CompositeResolver` to check for existing matches before creating new nodes
   - Configuration: `NAM_RESOLUTION__ENABLED=true`, `NAM_RESOLUTION__AUTO_MERGE_THRESHOLD=0.95`
   - File: `_tools.py` (modify `memory_store` tool)

**Estimated scope**: ~150 lines of new tools, ~30 lines modifying existing tools

### Phase 3: Expose Additional Base Library Features

**Priority features to expose as MCP tools (ordered by value):**

| Feature | Base Method | New MCP Tool | Priority |
|---|---|---|---|
| Similar traces | `get_similar_traces()` | `find_similar_reasoning` | High |
| Trace provenance | `get_trace_provenance()` | Part of `explain_reasoning` | High |
| Session management | `delete_session()` | `manage_session` | Medium |
| Streaming trace | `StreamingTraceRecorder` | N/A (requires streaming MCP) | Medium |
| Geospatial search | `search_by_location()` | `search_nearby` | Low |
| Entity enrichment | `enrich_entity()` | `enrich_entity` | Low |
| Graph visualization | `export_visualization()` | `export_graph` | Low |

### Phase 4: Enrichment Pipeline (Optional)

**Goal**: Enable Wikipedia/Diffbot enrichment for extracted entities.

- Wire `WikimediaProvider` into entity storage flow
- Add `NAM_ENRICHMENT__ENABLED=true` env var
- Background enrichment after entity extraction
- Low priority — most valuable for public knowledge bases, less so for private memory

---

## Where BAML Should Stay vs Where Base Extractors Win

### Keep BAML For:

1. **Entity extraction as LLM stage** — BAML's type safety and multi-provider fallback (Resilient client) is better than the base `LLMEntityExtractor` which only supports OpenAI
2. **Reasoning extraction** (`ExtractReasoning`) — entirely our addition, no base equivalent
3. **Reasoning synthesis** (`SynthesizeExplanation`) — entirely our addition
4. **Any future structured LLM tasks** — BAML's schema validation catches malformed responses

### Use Base Library For:

1. **spaCy stage** — free, fast (~5ms), great first pass for common entities
2. **GLiNER stage** — free, fast (~50ms), domain-aware schemas (8 built-in)
3. **Pipeline orchestration** — `ExtractionPipeline` handles stage ordering, timing, error recovery
4. **Merge strategies** — proven dedup logic at extraction time
5. **Entity resolution** — `CompositeResolver` with fuzzy/semantic matching
6. **Enrichment** — Wikipedia API, caching, background processing

### Hybrid Architecture (Target State):

```
Text Input
  │
  ├─ Stage 1: spaCy (~5ms, free)
  │    └─ Fast NER for common entity types
  │
  ├─ Stage 2: GLiNER (~50ms, free)  [optional]
  │    └─ Domain-specific entity detection
  │
  └─ Stage 3: BAML (~500ms, API cost)
       └─ Type-safe LLM extraction with multi-provider fallback
       └─ Replaces base LLMEntityExtractor
  │
  ▼
  Merge (CONFIDENCE strategy)
  │
  ▼
  Entity Resolution (CompositeResolver)
  │  exact → fuzzy (RapidFuzz) → semantic (embeddings)
  │
  ▼
  Store in Neo4j (with dedup)
```

---

## Implementation Priority

1. **Phase 1** (extraction pipeline integration) — Highest impact, enables multi-stage quality
2. **Phase 2** (entity resolution/dedup) — Second highest, prevents data quality degradation over time
3. **Phase 3** (expose base features as MCP tools) — Incremental value, can be done tool-by-tool
4. **Phase 4** (enrichment) — Nice to have, lowest priority

**Total estimated effort**: ~500 lines of new/modified code across 4-6 files

---

## Code References

- `src/neo4j_agent_memory/extraction/baml_extractor.py` — Current BAML extractor (satisfies EntityExtractor protocol)
- `src/neo4j_agent_memory/extraction/factory_ext.py:55-62` — Current binary BAML/base switch
- `src/neo4j_agent_memory/mcp/server.py:63-68` — Monkey-patch site
- `src/neo4j_agent_memory/mcp/_tools.py` — All 8 current MCP tools
- Base `extraction/factory.py:297-487` — ExtractorBuilder (needs `.with_baml()` extension)
- Base `extraction/pipeline.py:353-678` — ExtractionPipeline (BAML should participate here)
- Base `resolution/composite.py:20-346` — CompositeResolver (needs MCP tool exposure)
- Base `memory/long_term.py` — Dedup methods (find_duplicate_entities, merge_entities, auto_deduplicate)

## Open Questions

1. **GLiNER dependency**: Adding GLiNER as a pipeline stage requires the `gliner` Python package + model download (~500MB). Should this be optional?
2. **spaCy model**: The base library uses `en_core_web_sm` (English only). Do we need multilingual support?
3. **Embedding provider for semantic resolution**: `SemanticMatchResolver` needs an `Embedder`. Should we use OpenAI embeddings or a local model?
4. **Auto-dedup on store**: Should `memory_store` automatically run entity resolution, or should it be a separate explicit step?
5. **Pipeline configuration**: Should pipeline mode be configurable per-call or only via environment variables?
