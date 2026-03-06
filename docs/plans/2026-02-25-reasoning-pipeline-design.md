# Real Reasoning Pipeline — Design & Implementation Plan

## Overview

Transform the reasoning trace system from a post-hoc activity logger into a real reasoning capture and retrieval pipeline. Three parts:

1. **Upgrade `add_reasoning_trace`** — Accept actual thought/observation/reasoning per step from callers
2. **New BAML `ExtractReasoning` function** — Extract structured reasoning from raw conversation text
3. **New `explain_reasoning` MCP tool** — Retrieve reasoning chains with optional LLM synthesis
4. **New `extract_reasoning` MCP tool** — Extract reasoning from conversation transcripts

## Current State Analysis

### The Problem

The `add_reasoning_trace` MCP tool (`_tools.py:499-567`) creates a 1:1 mapping of steps to tool calls, but auto-generates the `thought` field:

```python
thought=f"Calling {tc.get('tool_name', 'unknown')}"
```

The base library's `ReasoningStep` model has proper `thought`, `action`, `observation` fields designed for a ReAct loop, but we never populate them meaningfully. The actual reasoning happens in the LLM's context window and evaporates when the session ends.

### What the Base Library Already Supports

The base `ReasoningMemory` class (`reasoning.py:324-1101`) has rich capabilities we don't expose:

- `add_step(trace_id, thought=..., action=..., observation=...)` — proper TAO fields
- `get_trace_with_steps(trace_id)` — full trace with steps and tool calls
- `get_similar_traces(task)` — vector similarity search on task embeddings
- `list_traces(session_id, success_only, since, until)` — paginated listing
- `get_context(query)` — formats traces as markdown for LLM injection
- `StreamingTraceRecorder` — async context manager for real-time capture
- Step embeddings generated from concatenated "Thought: ... Action: ... Observation: ..."

### Key Discoveries

- `_tools.py:537`: `thought` is auto-filled, never from caller input
- `_tools.py:539`: `observation` takes raw `tc.get("result")`, not interpreted meaning
- `_tools.py:534-546`: Individual `success` field on tool call dicts is **never read** — status always defaults to SUCCESS
- `memory_search` excludes traces by default (`memory_types` defaults to `["messages", "entities", "preferences"]`)
- Trace search results (`_tools.py:193-201`) return only task/outcome/success — no steps or reasoning chain
- No MCP tool to retrieve a full trace with steps
- No MCP prompt to guide reasoning explanation

### Codebase Review Notes (2026-02-25)

> Verified against live codebase at HEAD. All line number references and code claims confirmed accurate.

**Verified claims**:
- `_tools.py:537` — `thought` auto-generated as `f"Calling {tool_name}"` ✅
- `_tools.py:539` — `observation` passes through raw `tc.get("result")` ✅
- `_tools.py:534-546` — per-step `success` field never read ✅
- `_tools.py:132-133` — default `memory_types` excludes traces ✅
- `_tools.py:194-200` — trace search returns only 4 fields ✅
- Module docstring lists 6 tools ✅
- Base library `ReasoningMemory` class — all 9 claimed methods/classes confirmed present ✅
- `ToolCallStatus` enum has 6 members (PENDING, SUCCESS, FAILURE, ERROR, TIMEOUT, CANCELLED) ✅
- `ReasoningStep` model has `thought`, `action`, `observation` (all optional strings) ✅

**Additional base library details not mentioned in plan**:
- `list_traces` also accepts `limit`, `offset`, `order_by`, `order_dir` (plan only mentions first 4 params — usage in plan is correct, passes `limit=1`)
- `record_tool_call` also accepts keyword-only `duration_ms`, `error`, `auto_observation`, `message_id` — not needed for this plan but available for future use
- `get_trace_with_steps` also has a convenience alias `get_trace()` accepting `UUID | str`
- `get_similar_traces` stores similarity score in `trace.metadata["similarity"]`

## Desired End State

After implementation:

1. **Callers can store real reasoning** — `add_reasoning_trace` accepts `thought` (why), `observation` (what was learned), and `alternatives_considered` per step
2. **Users can ask "Why?"** — `explain_reasoning` retrieves the full reasoning chain for a trace or finds the most relevant trace for a query, optionally synthesized by an LLM
3. **Conversations become reasoning** — `extract_reasoning` takes conversation text and uses BAML to extract structured reasoning steps
4. **Search includes reasoning by default** — `memory_search` includes traces in default memory types

### Verification

- Calling `add_reasoning_trace` with `thought` and `observation` per step stores them in Neo4j
- Calling `explain_reasoning` with a trace_id returns the full chain
- Calling `explain_reasoning` with a query finds similar traces and returns reasoning
- Calling `explain_reasoning` with `synthesize=true` produces a natural-language explanation
- Calling `extract_reasoning` with conversation text produces structured reasoning steps stored as a trace
- `memory_search` with default types now includes trace results

## What We're NOT Doing

- Not modifying the base `neo4j-agent-memory` library — overlay only
- Not building a real-time streaming reasoning recorder (StreamingTraceRecorder is for framework integrations)
- Not adding a UI or visualization
- Not changing the graph schema — the existing ReasoningTrace/ReasoningStep/ToolCall nodes are sufficient
- Not building agent-side middleware — the calling LLM/agent is responsible for providing good content

---

## Phase 1: Upgrade `add_reasoning_trace` Tool Interface

### Overview
Fix the MCP tool to accept and store real reasoning content instead of auto-generating placeholders.

### Changes Required

#### 1. Update `add_reasoning_trace` in `_tools.py`

**File**: `src/neo4j_agent_memory/mcp/_tools.py`
**Lines**: 499-567

**Current tool_calls dict structure** (undocumented):
```python
{"tool_name": str, "arguments": dict, "result": Any, "success": bool}
```

**New structure — backward compatible**:
```python
{
    "tool_name": str,          # required — what tool was called
    "thought": str | None,     # NEW — WHY this action was chosen (hypothesis, reasoning)
    "observation": str | None, # NEW — WHAT was learned from the result (interpretation)
    "alternatives_considered": str | None,  # NEW — what other approaches were weighed
    "arguments": dict | None,  # existing — tool arguments
    "result": Any | None,      # existing — raw result
    "success": bool | None,    # existing — NOW ACTUALLY USED for ToolCallStatus
}
```

**Implementation**:

```python
@mcp.tool()
async def add_reasoning_trace(
    ctx: Context,
    session_id: str,
    task: str,
    steps: list[dict[str, Any]] | None = None,       # renamed from tool_calls
    tool_calls: list[dict[str, Any]] | None = None,   # backward compat alias
    outcome: str | None = None,
    success: bool = True,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Store a reasoning trace capturing HOW and WHY a task was solved.

    Records the task, reasoning steps with thought/action/observation,
    tool calls, and final outcome. Enables later retrieval via
    explain_reasoning to understand the agent's decision process.

    Args:
        session_id: Session ID for the reasoning trace.
        task: Description of the task being solved.
        steps: List of reasoning steps. Each step should include:
            - thought: WHY this action was chosen (hypothesis, reasoning)
            - tool_name: What tool/action was taken
            - observation: WHAT was learned from the result
            - alternatives_considered: Other approaches weighed (optional)
            - arguments: Tool arguments (optional)
            - result: Raw result (optional)
            - success: Whether this step succeeded (optional)
        tool_calls: Deprecated alias for steps (backward compatibility).
        outcome: Final outcome or result of the task.
        success: Whether the task was completed successfully.
        metadata: Optional metadata (model, latency, etc.).
    """
    client = get_client(ctx)
    # Use steps if provided, fall back to tool_calls for backward compat
    step_list = steps or tool_calls or []

    try:
        trace = await client.reasoning.start_trace(
            session_id=session_id,
            task=task,
            metadata=metadata or {},
        )

        for tc in step_list:
            tool_name = tc.get("tool_name", "unknown")

            # Use caller-provided thought, or generate a default
            thought = tc.get("thought") or f"Calling {tool_name}"

            # Use caller-provided observation, or fall back to raw result
            observation = tc.get("observation") or tc.get("result")
            if observation and not isinstance(observation, str):
                observation = str(observation)

            # Include alternatives in thought if provided
            alternatives = tc.get("alternatives_considered")
            if alternatives:
                thought = f"{thought}\n\nAlternatives considered: {alternatives}"

            step = await client.reasoning.add_step(
                trace_id=trace.id,
                thought=thought,
                action=tool_name,
                observation=observation,
            )

            # Map success field to ToolCallStatus
            from neo4j_agent_memory.memory.reasoning import ToolCallStatus
            tc_success = tc.get("success")
            if tc_success is False:
                status = ToolCallStatus.FAILURE
            elif tc_success is True:
                status = ToolCallStatus.SUCCESS
            else:
                status = ToolCallStatus.SUCCESS

            # NOTE: record_tool_call signature is:
            #   record_tool_call(step_id, tool_name, arguments, *, result=, status=, ...)
            # `result` and `status` are keyword-only in the base library.
            await client.reasoning.record_tool_call(
                step_id=step.id,
                tool_name=tool_name,
                arguments=tc.get("arguments", {}),
                result=tc.get("result"),
                status=status,
            )

        await client.reasoning.complete_trace(
            trace_id=trace.id,
            outcome=outcome,
            success=success,
        )

        return json.dumps({
            "success": True,
            "stored": True,
            "trace_id": str(trace.id),
            "session_id": session_id,
            "task": task,
            "step_count": len(step_list),
        })

    except Exception as e:
        logger.error(f"Error in add_reasoning_trace: {e}")
        return json.dumps({"error": str(e)})
```

#### 2. Update default memory_types in `memory_search`

**File**: `src/neo4j_agent_memory/mcp/_tools.py`
**Line**: 132-133

Change:
```python
if memory_types is None:
    memory_types = ["messages", "entities", "preferences"]
```

To:
```python
if memory_types is None:
    memory_types = ["messages", "entities", "preferences", "traces"]
```

#### 3. Enrich trace search results in `memory_search`

**File**: `src/neo4j_agent_memory/mcp/_tools.py`
**Lines**: 188-201

Change trace results to include step count and session_id:
```python
if "traces" in memory_types:
    traces = await client.reasoning.get_similar_traces(
        task=query,
        limit=limit,
    )
    results["traces"] = [
        {
            "id": str(trace.id),
            "task": trace.task,
            "outcome": trace.outcome,
            "success": trace.success,
            "session_id": trace.session_id,
            "started_at": trace.started_at.isoformat() if trace.started_at else None,
        }
        for trace in traces
    ]
```

### Success Criteria

#### Automated Verification:
- [x] Existing tests still pass: `uv run pytest tests/`
- [ ] Type checking passes: `uv run mypy src/`
- [x] Linting passes: `uv run ruff check src/`
- [x] Calling `add_reasoning_trace` with `thought` and `observation` per step stores them correctly (new test)
- [x] Calling `add_reasoning_trace` with old `tool_calls` format still works (backward compat test)
- [x] `memory_search` with default types now returns traces (new test)

#### Manual Verification:
- [ ] From Claude Code, store a reasoning trace with real thought/observation content
- [ ] Verify via `graph_query` that thought and observation are stored on ReasoningStep nodes

---

## Phase 2: New BAML Reasoning Extraction

### Overview
Add BAML functions to (a) extract structured reasoning from conversation text, and (b) synthesize explanations from stored reasoning chains.

### Changes Required

#### 1. New BAML types and functions

**File**: `baml_src/reasoning.baml` (NEW)

```baml
// Structured reasoning step extracted from conversation
class ExtractedReasoningStep {
  thought string @description("The reasoning or hypothesis behind the action — WHY it was done")
  action string @description("What action was taken or decision made")
  observation string @description("What was learned or observed from the action")
  alternatives_considered string? @description("Other approaches that were weighed and rejected")
  confidence float @description("Confidence in the extraction, 0.0 to 1.0")
}

class ReasoningExtractionOutput {
  task string @description("The overall task or question being addressed")
  steps ExtractedReasoningStep[] @description("The reasoning steps in chronological order")
  final_conclusion string @description("The final conclusion or answer reached")
  success bool @description("Whether the task was successfully completed")
}

function ExtractReasoning(text: string) -> ReasoningExtractionOutput {
  client Anthropic
  prompt #"
    Analyze the following conversation or text and extract the reasoning chain.

    Identify:
    1. What task or question was being addressed
    2. Each reasoning step: what was the hypothesis/thought, what action was taken,
       what was observed/learned, and what alternatives were considered
    3. The final conclusion reached
    4. Whether the task was successfully completed

    Focus on the REASONING — the "why" behind decisions, not just the "what".
    Look for:
    - Hypotheses being formed and tested
    - Evidence being gathered and interpreted
    - Decisions between alternatives
    - Conclusions drawn from observations
    - Course corrections when initial approaches failed

    ## Text to Analyze
    {{ text }}

    {{ ctx.output_format }}
  "#
}

class ReasoningChainInput {
  task string @description("The task that was solved")
  steps ReasoningStepInput[] @description("The reasoning steps taken")
  outcome string @description("The final outcome")
}

class ReasoningStepInput {
  thought string @description("Why this action was chosen")
  action string @description("What was done")
  observation string @description("What was learned")
}

function SynthesizeExplanation(chain: ReasoningChainInput) -> string {
  client Anthropic
  prompt #"
    Given the following reasoning chain from an AI agent's problem-solving process,
    produce a clear, natural-language explanation of WHY the agent came to its conclusion.

    Write as if answering the question: "Why did you come up with this answer?"

    Focus on:
    - The key decisions and why they were made
    - What evidence led to the conclusion
    - What alternatives were considered and why they were rejected
    - The logical flow from question to answer

    Be concise but thorough. Use first person ("I").

    ## Task
    {{ chain.task }}

    ## Reasoning Steps
    {% for step in chain.steps %}
    ### Step {{ loop.index }}
    **Thought**: {{ step.thought }}
    **Action**: {{ step.action }}
    **Observation**: {{ step.observation }}
    {% endfor %}

    ## Outcome
    {{ chain.outcome }}

    {{ ctx.output_format }}
  "#
}
```

#### 2. Regenerate BAML client

Run `uv run baml-cli generate` after creating the new `.baml` file. This will update the generated client at `src/neo4j_agent_memory/baml_client/` with new functions `ExtractReasoning` and `SynthesizeExplanation`.

> **Verified**: `baml-cli generate` compiles the entire `baml_src/` directory, so adding a second `.baml` file alongside `extraction.baml` is fully supported. The existing project has 3 files in `baml_src/`: `extraction.baml` (functions), `clients.baml` (LLM clients including Anthropic with `claude-sonnet-4-20250514`), and `generators.baml` (output config targeting `../src/neo4j_agent_memory` with `python/pydantic`). The new `reasoning.baml` will be picked up automatically.

#### 3. New reasoning extractor class

**File**: `src/neo4j_agent_memory/extraction/reasoning_extractor.py` (NEW)

```python
"""BAML-based reasoning extraction from conversation text."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BamlReasoningExtractor:
    """Extract structured reasoning chains from conversation text using BAML.

    Uses the ExtractReasoning BAML function to identify hypotheses,
    decisions, evidence, and conclusions from raw text.
    """

    def __init__(
        self,
        *,
        client_name: str = "Anthropic",
        client_registry: Any | None = None,
    ):
        self._client_name = client_name
        self._baml_options: dict[str, Any] = {}

        if client_registry:
            self._baml_options["client_registry"] = client_registry
        elif client_name != "Anthropic":
            try:
                from baml_py import ClientRegistry

                registry = ClientRegistry()
                registry.set_primary(client_name)
                self._baml_options["client_registry"] = registry
            except ImportError:
                logger.warning("baml-py not installed, client_name override ignored")

    async def extract_reasoning(self, text: str) -> dict[str, Any]:
        """Extract structured reasoning from conversation text.

        Args:
            text: Conversation transcript or text to analyze.

        Returns:
            Dict with task, steps (thought/action/observation), conclusion, success.
        """
        if not text or not text.strip():
            return {"task": "", "steps": [], "final_conclusion": "", "success": False}

        from neo4j_agent_memory.baml_client.async_client import b

        result = await b.ExtractReasoning(
            text=text,
            **(self._baml_options if self._baml_options else {}),
        )

        return {
            "task": result.task,
            "steps": [
                {
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "alternatives_considered": step.alternatives_considered,
                    "confidence": max(0.0, min(1.0, step.confidence)),
                }
                for step in result.steps
            ],
            "final_conclusion": result.final_conclusion,
            "success": result.success,
        }

    async def synthesize_explanation(
        self,
        task: str,
        steps: list[dict[str, str]],
        outcome: str,
    ) -> str:
        """Synthesize a natural-language explanation from a reasoning chain.

        Args:
            task: The task that was solved.
            steps: List of {thought, action, observation} dicts.
            outcome: The final outcome.

        Returns:
            Natural-language explanation string.
        """
        from neo4j_agent_memory.baml_client.async_client import b
        from neo4j_agent_memory.baml_client.types import (
            ReasoningChainInput,
            ReasoningStepInput,
        )

        chain = ReasoningChainInput(
            task=task,
            steps=[
                ReasoningStepInput(
                    thought=s.get("thought", ""),
                    action=s.get("action", ""),
                    observation=s.get("observation", ""),
                )
                for s in steps
            ],
            outcome=outcome,
        )

        return await b.SynthesizeExplanation(
            chain=chain,
            **(self._baml_options if self._baml_options else {}),
        )
```

### Success Criteria

#### Automated Verification:
- [x] BAML generation succeeds: `uv run baml-cli generate`
- [x] New types exist in `src/neo4j_agent_memory/baml_client/types.py`
- [x] New functions exist in `src/neo4j_agent_memory/baml_client/async_client.py`
- [x] Unit tests for `BamlReasoningExtractor` pass (mock BAML calls)
- [x] Linting passes: `uv run ruff check src/`

#### Manual Verification:
- [ ] `ExtractReasoning` with a sample conversation returns sensible steps
- [ ] `SynthesizeExplanation` with a sample chain returns a coherent narrative

---

## Phase 3: New MCP Tools — `explain_reasoning` and `extract_reasoning`

### Overview
Add two new MCP tools: one to retrieve and explain past reasoning, one to extract reasoning from text.

### Changes Required

#### 1. Add `explain_reasoning` tool

**File**: `src/neo4j_agent_memory/mcp/_tools.py`

```python
@mcp.tool()
async def explain_reasoning(
    ctx: Context,
    trace_id: str | None = None,
    query: str | None = None,
    session_id: str | None = None,
    synthesize: bool = False,
) -> str:
    """Retrieve and explain the reasoning behind a past decision or answer.

    Use this to answer "Why did you come up with this answer?" by retrieving
    the full reasoning chain (thought → action → observation for each step).

    Provide ONE of:
    - trace_id: Get reasoning for a specific trace
    - query: Find the most relevant trace by semantic similarity
    - session_id: Get the most recent trace from a session

    Set synthesize=true for a natural-language explanation powered by LLM.
    Default returns the raw reasoning chain.

    Args:
        trace_id: Specific trace ID to explain.
        query: Semantic search query to find relevant trace.
        session_id: Session ID to get most recent trace from.
        synthesize: If true, produce LLM-synthesized explanation (slower).
    """
    client = get_client(ctx)

    try:
        trace = None

        # Route 1: Direct trace lookup
        if trace_id:
            trace = await client.reasoning.get_trace_with_steps(trace_id)
            if not trace:
                return json.dumps({"error": f"Trace not found: {trace_id}"})

        # Route 2: Semantic search for most relevant trace
        elif query:
            similar = await client.reasoning.get_similar_traces(
                task=query, limit=1, success_only=False,
            )
            if not similar:
                return json.dumps({
                    "found": False,
                    "message": "No reasoning traces found matching query",
                    "query": query,
                })
            trace = await client.reasoning.get_trace_with_steps(similar[0].id)

        # Route 3: Most recent trace from session
        elif session_id:
            traces = await client.reasoning.list_traces(
                session_id=session_id, limit=1,
            )
            if not traces:
                return json.dumps({
                    "found": False,
                    "message": f"No traces found for session: {session_id}",
                })
            trace = await client.reasoning.get_trace_with_steps(traces[0].id)

        else:
            return json.dumps({
                "error": "Provide trace_id, query, or session_id"
            })

        if not trace:
            return json.dumps({"error": "Failed to retrieve trace details"})

        # Build the reasoning chain
        chain = {
            "trace_id": str(trace.id),
            "task": trace.task,
            "session_id": trace.session_id,
            "success": trace.success,
            "outcome": trace.outcome,
            "started_at": trace.started_at.isoformat() if trace.started_at else None,
            "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
            "steps": [
                {
                    "step_number": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "arguments": tc.arguments,
                            "status": tc.status.value if hasattr(tc.status, "value") else str(tc.status),
                        }
                        for tc in (step.tool_calls or [])
                    ],
                }
                for step in (trace.steps or [])
            ],
        }

        # Optional: synthesize a natural-language explanation
        if synthesize and trace.steps:
            try:
                from neo4j_agent_memory.extraction.reasoning_extractor import (
                    BamlReasoningExtractor,
                )

                extractor = BamlReasoningExtractor()
                explanation = await extractor.synthesize_explanation(
                    task=trace.task,
                    steps=[
                        {
                            "thought": s.thought or "",
                            "action": s.action or "",
                            "observation": s.observation or "",
                        }
                        for s in trace.steps
                    ],
                    outcome=trace.outcome or "",
                )
                chain["synthesized_explanation"] = explanation
            except Exception as synth_err:
                logger.warning(f"Synthesis failed, returning raw chain: {synth_err}")
                chain["synthesis_error"] = str(synth_err)

        return json.dumps(chain, default=str)

    except Exception as e:
        logger.error(f"Error in explain_reasoning: {e}")
        return json.dumps({"error": str(e)})
```

#### 2. Add `extract_reasoning` tool

**File**: `src/neo4j_agent_memory/mcp/_tools.py`

```python
@mcp.tool()
async def extract_reasoning(
    ctx: Context,
    text: str,
    session_id: str,
    store: bool = True,
) -> str:
    """Extract structured reasoning from conversation text using LLM analysis.

    Analyzes a conversation transcript to identify the reasoning chain:
    hypotheses formed, evidence gathered, decisions made, and conclusions drawn.

    Optionally stores the extracted reasoning as a trace in memory.

    Args:
        text: Conversation transcript or text to analyze.
        session_id: Session ID to associate with the extracted trace.
        store: If true (default), store extracted reasoning as a trace.
    """
    client = get_client(ctx)

    try:
        from neo4j_agent_memory.extraction.reasoning_extractor import (
            BamlReasoningExtractor,
        )

        extractor = BamlReasoningExtractor()
        extracted = await extractor.extract_reasoning(text)

        result = {
            "extracted": True,
            "task": extracted["task"],
            "step_count": len(extracted["steps"]),
            "conclusion": extracted["final_conclusion"],
            "success": extracted["success"],
            "steps": extracted["steps"],
        }

        # Store as a reasoning trace if requested
        if store and extracted["steps"]:
            trace = await client.reasoning.start_trace(
                session_id=session_id,
                task=extracted["task"],
                metadata={"source": "conversation_extraction"},
            )

            for ext_step in extracted["steps"]:
                await client.reasoning.add_step(
                    trace_id=trace.id,
                    thought=ext_step["thought"],
                    action=ext_step["action"],
                    observation=ext_step["observation"],
                )

            await client.reasoning.complete_trace(
                trace_id=trace.id,
                outcome=extracted["final_conclusion"],
                success=extracted["success"],
            )

            result["stored"] = True
            result["trace_id"] = str(trace.id)

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Error in extract_reasoning: {e}")
        return json.dumps({"error": str(e)})
```

#### 3. Update module docstring and tool count

**File**: `src/neo4j_agent_memory/mcp/_tools.py`
**Lines**: 1-9

Update to reflect 8 tools:
```python
"""MCP tool implementations for Neo4j Agent Memory.

Defines the 8 core tools as FastMCP @mcp.tool decorated functions:
- memory_search: Hybrid vector + graph search
- memory_store: Store memories (messages, facts, preferences)
- entity_lookup: Get entity with relationships
- conversation_history: Get conversation for session
- graph_query: Execute read-only Cypher queries
- add_reasoning_trace: Store procedural memory with real reasoning
- explain_reasoning: Retrieve and explain past reasoning chains
- extract_reasoning: Extract reasoning from conversation text
"""
```

#### 4. Add `reasoning_explanation` MCP prompt

**File**: `src/neo4j_agent_memory/mcp/_prompts.py`

> **Important**: All prompts in `_prompts.py` are defined **inside** the `register_prompts(mcp: FastMCP)` wrapper function (line 19). The new prompt must be nested inside this function, not at module level. The existing prompts (`memory_search_guide`, `entity_analysis`, `conversation_summary`) are all defined as `@mcp.prompt()` decorated functions within `register_prompts()`.

Add after `conversation_summary` (inside `register_prompts()`):

```python
@mcp.prompt()
def reasoning_explanation(
    query: str,
    synthesize: str = "false",
) -> list[Message]:
    """Explain the reasoning behind a past decision or answer.

    Guides retrieval and explanation of reasoning chains
    from stored traces.
    """
    synth = synthesize.lower() in ("true", "1", "yes")
    return [
        Message(
            role="user",
            content=(
                f"Explain the reasoning behind: {query}\n\n"
                "Steps:\n"
                f"1. Use explain_reasoning with query='{query}'"
                f"{' and synthesize=true' if synth else ''}\n"
                "2. If a reasoning trace is found, present:\n"
                "   - The task that was being solved\n"
                "   - Each reasoning step: thought (why), action (what), observation (learned)\n"
                "   - The final outcome and conclusion\n"
                "3. If synthesize was used, present the natural-language explanation\n"
                "4. If no trace is found, suggest using memory_search to find related context"
            ),
        )
    ]
```

### Success Criteria

#### Automated Verification:
- [x] All existing tests pass: `uv run pytest tests/`
- [x] New tests for `explain_reasoning` pass (mock client)
- [x] New tests for `extract_reasoning` pass (mock BAML + client)
- [ ] Type checking passes: `uv run mypy src/`
- [x] Linting passes: `uv run ruff check src/`

#### Manual Verification:
- [ ] Store a reasoning trace with real thought/observation via Claude Code
- [ ] Call `explain_reasoning` with the trace_id — see full chain
- [ ] Call `explain_reasoning` with a query — finds relevant trace
- [ ] Call `explain_reasoning` with `synthesize=true` — get natural-language explanation
- [ ] Call `extract_reasoning` with a conversation transcript — see extracted steps
- [ ] Verify extracted reasoning is stored as a trace in Neo4j

---

## Phase 4: Tests

### Overview
Add tests for all new and modified functionality.

### New Test Files

#### 1. `tests/test_reasoning_tools.py` (NEW)

Tests for:
- `add_reasoning_trace` with new `steps` format (thought/observation/alternatives)
- `add_reasoning_trace` backward compatibility with old `tool_calls` format
- `add_reasoning_trace` with mixed (some steps have thought, some don't)
- `explain_reasoning` with trace_id lookup
- `explain_reasoning` with query-based search
- `explain_reasoning` with session_id lookup
- `explain_reasoning` with synthesize=true (mock BAML)
- `explain_reasoning` with no matching trace
- `extract_reasoning` with conversation text (mock BAML)
- `extract_reasoning` with store=true creates trace
- `extract_reasoning` with store=false skips storage
- `memory_search` default types now include traces

#### 2. `tests/test_reasoning_extractor.py` (NEW)

Tests for:
- `BamlReasoningExtractor.extract_reasoning` with valid text
- `BamlReasoningExtractor.extract_reasoning` with empty text
- `BamlReasoningExtractor.synthesize_explanation` with valid chain
- Client name override behavior
- Confidence clamping on extracted steps

### Success Criteria

#### Automated Verification:
- [x] All tests pass: `uv run pytest tests/ -v`
- [x] No test uses live API calls (all BAML calls mocked)

---

## Testing Strategy

### Unit Tests (mocked)
- All BAML calls mocked via `conftest.py` fixtures (extending existing `mock_baml_extract`)
- Client reasoning methods mocked for tool-level tests
- Focus on data flow: correct fields passed through, backward compat maintained

> **Test Infrastructure Notes (from codebase review)**:
>
> **Existing conventions to follow**:
> - `asyncio_mode = "auto"` in `pyproject.toml:31` — no `@pytest.mark.asyncio` decorators needed
> - Class-based test grouping (`class TestXyz:`) with bare `async def test_*` methods
> - Deferred imports inside test functions (not at module top)
> - `conftest.py` fixtures return tuples of `(mock_fn, mock_result)` for control + inspection
> - `monkeypatch.setattr` preferred over `with patch(...)` context managers
>
> **Key mocking difference from existing tests**:
> The existing `mock_baml_extract` fixture patches `b.ExtractEntities` on the BAML client.
> For reasoning tool tests, you need a **different mocking layer** — the MCP tools call
> `get_client(ctx)` which returns a `MemoryClient` with a `.reasoning` attribute.
> New fixtures should mock `client.reasoning.start_trace`, `client.reasoning.add_step`,
> `client.reasoning.record_tool_call`, `client.reasoning.complete_trace`,
> `client.reasoning.get_trace_with_steps`, `client.reasoning.get_similar_traces`,
> and `client.reasoning.list_traces`. Consider a `mock_reasoning_client` fixture in `conftest.py`.
>
> **No existing tool-level tests**: Currently zero tests cover any of the 6 MCP tool functions
> in `_tools.py`. These will be the first tool-level tests in the project.

### Integration Tests (requires Neo4j)
- Store trace → retrieve via explain_reasoning → verify chain completeness
- Extract reasoning → verify stored trace in Neo4j
- Search traces → verify results include reasoning content

### Manual Testing Steps
1. Start the MCP server: `neo4j-memory-mcp`
2. From Claude Code, have a conversation and store reasoning with `add_reasoning_trace`
3. Ask "Why did you make that decision?" and use `explain_reasoning` to retrieve it
4. Use `extract_reasoning` on a conversation transcript
5. Use `explain_reasoning` with `synthesize=true` to get a polished explanation

## Performance Considerations

- `explain_reasoning` without `synthesize` is a pure graph read — fast
- `explain_reasoning` with `synthesize=true` requires an LLM call — adds 2-5s latency
- `extract_reasoning` always requires an LLM call (BAML extraction)
- No additional vector indexes needed — existing `task_embedding_idx` is sufficient

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `src/neo4j_agent_memory/mcp/_tools.py` | MODIFY | Upgrade `add_reasoning_trace`, add `explain_reasoning`, add `extract_reasoning`, update default memory_types |
| `baml_src/reasoning.baml` | CREATE | New BAML types and functions for reasoning extraction/synthesis |
| `src/neo4j_agent_memory/extraction/reasoning_extractor.py` | CREATE | `BamlReasoningExtractor` class |
| `src/neo4j_agent_memory/mcp/_prompts.py` | MODIFY | Add `reasoning_explanation` prompt |
| `src/neo4j_agent_memory/baml_client/*` | REGENERATE | Updated by `baml-cli generate` |
| `tests/test_reasoning_tools.py` | CREATE | Tests for new/modified tools |
| `tests/test_reasoning_extractor.py` | CREATE | Tests for BAML reasoning extractor |

## References

- Base library reasoning module: `.venv/lib/python3.11/site-packages/neo4j_agent_memory/memory/reasoning.py`
- Base library graph queries: `.venv/lib/python3.11/site-packages/neo4j_agent_memory/graph/queries.py`
- Existing BAML extraction: `baml_src/extraction.baml`
- Existing BAML extractor: `src/neo4j_agent_memory/extraction/baml_extractor.py`
- Previous implementation plans: `docs/plans/2026-02-19-baml-extraction-implementation.md`
