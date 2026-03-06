# BAML Entity Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add BAML as a new entity extraction backend with multi-provider LLM support (OpenAI, Anthropic, Gemini, fallback chains) alongside existing extractors.

**Architecture:** Overlay pattern — new files in `src/neo4j_agent_memory/extraction/` shadow the installed package's extraction subpackage. A `__path__` extension trick (same as the top-level `__init__.py`) ensures both overlay modules (BAML) and base modules (LLM, spaCy, GLiNER, factory) remain importable. The factory is monkey-patched at server startup to route `BAML_ENABLED=true` requests to `BamlEntityExtractor`. BAML's generated client lives inside the overlay package so it ships in the wheel.

**Tech Stack:** baml-py, BAML DSL, Python asyncio, Pydantic, pytest

**RFI Analysis:** See [2026-02-19-baml-integration-rfi-analysis.md](./2026-02-19-baml-integration-rfi-analysis.md) for the full risk/friction/integration review. All amendments (I1, I3, R1, F1, F4) are incorporated below.

---

## Task 1: Add baml-py Dependency

**Files:**
- Modify: `pyproject.toml:6-9`

**Step 1: Add the dependency**

In `pyproject.toml`, add `baml-py` to the dependencies list:

```toml
dependencies = [
    "neo4j-agent-memory[mcp]",
    "fastmcp>=2.0.0,<3",
    "baml-py>=0.70.0",
]
```

**Step 2: Install**

Run: `uv sync`
Expected: completes successfully, `baml-py` appears in `.venv`

**Step 3: Verify**

Run: `uv run python -c "import baml_py; print(f'baml-py {baml_py.__version__}')" `
Expected: prints version like `baml-py 0.7x.x`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add baml-py dependency for BAML extraction backend"
```

---

## Task 2: Create BAML Source Files

**Files:**
- Create: `baml_src/generators.baml`
- Create: `baml_src/clients.baml`
- Create: `baml_src/extraction.baml`

**Step 1: Create generator config**

Create `baml_src/generators.baml`. Note: `output_dir` points into the overlay package so the generated client ships in the wheel `[RFI-I3]`:

```baml
generator lang_python {
  output_type "python/pydantic"
  output_dir "../src/neo4j_agent_memory"
  version "0.70.0"
}
```

> **Important:** The `version` field must match the installed `baml-py` version exactly. After `uv sync`, check with `uv run python -c "import baml_py; print(baml_py.__version__)"` and update this field if it differs.

**Step 2: Create client definitions**

Create `baml_src/clients.baml`:

```baml
// Retry policy for transient failures
retry_policy StandardRetry {
  max_retries 2
}

// --- Individual Providers ---

client<llm> OpenAI {
  provider openai
  retry_policy StandardRetry
  options {
    model "gpt-4o-mini"
    temperature 0
  }
}

client<llm> Anthropic {
  provider anthropic
  retry_policy StandardRetry
  options {
    model "claude-sonnet-4-20250514"
    temperature 0
    max_tokens 4096
  }
}

client<llm> Gemini {
  provider google-ai
  retry_policy StandardRetry
  options {
    model "gemini-2.5-flash"
    api_key env.GEMINI_API_KEY
  }
}

// --- Fallback Chain ---
// Tries each provider in order until one succeeds

client<llm> Resilient {
  provider fallback
  options {
    strategy [
      OpenAI
      Anthropic
      Gemini
    ]
  }
}
```

**Step 3: Create extraction types and function**

Create `baml_src/extraction.baml`:

```baml
// POLE+O Entity Types
enum EntityType {
  PERSON @description("Individuals, people mentioned by name or role")
  ORGANIZATION @description("Companies, groups, institutions")
  LOCATION @description("Places, addresses, geographic areas, landmarks")
  EVENT @description("Incidents, meetings, transactions, things that happened")
  OBJECT @description("Physical or digital items: vehicles, phones, documents, devices")
}

class ExtractedEntity {
  name string @description("The entity name as it appears in text")
  type EntityType
  subtype string? @description("Optional specific subtype, e.g. VEHICLE, ADDRESS, COMPANY")
  confidence float @description("Extraction confidence from 0.0 to 1.0")
}

class ExtractedRelation {
  source string @description("Source entity name")
  target string @description("Target entity name")
  relation_type string @description("Relationship type, e.g. WORKS_AT, LIVES_IN, OWNS")
  confidence float @description("Relation confidence from 0.0 to 1.0")
}

class ExtractedPreference {
  category string @description("Preference category: food, music, tech, communication, etc.")
  preference string @description("The preference statement")
  context string? @description("When/where this preference applies")
  confidence float @description("Preference confidence from 0.0 to 1.0")
}

class ExtractionOutput {
  entities ExtractedEntity[]
  relations ExtractedRelation[]
  preferences ExtractedPreference[]
}

function ExtractEntities(text: string, entity_types: string) -> ExtractionOutput {
  client OpenAI
  prompt #"
    Extract entities, relationships, and preferences from the following text.

    ## Entity Types (POLE+O Model)
    Extract entities of these types: {{ entity_types }}

    ## Guidelines
    - PERSON: Individuals, people mentioned by name or role
    - OBJECT: Physical or digital items (vehicles, phones, documents, devices)
    - LOCATION: Places, addresses, geographic areas, landmarks
    - EVENT: Incidents, meetings, transactions, things that happened
    - ORGANIZATION: Companies, groups, institutions

    For relations:
    - Identify how entities are connected
    - Use clear relationship types (WORKS_AT, LIVES_IN, OWNS, ATTENDED, KNOWS, etc.)
    - Only include relations between entities in the entities list

    For preferences:
    - User preferences, likes, dislikes, opinions
    - Categories: food, music, communication, style, technology, etc.

    Confidence: 0.0-1.0 based on certainty of extraction

    ## Text to Analyze
    {{ text }}

    {{ ctx.output_format }}
  "#
}
```

**Step 4: Generate the BAML client**

Run: `uv run baml-cli generate`
Expected: creates `src/neo4j_agent_memory/baml_client/` with `types.py`, `async_client.py`, `sync_client.py`

**Step 5: Verify the generated client imports**

Run: `uv run python -c "from neo4j_agent_memory.baml_client.async_client import b; print('BAML client OK')"`
Expected: prints `BAML client OK`

**Step 6: Commit**

```bash
git add baml_src/ src/neo4j_agent_memory/baml_client/
git commit -m "feat: add BAML source definitions and generated client

Defines POLE+O entity extraction types, LLM client configs (OpenAI,
Anthropic, Gemini, Resilient fallback), and generates the Python client
inside the overlay package for wheel inclusion."
```

---

## Task 3: Create Extraction Overlay `__init__.py` (RFI-I1 Showstopper Fix)

This is the most critical file. Without it, creating any file in `src/neo4j_agent_memory/extraction/` will shadow the entire installed extraction package, breaking all imports.

**Files:**
- Create: `src/neo4j_agent_memory/extraction/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_extraction_overlay.py`:

```python
"""Tests for extraction subpackage overlay [RFI-I1].

Verifies that the overlay extraction __init__.py correctly extends
__path__ so both overlay modules and base package modules are importable.
"""

import pytest


def test_base_extraction_imports_still_work():
    """Base package extraction modules must remain importable through overlay."""
    from neo4j_agent_memory.extraction.base import (
        EntityExtractor,
        ExtractedEntity,
        ExtractionResult,
        NoOpExtractor,
    )

    assert EntityExtractor is not None
    assert ExtractedEntity is not None
    assert ExtractionResult is not None
    assert NoOpExtractor is not None


def test_factory_importable_through_overlay():
    """Factory must be importable — monkey-patch depends on this."""
    from neo4j_agent_memory.extraction.factory import create_extractor

    assert callable(create_extractor)


def test_extraction_package_exports_preserved():
    """All __all__ exports from base extraction package must be available."""
    import neo4j_agent_memory.extraction as ext

    # Core exports that must exist
    assert hasattr(ext, "EntityExtractor")
    assert hasattr(ext, "ExtractedEntity")
    assert hasattr(ext, "ExtractionResult")
    assert hasattr(ext, "NoOpExtractor")
    assert hasattr(ext, "create_extractor")
    assert hasattr(ext, "ExtractionPipeline")


def test_overlay_path_has_both_dirs():
    """__path__ must contain both overlay and installed directories."""
    import neo4j_agent_memory.extraction as ext

    assert len(ext.__path__) >= 2, (
        f"Expected at least 2 entries in __path__, got {len(ext.__path__)}: {ext.__path__}"
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_extraction_overlay.py -v`
Expected: FAIL — `src/neo4j_agent_memory/extraction/` directory doesn't exist yet, so imports resolve to the installed package and the `__path__` test fails (only 1 entry). That's fine — we need the test to exist first.

> Note: Some tests may actually pass because the overlay dir doesn't exist yet. The key test is `test_overlay_path_has_both_dirs` which will fail once we create the directory, unless we implement the `__path__` trick.

**Step 3: Create the overlay `__init__.py`**

Create `src/neo4j_agent_memory/extraction/__init__.py`:

```python
"""Extraction subpackage overlay — extends installed package with BAML support.

This module replicates the __path__ extension trick from the top-level
neo4j_agent_memory/__init__.py so that BOTH overlay modules (baml_extractor,
baml_config, factory_ext) AND base package modules (base, factory,
llm_extractor, spacy_extractor, etc.) remain importable.

Without this, creating any .py file in this directory would shadow the
entire installed extraction package. See RFI-I1.
"""

import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# 1. Locate the installed extraction package in site-packages.
# ---------------------------------------------------------------------------
_overlay_dir = os.path.dirname(os.path.abspath(__file__))

_installed_dir = None
for _p in sys.path:
    _candidate = os.path.join(_p, "neo4j_agent_memory", "extraction")
    if (
        os.path.isdir(_candidate)
        and os.path.normpath(_candidate) != os.path.normpath(_overlay_dir)
        and os.path.isfile(os.path.join(_candidate, "__init__.py"))
    ):
        _installed_dir = _candidate
        break

# ---------------------------------------------------------------------------
# 2. Extend __path__: overlay first, installed second.
# ---------------------------------------------------------------------------
if _installed_dir:
    __path__ = [_overlay_dir, _installed_dir]
else:
    __path__ = [_overlay_dir]

# ---------------------------------------------------------------------------
# 3. Execute the installed extraction __init__.py to preserve all exports
#    (EntityExtractor, ExtractedEntity, create_extractor, pipelines, etc.).
# ---------------------------------------------------------------------------
if _installed_dir:
    _base_init = os.path.join(_installed_dir, "__init__.py")
    if os.path.isfile(_base_init):
        _spec = importlib.util.spec_from_file_location(
            "neo4j_agent_memory.extraction._base_init",
            _base_init,
            submodule_search_locations=[_installed_dir],
        )
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            # Copy all public symbols
            _base_all = getattr(_mod, "__all__", [])
            for _name in _base_all:
                if hasattr(_mod, _name):
                    globals()[_name] = getattr(_mod, _name)
            # Preserve __all__
            if _base_all:
                __all__ = list(_base_all)
            # Preserve __getattr__ for lazy imports (SpacyEntityExtractor, etc.)
            if hasattr(_mod, "__getattr__"):
                _base_getattr = _mod.__getattr__

                def __getattr__(name: str):
                    return _base_getattr(name)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_extraction_overlay.py -v`
Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add src/neo4j_agent_memory/extraction/__init__.py tests/test_extraction_overlay.py
git commit -m "feat: extraction overlay with __path__ extension [RFI-I1]

Critical fix — without this, creating overlay extraction modules would
shadow the entire installed extraction package, breaking all imports
for LLMEntityExtractor, factory, pipelines, etc."
```

---

## Task 4: Create BamlEntityExtractor

**Files:**
- Create: `src/neo4j_agent_memory/extraction/baml_extractor.py`
- Create: `tests/conftest.py`
- Create: `tests/test_baml_extractor.py`

**Step 1: Create the test fixtures**

Create `tests/conftest.py`:

```python
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
```

**Step 2: Write the failing tests**

Create `tests/test_baml_extractor.py`:

```python
"""Tests for BamlEntityExtractor."""

import pytest
from unittest.mock import MagicMock

from neo4j_agent_memory.extraction.base import (
    EntityExtractor,
    ExtractionResult,
)


class TestBamlExtractorProtocol:
    """Verify BamlEntityExtractor satisfies EntityExtractor protocol."""

    def test_satisfies_protocol(self):
        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        assert isinstance(extractor, EntityExtractor)

    def test_has_name_property(self):
        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor(client_name="Anthropic")
        assert extractor.name == "BamlEntityExtractor(Anthropic)"


class TestBamlExtractorEmptyInput:
    """Verify empty/whitespace text returns empty result without calling BAML."""

    @pytest.mark.asyncio
    async def test_empty_string(self):
        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("")
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("   \n  ")
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0


class TestBamlExtractorConversion:
    """Verify BAML types are correctly converted to base extraction types."""

    @pytest.mark.asyncio
    async def test_entity_conversion(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        # Set up mock BAML response
        entity = MagicMock()
        entity.name = "John Smith"
        entity.type = MagicMock()
        entity.type.value = "PERSON"
        entity.subtype = None
        entity.confidence = 0.95
        mock_result.entities = [entity]
        mock_result.relations = []
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("John Smith works at Acme Corp")

        assert len(result.entities) == 1
        assert result.entities[0].name == "John Smith"
        assert result.entities[0].type == "PERSON"
        assert result.entities[0].confidence == 0.95
        assert result.entities[0].extractor == "baml"

    @pytest.mark.asyncio
    async def test_relation_conversion(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        entity1 = MagicMock()
        entity1.name = "John"
        entity1.type = MagicMock(value="PERSON")
        entity1.subtype = None
        entity1.confidence = 0.9

        entity2 = MagicMock()
        entity2.name = "Acme"
        entity2.type = MagicMock(value="ORGANIZATION")
        entity2.subtype = None
        entity2.confidence = 0.85

        relation = MagicMock()
        relation.source = "John"
        relation.target = "Acme"
        relation.relation_type = "works_at"
        relation.confidence = 0.8

        mock_result.entities = [entity1, entity2]
        mock_result.relations = [relation]
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("John works at Acme")

        assert len(result.relations) == 1
        assert result.relations[0].source == "John"
        assert result.relations[0].target == "Acme"
        assert result.relations[0].relation_type == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_relation_filtered_when_entity_missing(self, mock_baml_extract):
        """Relations referencing non-extracted entities are filtered out."""
        mock_fn, mock_result = mock_baml_extract

        entity = MagicMock()
        entity.name = "John"
        entity.type = MagicMock(value="PERSON")
        entity.subtype = None
        entity.confidence = 0.9

        relation = MagicMock()
        relation.source = "John"
        relation.target = "UnknownEntity"
        relation.relation_type = "knows"
        relation.confidence = 0.5

        mock_result.entities = [entity]
        mock_result.relations = [relation]
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("John knows someone")

        assert len(result.relations) == 0

    @pytest.mark.asyncio
    async def test_preference_conversion(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        pref = MagicMock()
        pref.category = "food"
        pref.preference = "likes pizza"
        pref.context = "for dinner"
        pref.confidence = 0.7

        mock_result.entities = []
        mock_result.relations = []
        mock_result.preferences = [pref]

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("I like pizza for dinner")

        assert len(result.preferences) == 1
        assert result.preferences[0].category == "food"
        assert result.preferences[0].preference == "likes pizza"

    @pytest.mark.asyncio
    async def test_confidence_clamped(self, mock_baml_extract):
        """Confidence values outside [0,1] are clamped."""
        mock_fn, mock_result = mock_baml_extract

        entity = MagicMock()
        entity.name = "Test"
        entity.type = MagicMock(value="PERSON")
        entity.subtype = None
        entity.confidence = 1.5  # Out of range

        mock_result.entities = [entity]
        mock_result.relations = []
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor()
        result = await extractor.extract("Test entity")

        assert result.entities[0].confidence == 1.0


class TestBamlExtractorOptions:
    """Verify extract_relations and extract_preferences flags."""

    @pytest.mark.asyncio
    async def test_relations_disabled(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        entity = MagicMock()
        entity.name = "John"
        entity.type = MagicMock(value="PERSON")
        entity.subtype = None
        entity.confidence = 0.9

        relation = MagicMock()
        relation.source = "John"
        relation.target = "John"
        relation.relation_type = "self"
        relation.confidence = 0.5

        mock_result.entities = [entity]
        mock_result.relations = [relation]
        mock_result.preferences = []

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor(extract_relations=False)
        result = await extractor.extract("John")

        assert len(result.relations) == 0

    @pytest.mark.asyncio
    async def test_preferences_disabled(self, mock_baml_extract):
        mock_fn, mock_result = mock_baml_extract

        pref = MagicMock()
        pref.category = "food"
        pref.preference = "likes pizza"
        pref.context = None
        pref.confidence = 0.7

        mock_result.entities = []
        mock_result.relations = []
        mock_result.preferences = [pref]

        from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

        extractor = BamlEntityExtractor(extract_preferences=False)
        result = await extractor.extract("I like pizza")

        assert len(result.preferences) == 0
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_baml_extractor.py -v`
Expected: FAIL — `baml_extractor` module doesn't exist yet

**Step 4: Write the implementation**

Create `src/neo4j_agent_memory/extraction/baml_extractor.py`:

```python
"""BAML-based entity extraction with multi-provider support."""

import logging
from typing import Any

from neo4j_agent_memory.extraction.base import (
    ExtractedEntity,
    ExtractedPreference,
    ExtractedRelation,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

DEFAULT_BAML_CLIENT = "OpenAI"
DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "OBJECT"]


class BamlEntityExtractor:
    """Entity extractor powered by BAML with multi-provider LLM support.

    Satisfies the EntityExtractor protocol. Uses BAML's generated client
    for type-safe structured extraction with automatic retries and
    fallback chains.

    Provider selection:
        - Set ``client_name`` to choose: "OpenAI", "Anthropic", "Gemini", "Resilient"
        - Pass a ``ClientRegistry`` for runtime provider switching
    """

    def __init__(
        self,
        *,
        client_name: str = DEFAULT_BAML_CLIENT,
        entity_types: list[str] | None = None,
        extract_relations: bool = True,
        extract_preferences: bool = True,
        client_registry: Any | None = None,
    ):
        self._client_name = client_name
        self._entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self._extract_relations = extract_relations
        self._extract_preferences = extract_preferences
        self._client_registry = client_registry
        self._baml_options: dict[str, Any] = {}

        if client_registry:
            self._baml_options["client_registry"] = client_registry
        elif client_name != DEFAULT_BAML_CLIENT:
            try:
                from baml_py import ClientRegistry

                registry = ClientRegistry()
                registry.set_primary(client_name)
                self._baml_options["client_registry"] = registry
            except ImportError:
                logger.warning("baml-py not installed, client_name override ignored")

    @property
    def name(self) -> str:
        return f"BamlEntityExtractor({self._client_name})"

    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        extract_relations: bool = True,
        extract_preferences: bool = True,
    ) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult(source_text=text)

        try:
            from neo4j_agent_memory.baml_client.async_client import b
        except ImportError:
            raise RuntimeError(
                "BAML client not generated. Run: uv run baml-cli generate"
            )

        types_to_use = entity_types or self._entity_types
        entity_types_str = ", ".join(types_to_use)

        try:
            result = await b.ExtractEntities(
                text=text,
                entity_types=entity_types_str,
                **(self._baml_options if self._baml_options else {}),
            )

            entities = [
                ExtractedEntity(
                    name=e.name,
                    type=e.type.value if hasattr(e.type, "value") else str(e.type),
                    subtype=e.subtype,
                    confidence=max(0.0, min(1.0, e.confidence)),
                    extractor="baml",
                )
                for e in result.entities
            ]

            include_relations = (
                extract_relations
                if extract_relations is not True  # caller passed explicit value
                else self._extract_relations
            )
            relations = []
            if include_relations:
                entity_names = {e.name.lower() for e in entities}
                relations = [
                    ExtractedRelation(
                        source=r.source,
                        target=r.target,
                        relation_type=r.relation_type.upper(),
                        confidence=max(0.0, min(1.0, r.confidence)),
                    )
                    for r in result.relations
                    if r.source.lower() in entity_names
                    and r.target.lower() in entity_names
                ]

            include_preferences = (
                extract_preferences
                if extract_preferences is not True
                else self._extract_preferences
            )
            preferences = []
            if include_preferences:
                preferences = [
                    ExtractedPreference(
                        category=p.category,
                        preference=p.preference,
                        context=p.context,
                        confidence=max(0.0, min(1.0, p.confidence)),
                    )
                    for p in result.preferences
                ]

            logger.debug(
                "BAML extracted %d entities, %d relations, %d preferences (client=%s)",
                len(entities),
                len(relations),
                len(preferences),
                self._client_name,
            )

            return ExtractionResult(
                entities=entities,
                relations=relations,
                preferences=preferences,
                source_text=text,
            )

        except Exception as e:
            from neo4j_agent_memory.core.exceptions import ExtractionError

            raise ExtractionError(
                f"BAML extraction failed ({type(e).__name__}): {e}"
            ) from e
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_baml_extractor.py tests/test_extraction_overlay.py -v`
Expected: all tests PASS

**Step 6: Commit**

```bash
git add src/neo4j_agent_memory/extraction/baml_extractor.py tests/conftest.py tests/test_baml_extractor.py
git commit -m "feat: add BamlEntityExtractor with multi-provider support

Implements EntityExtractor protocol using BAML generated client.
Supports OpenAI, Anthropic, Gemini, and Resilient (fallback chain).
Includes comprehensive tests with mocked BAML client [RFI-F4]."
```

---

## Task 5: Create Factory Extension and Config

**Files:**
- Create: `src/neo4j_agent_memory/extraction/baml_config.py`
- Create: `src/neo4j_agent_memory/extraction/factory_ext.py`
- Create: `tests/test_factory_ext.py`

**Step 1: Write the failing tests**

Create `tests/test_factory_ext.py`:

```python
"""Tests for the extended factory with BAML support."""

import os
import pytest
from unittest.mock import patch


class TestFactoryExtBamlRouting:
    """Verify factory routes to BAML when BAML_ENABLED is set."""

    def test_baml_enabled_creates_baml_extractor(self):
        with patch.dict(os.environ, {
            "NAM_EXTRACTION__BAML_ENABLED": "true",
            "NAM_EXTRACTION__BAML_CLIENT": "OpenAI",
        }):
            from neo4j_agent_memory.extraction.factory_ext import create_extractor
            from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor
            from neo4j_agent_memory.config.settings import ExtractionConfig

            config = ExtractionConfig()
            extractor = create_extractor(config)
            assert isinstance(extractor, BamlEntityExtractor)

    def test_baml_disabled_falls_through(self):
        with patch.dict(os.environ, {
            "NAM_EXTRACTION__BAML_ENABLED": "false",
        }, clear=False):
            # Remove BAML_ENABLED if set
            os.environ.pop("NAM_EXTRACTION__BAML_ENABLED", None)

            from neo4j_agent_memory.extraction.factory_ext import create_extractor
            from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor
            from neo4j_agent_memory.config.settings import ExtractionConfig

            config = ExtractionConfig()
            extractor = create_extractor(config)
            assert not isinstance(extractor, BamlEntityExtractor)

    def test_baml_client_env_var_respected(self):
        with patch.dict(os.environ, {
            "NAM_EXTRACTION__BAML_ENABLED": "true",
            "NAM_EXTRACTION__BAML_CLIENT": "Anthropic",
        }):
            from neo4j_agent_memory.extraction.factory_ext import create_extractor
            from neo4j_agent_memory.config.settings import ExtractionConfig

            config = ExtractionConfig()
            extractor = create_extractor(config)
            assert "Anthropic" in extractor.name

    def test_baml_enabled_overrides_any_extractor_type(self):
        """BAML_ENABLED=true works regardless of extractor_type [RFI-F1]."""
        with patch.dict(os.environ, {
            "NAM_EXTRACTION__BAML_ENABLED": "true",
            "NAM_EXTRACTION__BAML_CLIENT": "OpenAI",
        }):
            from neo4j_agent_memory.extraction.factory_ext import create_extractor
            from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor
            from neo4j_agent_memory.config.settings import ExtractionConfig, ExtractorType

            # Even with NONE type, BAML overrides
            config = ExtractionConfig(extractor_type=ExtractorType.NONE)
            extractor = create_extractor(config)
            assert isinstance(extractor, BamlEntityExtractor)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_factory_ext.py -v`
Expected: FAIL — modules don't exist yet

**Step 3: Create baml_config.py**

Create `src/neo4j_agent_memory/extraction/baml_config.py`:

```python
"""BAML extraction configuration constants."""

BAML_EXTRACTOR_TYPE = "baml"
DEFAULT_BAML_CLIENT = "OpenAI"
```

**Step 4: Create factory_ext.py**

Create `src/neo4j_agent_memory/extraction/factory_ext.py`:

```python
"""Extended factory that adds BAML extractor support.

Wraps the base package's create_extractor() and intercepts calls when
NAM_EXTRACTION__BAML_ENABLED=true. The BAML_ENABLED env var overrides
regardless of the configured extractor_type [RFI-F1].
"""

import logging
import os

from neo4j_agent_memory.extraction.base import EntityExtractor
from neo4j_agent_memory.extraction.factory import (
    create_extractor as _base_create_extractor,
)
from neo4j_agent_memory.extraction.baml_config import DEFAULT_BAML_CLIENT

logger = logging.getLogger(__name__)


def _is_baml_enabled() -> bool:
    """Check if BAML extraction is enabled via env var."""
    return os.environ.get(
        "NAM_EXTRACTION__BAML_ENABLED", ""
    ).lower() in ("true", "1", "yes")


def create_baml_extractor(
    extraction_config, schema_config=None
) -> EntityExtractor:
    """Create a BAML entity extractor from config."""
    from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

    client_name = os.environ.get(
        "NAM_EXTRACTION__BAML_CLIENT", DEFAULT_BAML_CLIENT
    )

    entity_types = extraction_config.entity_types
    if schema_config and hasattr(schema_config, "entity_types") and schema_config.entity_types:
        entity_types = schema_config.entity_types

    logger.info(
        "BAML extraction enabled (overriding extractor_type=%s, client=%s)",
        extraction_config.extractor_type,
        client_name,
    )

    return BamlEntityExtractor(
        client_name=client_name,
        entity_types=entity_types,
        extract_relations=extraction_config.extract_relations,
        extract_preferences=extraction_config.extract_preferences,
    )


def create_extractor(
    extraction_config, schema_config=None, llm_config=None
) -> EntityExtractor:
    """Extended factory — routes to BAML when enabled, else base factory."""
    if _is_baml_enabled():
        return create_baml_extractor(extraction_config, schema_config)

    return _base_create_extractor(extraction_config, schema_config, llm_config)
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_factory_ext.py tests/test_extraction_overlay.py -v`
Expected: all tests PASS

**Step 6: Commit**

```bash
git add src/neo4j_agent_memory/extraction/baml_config.py src/neo4j_agent_memory/extraction/factory_ext.py tests/test_factory_ext.py
git commit -m "feat: factory extension routes to BAML when BAML_ENABLED=true

BAML_ENABLED overrides regardless of extractor_type [RFI-F1].
BAML_CLIENT env var selects the provider (OpenAI, Anthropic, etc.)."
```

---

## Task 6: Wire Factory Patch Into Server Lifespan

**Files:**
- Modify: `src/neo4j_agent_memory/mcp/server.py:50-57`

**Step 1: Write the failing test**

Create `tests/test_server_baml_patch.py`:

```python
"""Tests for server lifespan BAML factory patching [RFI-R1]."""

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def test_factory_module_is_patched_when_baml_enabled():
    """Verify the factory module attribute gets replaced."""
    with patch.dict(os.environ, {"NAM_EXTRACTION__BAML_ENABLED": "true"}):
        import neo4j_agent_memory.extraction.factory as factory_mod
        from neo4j_agent_memory.extraction.factory_ext import (
            create_extractor as ext_create,
        )

        # Simulate what the server lifespan does
        original = factory_mod.create_extractor
        factory_mod.create_extractor = ext_create

        assert factory_mod.create_extractor is ext_create
        assert factory_mod.create_extractor.__module__ == "neo4j_agent_memory.extraction.factory_ext"

        # Restore
        factory_mod.create_extractor = original
```

**Step 2: Run test to verify it fails (or passes — this is a unit test of the mechanism)**

Run: `uv run pytest tests/test_server_baml_patch.py -v`
Expected: PASS (this tests the mechanism, not the integration)

**Step 3: Modify server.py lifespan**

In `src/neo4j_agent_memory/mcp/server.py`, replace the lifespan function (lines 51-57):

```python
            @asynccontextmanager
            async def lifespan(server: FastMCP):  # noqa: E303
                """Manage MemoryClient lifecycle for the MCP server."""
                import os

                from neo4j_agent_memory import MemoryClient as _MemoryClient

                # Patch factory to support BAML extraction [RFI-R1]
                import neo4j_agent_memory.extraction.factory as _factory_mod
                from neo4j_agent_memory.extraction.factory_ext import (
                    create_extractor as _ext_create_extractor,
                )

                _factory_mod.create_extractor = _ext_create_extractor
                logger.info("Extraction factory patched with BAML support")

                async with _MemoryClient(settings) as client:
                    # Verify BAML patch took effect [RFI-R1]
                    baml_enabled = os.environ.get(
                        "NAM_EXTRACTION__BAML_ENABLED", ""
                    ).lower() in ("true", "1", "yes")
                    if baml_enabled:
                        _ext = getattr(client, "_extractor", None)
                        _ext_name = getattr(_ext, "name", str(type(_ext)))
                        if _ext and "Baml" in str(_ext_name):
                            logger.info("BAML extraction active: %s", _ext_name)
                        else:
                            logger.error(
                                "BAML enabled but extractor is %s — patch may have failed",
                                _ext_name,
                            )

                    yield {"client": client}
```

**Step 4: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/neo4j_agent_memory/mcp/server.py tests/test_server_baml_patch.py
git commit -m "feat: patch extraction factory in server lifespan [RFI-R1]

Monkey-patches factory at startup to route BAML_ENABLED=true to
BamlEntityExtractor. Includes post-connect verification logging
to catch patch failures early."
```

---

## Task 7: Run Full Test Suite and Final Verification

**Files:**
- No new files

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: all tests PASS

**Step 2: Verify imports work end-to-end**

Run:
```bash
uv run python -c "
from neo4j_agent_memory.extraction.base import EntityExtractor, ExtractionResult
from neo4j_agent_memory.extraction.factory import create_extractor
from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor
from neo4j_agent_memory.extraction.factory_ext import create_extractor as ext_create
from neo4j_agent_memory.baml_client.async_client import b
print('All imports OK')
print(f'BamlEntityExtractor satisfies protocol: {isinstance(BamlEntityExtractor(), EntityExtractor)}')
"
```
Expected: prints confirmation with no errors

**Step 3: Verify BAML generation is reproducible**

Run:
```bash
uv run baml-cli generate
git diff src/neo4j_agent_memory/baml_client/
```
Expected: no changes (generated code matches committed code)

**Step 4: Commit if any test infrastructure changes were needed**

```bash
git status
# Only commit if there are changes
```

---

## Task 8: Documentation

**Files:**
- Create or modify: `README.md`

**Step 1: Add BAML section to README**

Add the following section to the project README (create if it doesn't exist):

```markdown
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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add BAML extraction setup and configuration guide"
```

---

## File Summary

### New Files Created

| File | Task | Purpose |
|------|------|---------|
| `baml_src/generators.baml` | 2 | BAML code generation config |
| `baml_src/clients.baml` | 2 | LLM provider definitions |
| `baml_src/extraction.baml` | 2 | Entity extraction types and function |
| `src/neo4j_agent_memory/baml_client/` | 2 | Auto-generated BAML Python client |
| `src/neo4j_agent_memory/extraction/__init__.py` | 3 | Overlay with `__path__` extension [RFI-I1] |
| `src/neo4j_agent_memory/extraction/baml_extractor.py` | 4 | BamlEntityExtractor class |
| `src/neo4j_agent_memory/extraction/baml_config.py` | 5 | Config constants |
| `src/neo4j_agent_memory/extraction/factory_ext.py` | 5 | Extended factory with BAML routing |
| `tests/conftest.py` | 4 | BAML mock fixtures [RFI-F4] |
| `tests/test_extraction_overlay.py` | 3 | Overlay import tests [RFI-I1] |
| `tests/test_baml_extractor.py` | 4 | Extractor unit tests |
| `tests/test_factory_ext.py` | 5 | Factory routing tests |
| `tests/test_server_baml_patch.py` | 6 | Patch verification tests [RFI-R1] |

### Modified Files

| File | Task | Change |
|------|------|--------|
| `pyproject.toml` | 1 | Add `baml-py` dependency |
| `src/neo4j_agent_memory/mcp/server.py` | 6 | Factory patch in lifespan |
| `README.md` | 8 | BAML documentation |
