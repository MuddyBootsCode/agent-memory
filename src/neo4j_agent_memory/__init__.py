"""Overlay package for neo4j_agent_memory.

Extends the installed neo4j-agent-memory package with a custom FastMCP-based
MCP server implementation.  Our ``mcp/`` subpackage takes priority over the
one shipped with the base package, while all other subpackages (config, core,
memory, graph, etc.) pass through to the installed version.
"""

import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# 1. Locate the *installed* neo4j_agent_memory package in site-packages so we
#    can extend __path__ and re-export its symbols.
# ---------------------------------------------------------------------------
_overlay_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_overlay_dir)

_installed_dir = None
for _entry in sys.path:
    # Skip our own src/ directory (compare absolute paths for safety)
    if os.path.normpath(os.path.abspath(_entry)) == os.path.normpath(_src_dir):
        continue
    _candidate = os.path.join(_entry, "neo4j_agent_memory")
    if (
        os.path.isdir(_candidate)
        and os.path.isfile(os.path.join(_candidate, "__init__.py"))
        and os.path.normpath(_candidate) != os.path.normpath(_overlay_dir)
    ):
        _installed_dir = _candidate
        break

# ---------------------------------------------------------------------------
# 2. Extend __path__: our overlay first, installed package second.
#    This means ``import neo4j_agent_memory.mcp`` resolves to *our* mcp/
#    directory, while ``import neo4j_agent_memory.config`` falls through to
#    the installed package.
# ---------------------------------------------------------------------------
if _installed_dir:
    __path__ = [_overlay_dir, _installed_dir]
else:
    __path__ = [_overlay_dir]

# ---------------------------------------------------------------------------
# 3. Execute the installed package's __init__.py in our namespace so that all
#    its exports (MemoryClient, MemorySettings, exceptions, …) are available
#    via ``from neo4j_agent_memory import ...`` as usual.
# ---------------------------------------------------------------------------
if _installed_dir:
    _init_file = os.path.join(_installed_dir, "__init__.py")
    if os.path.isfile(_init_file):
        _spec = importlib.util.spec_from_file_location(
            "neo4j_agent_memory._base_init",
            _init_file,
            submodule_search_locations=[_installed_dir],
        )
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            # Pull every public symbol into our namespace
            for _name in getattr(_mod, "__all__", []):
                if hasattr(_mod, _name):
                    globals()[_name] = getattr(_mod, _name)
            # Preserve __all__ from the base package
            if hasattr(_mod, "__all__"):
                __all__ = list(_mod.__all__)  # noqa: F841
            # Preserve __version__
            if hasattr(_mod, "__version__"):
                __version__ = _mod.__version__  # noqa: F841
