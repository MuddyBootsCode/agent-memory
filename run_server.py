#!/usr/bin/env python
"""Bootstrap script for the local FastMCP-based MCP server.

Adds src/ to sys.path so our overlay neo4j_agent_memory.mcp package
takes priority over the installed one, then delegates to the server's
main() entry point.
"""
import os
import sys

# Insert our src/ directory at the front of sys.path so our overlay
# neo4j_agent_memory package (with the FastMCP-based mcp/ subpackage)
# is found before the installed neo4j-agent-memory package.
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from neo4j_agent_memory.mcp.server import main

if __name__ == "__main__":
    main()
