"""mcp-relay: a multi-user broker for MCP hosts and servers.

Both MCP hosts (clients) and MCP servers dial *into* the relay over HTTP.
The relay pairs them per-user and routes JSON-RPC traffic in both directions.
"""

__version__ = "0.1.0"

PROTOCOL_VERSION = "2024-11-05"
