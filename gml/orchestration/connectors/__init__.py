"""Server-side connector packaging.

Helpers that bake a per-user MCP key into the install artifacts served by the
``/api/me/install/*`` surface in :mod:`orchestration.api_routes`. Currently:

* :mod:`orchestration.connectors.codex` — OpenAI Codex (and Claude Desktop)
  one-click installers + plugin bundle, built around the vendored stdio↔HTTP
  bridge (``codex_bridge.js``).
"""
