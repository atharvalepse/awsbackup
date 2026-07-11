"""OpenAI Codex MCP connector — per-user install artifact generators.

Codex (and Claude Desktop) speak MCP over **stdio**; GML exposes a remote
**streamable-HTTP** endpoint at ``/mcp``. The vendored ``codex_bridge.js`` is a
zero-dependency Node bridge that proxies one to the other, authenticating with
``Authorization: Bearer $GML_TOKEN`` — the same per-user ``gml_…`` key the rest
of the install surface mints (see :func:`orchestration.api_routes`).

This module turns a freshly-issued key into the things a user double-clicks:

* :func:`windows_installer` — a ``.cmd`` that drops the bridge into
  ``~/.codex/akhrot-memory/index.js`` and merges an idempotent
  ``[mcp_servers.akhrot-memory]`` block into ``~/.codex/config.toml``.
* :func:`unix_installer` — the macOS/Linux ``.command`` equivalent.
* :func:`plugin_zip` — a Codex ``/plugins`` bundle (``.mcp.json`` + bridge).
* :func:`config_toml` — the manual snippet, for docs / copy-paste.

The installer scripts are ported verbatim from the upstream generator
(``server.mjs`` in Sakshxm-py/akhrot-codex-mcp); only the bridge bytes, the MCP
URL, and the per-user token are substituted in. Keeping them byte-faithful to
the proven originals is deliberate — the PowerShell/bash escaping is fiddly and
already field-tested. Substitution is positional (``str.replace`` of unique
sentinels), never f-strings, so a brace or backslash in the template can't be
misread.

These functions are pure (token in → bytes/str out) and import nothing from
FastAPI, so they unit-test without a server.
"""
from __future__ import annotations

import base64
import io
import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path

# Server name as it appears in the client's MCP config + tool namespace. Must
# match what the bridge announces in its synthesized `initialize` response.
SERVER_NAME = "akhrot-memory"

# Where the bridge lands on the user's machine (documentation + the absolute
# path the installers write into config.toml). The installers compute the real
# path at run time; this constant is only for the manual snippet.
_INSTALL_HINT = "~/.codex/akhrot-memory/index.js"


def _bridge_path() -> Path:
    """Filesystem path to the vendored bridge JS.

    Override with ``AKHROT_CODEX_BRIDGE`` if the deploy keeps it elsewhere;
    defaults to the copy shipped next to this module.
    """
    override = os.environ.get("AKHROT_CODEX_BRIDGE", "").strip()
    if override:
        return Path(override)
    return Path(__file__).with_name("codex_bridge.js")


@lru_cache(maxsize=1)
def _bridge_cached(path_str: str, mtime: float) -> str:
    # Keyed on (path, mtime) so an updated file is re-read, but the common
    # case (unchanged file) is served from cache instead of hitting disk on
    # every download.
    return Path(path_str).read_text(encoding="utf-8")


def bridge_source() -> str:
    """Return the vendored bridge JavaScript source.

    Raises ``FileNotFoundError`` if the vendored file is missing — a deploy
    packaging error we want loud, not a silently-empty installer.
    """
    p = _bridge_path()
    return _bridge_cached(str(p), p.stat().st_mtime)


def _bridge_b64() -> str:
    return base64.b64encode(bridge_source().encode("utf-8")).decode("ascii")


def _validate(token: str, mcp_url: str) -> None:
    """Reject values that could break out of the installer string contexts.

    Issued keys are URL-safe (``gml_`` + base64url) and the MCP URL is an
    operator-set env var, so this never trips in practice. It exists so a
    malformed input fails closed instead of injecting shell/TOML — the
    installers embed both values inside quoted literals.
    """
    for label, val in (("token", token), ("mcp_url", mcp_url)):
        if not val:
            raise ValueError(f"{label} must be non-empty")
        if any(c in val for c in '"\r\n') or "\\" in val:
            raise ValueError(f"{label} contains characters unsafe to embed")


# ---------------------------------------------------------------------------
# config.toml block (shared shape; also the manual copy-paste snippet)
# ---------------------------------------------------------------------------
def config_toml(token: str, mcp_url: str, *, node: str = "node",
                index_path: str = _INSTALL_HINT) -> str:
    """The ``[mcp_servers.akhrot-memory]`` block for ``~/.codex/config.toml``.

    Used for the docs / manual path. ``index_path`` defaults to a placeholder
    because the real absolute path is only known on the user's machine (the
    installers fill it in there).
    """
    _validate(token, mcp_url)
    return (
        "[mcp_servers.akhrot-memory]\n"
        f'command = "{node}"\n'
        f'args = ["{index_path}"]\n'
        f'env = {{ GML_MCP_URL = "{mcp_url}", GML_TOKEN = "{token}" }}\n'
    )


# ---------------------------------------------------------------------------
# Windows: a .cmd that double-clicks to run an embedded PowerShell installer
# (base64 -EncodedCommand to dodge all quoting issues). Ported verbatim from
# the upstream genWindowsInstaller; only bridge bytes / URL / token differ.
# ---------------------------------------------------------------------------
def windows_installer(token: str, mcp_url: str) -> bytes:
    _validate(token, mcp_url)
    ps_lines = [
        "$ErrorActionPreference = 'Stop'",
        "try {",
        "  $codex = Join-Path $env:USERPROFILE '.codex'",
        "  $dest  = Join-Path $codex 'akhrot-memory'",
        "  New-Item -ItemType Directory -Force -Path $dest | Out-Null",
        "  $indexPath = Join-Path $dest 'index.js'",
        "  [IO.File]::WriteAllBytes($indexPath, [Convert]::FromBase64String('"
        + _bridge_b64() + "'))",
        "  $node = (Get-Command node -ErrorAction SilentlyContinue).Source",
        "  if (-not $node) { $node = 'node' }",
        "  $cfg = Join-Path $codex 'config.toml'",
        "  if (Test-Path $cfg) { $txt = [IO.File]::ReadAllText($cfg) } else { $txt = '' }",
        "  # idempotent: drop any prior akhrot-memory block (handles token rotation)",
        r"  $txt = [regex]::Replace($txt, '(?ms)^\[mcp_servers\.akhrot-memory\].*?(?=^\[|\Z)', '')",
        "  $txt = $txt.TrimEnd()",
        r"  $en = ($node -replace '\\','\\'); $ei = ($indexPath -replace '\\','\\')",
        "  $lines = @(",
        "    '[mcp_servers.akhrot-memory]',",
        "    ('command = \"' + $en + '\"'),",
        "    ('args = [\"' + $ei + '\"]'),",
        "    'env = { GML_MCP_URL = \"" + mcp_url + "\", GML_TOKEN = \"" + token + "\" }'",
        "  )",
        "  $block = ($lines -join [Environment]::NewLine)",
        "  if ($txt.Length -gt 0) { $out = $txt + [Environment]::NewLine + [Environment]::NewLine + $block } else { $out = $block }",
        "  [IO.File]::WriteAllText($cfg, $out + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))",
        "  $hasNode = [bool]((Get-Command node -ErrorAction SilentlyContinue))",
        "  Write-Host ''",
        "  Write-Host '   Akhrot GML Memory connected to Codex.' -ForegroundColor Green",
        "  Write-Host ('   config : ' + $cfg)",
        "  if (-not $hasNode) { Write-Host '   WARNING: Node 18+ was not found on PATH - install it so Codex can launch the bridge.' -ForegroundColor Yellow }",
        "  Write-Host '   Say \"use akhrots memory to save ...\" in Codex to use it. Restart Codex if it was open.'",
        "  Write-Host ''",
        "} catch {",
        "  Write-Host ('   Install failed: ' + $_.Exception.Message) -ForegroundColor Red",
        "}",
    ]
    # PowerShell -EncodedCommand expects UTF-16LE, base64.
    encoded = base64.b64encode(
        "\n".join(ps_lines).encode("utf-16-le")
    ).decode("ascii")
    cmd = "\r\n".join([
        "@echo off",
        "title Install Akhrot GML Memory for Codex",
        "echo Connecting Akhrot GML Memory to Codex...",
        f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}",
        "echo.",
        "pause",
        "",
    ])
    return cmd.encode("utf-8")


# ---------------------------------------------------------------------------
# macOS / Linux: a .command bash script (double-clickable in Finder). Ported
# verbatim from the upstream genUnixInstaller. The bridge rides in a base64
# heredoc; $NODE/$DEST expand on the user's machine, the URL/token are baked.
# ---------------------------------------------------------------------------
_UNIX_TEMPLATE = r"""#!/bin/bash
set -e
CODEX="$HOME/.codex"
DEST="$CODEX/akhrot-memory"
mkdir -p "$DEST"
base64 --decode > "$DEST/index.js" <<'AKHROT_BRIDGE_B64'
@@BRIDGE_B64@@
AKHROT_BRIDGE_B64
NODE="$(command -v node || echo node)"
CFG="$CODEX/config.toml"
TMP="$(mktemp)"
# idempotent: drop any prior akhrot-memory block (handles token rotation)
if [ -f "$CFG" ]; then
  awk 'BEGIN{skip=0}
       /^\[mcp_servers\.akhrot-memory\]/{skip=1;next}
       /^\[/{if(skip){skip=0}}
       skip==0{print}' "$CFG" > "$TMP"
else
  : > "$TMP"
fi
cat >> "$TMP" <<EOF

[mcp_servers.akhrot-memory]
command = "$NODE"
args = ["$DEST/index.js"]
env = { GML_MCP_URL = "@@MCP_URL@@", GML_TOKEN = "@@TOKEN@@" }
EOF
mv "$TMP" "$CFG"
echo ""
echo "   Akhrot GML Memory connected to Codex."
echo "   config: $CFG"
command -v node >/dev/null || echo "   WARNING: Node 18+ not found on PATH - install it so Codex can launch the bridge."
echo "   Say 'use akhrots memory to save ...' in Codex to use it. Restart Codex if it was open."
echo ""
"""


def unix_installer(token: str, mcp_url: str) -> bytes:
    _validate(token, mcp_url)
    sh = (
        _UNIX_TEMPLATE
        .replace("@@BRIDGE_B64@@", _bridge_b64())
        .replace("@@MCP_URL@@", mcp_url)
        .replace("@@TOKEN@@", token)
    )
    return sh.encode("utf-8")


# ---------------------------------------------------------------------------
# Codex /plugins bundle: .codex-plugin/plugin.json + .mcp.json + bridge.
# The token lives ONLY in .mcp.json's env — never in plugin.json or the bridge.
# ---------------------------------------------------------------------------
def plugin_zip(token: str, mcp_url: str) -> bytes:
    _validate(token, mcp_url)
    plugin_json = {
        "name": SERVER_NAME,
        "version": "1.0.0",
        "description": "Akhrot GML long-term memory for Codex (query / ingest).",
        "author": "Akhrots",
    }
    # stdio via the bundled gating bridge. Codex launches it from the plugin
    # dir, so the relative path resolves.
    mcp_json = {
        SERVER_NAME: {
            "command": "node",
            "args": ["server/index.js"],
            "env": {"GML_MCP_URL": mcp_url, "GML_TOKEN": token},
        }
    }
    readme = "\n".join([
        "# Akhrot GML Memory — Codex plugin",
        "",
        "Install from inside Codex:",
        "",
        "1. Unzip this anywhere (or point Codex at it).",
        "2. In Codex run `/plugins`, choose **Install plugin**, complete any prompt,",
        "   then start a new thread.",
        "",
        'Usage: say "use akhrots memory to save ..." / "use akhrots memory to recall ...".',
        "The bundled bridge gates the tools to that phrase. Requires Node 18+ on PATH.",
        "The bundled token is per-user — re-download from the dashboard to rotate.",
        "",
    ])

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(".codex-plugin/plugin.json", json.dumps(plugin_json, indent=2))
        z.writestr(".mcp.json", json.dumps(mcp_json, indent=2))
        z.writestr("server/index.js", bridge_source())
        z.writestr("README.md", readme)
    return buf.getvalue()
