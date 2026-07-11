"""Relay-side aggregation: the virtual ``*`` target.

When a host opens a session against ``?server=*``, the relay presents **all** of
that user's connected servers as a single MCP endpoint. This module multiplexes
one host session over N backend sub-sessions:

* ``initialize`` fans out to every backend; capabilities are unioned and a
  synthetic ``serverInfo`` is returned.
* ``tools/list`` / ``prompts/list`` / ``resources/list`` are fanned out and
  merged, with tool/prompt names **namespaced** as ``<server>__<name>`` so they
  never collide. The relay remembers the owner of each name.
* ``tools/call`` / ``prompts/get`` / ``resources/read`` are routed to the owning
  backend (the namespace prefix is stripped first).
* Server-initiated requests (e.g. sampling) get a rewritten id so the host's
  reply can be routed back to the originating backend.

Because one host session now spans several backends, ids that fan out are
rewritten (``agg:N``) and server-initiated ids are rewritten (``si:N``). Single
target requests keep the host's original id — only one backend ever sees it.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import PROTOCOL_VERSION, __version__
from .registry import ENVELOPE_VERSION, RelayState, ServerConnection, ServerUnavailable

Json = dict[str, Any]


@dataclass
class BackendLeg:
    """One backend sub-session belonging to an AggregateSession."""

    id: str  # sub-session id used as the envelope 'session' to the connector
    user_id: str
    server_name: str
    conn: ServerConnection
    parent: "AggregateSession"

    async def send(self, payload: Json) -> None:
        # "user" rides every message (same as registry._send_to_server): if
        # the connector restarted and lost its session map, its feed()
        # fallback re-opens this leg's child — without the user the child
        # would respawn unscoped and touch the wrong tenant's memories.
        await self.conn.push(
            {"v": ENVELOPE_VERSION, "type": "message", "session": self.id,
             "payload": payload, "user": self.user_id}
        )


@dataclass
class FanoutGroup:
    host_id: Any
    mode: str  # init | tools | prompts | resources | templates | first_success
    remaining: set = field(default_factory=set)
    id_to_server: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)  # server_name -> response payload


class AggregateSession:
    """A host session bound to the virtual ``*`` target, fanning across backends."""

    server_name = "*"

    def __init__(self, user_id: str):
        self.id = secrets.token_urlsafe(18)
        self.user_id = user_id
        self.standby: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.pending: dict[str, asyncio.Future] = {}
        self.backends: dict[str, BackendLeg] = {}
        self.created_at = time.time()
        self.last_seen = time.time()
        self._counter = 0
        self._client_protocol: Optional[str] = None
        self._groups: dict[str, FanoutGroup] = {}  # rewritten fan-out id -> group
        self._server_initiated: dict[str, tuple[str, Any]] = {}
        self._tool_owner: dict[str, tuple[str, str]] = {}
        self._prompt_owner: dict[str, tuple[str, str]] = {}
        self._resource_owner: dict[str, str] = {}

    # -- shared session interface (mirrors ClientSession) -------------------
    def register_pending(self, msg_id: Any) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending[str(msg_id)] = fut
        return fut

    def cancel_all_pending(self) -> None:
        for fut in self.pending.values():
            if not fut.done():
                fut.cancel()
        self.pending.clear()

    def _next(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}:{self._counter}"

    # -- host -> backends ---------------------------------------------------
    async def send_from_host(self, msg: Json, state: RelayState) -> None:
        self.last_seen = time.time()
        method = msg.get("method")
        if method is None:  # a response from the host to a server-initiated request
            await self._route_host_response(msg)
            return
        if msg.get("id") is None:  # notification
            await self._broadcast(msg)
            return

        mid = msg["id"]
        params = msg.get("params")
        if method == "initialize":
            self._client_protocol = (params or {}).get("protocolVersion")
            await self._fanout(state, mid, "initialize", params, "init")
        elif method == "ping":
            self._resolve(mid, {})
        elif method == "tools/list":
            await self._fanout(state, mid, method, params, "tools")
        elif method == "prompts/list":
            await self._fanout(state, mid, method, params, "prompts")
        elif method == "resources/list":
            await self._fanout(state, mid, method, params, "resources")
        elif method == "resources/templates/list":
            await self._fanout(state, mid, method, params, "templates")
        elif method == "tools/call":
            await self._route_named(mid, msg, self._tool_owner)
        elif method == "prompts/get":
            await self._route_named(mid, msg, self._prompt_owner)
        elif method in ("resources/read", "resources/subscribe", "resources/unsubscribe"):
            await self._route_resource(state, mid, msg)
        else:
            self._resolve_error(mid, -32601,
                                f"method '{method}' is not supported in aggregate (server='*') mode")

    async def _fanout(self, state: RelayState, host_id: Any, method: str,
                      params: Any, mode: str) -> None:
        group = FanoutGroup(host_id=host_id, mode=mode)
        for name, leg in list(self.backends.items()):
            if not leg.conn.connected:
                group.results[name] = {"error": {"code": -32002, "message": "server disconnected"}}
                continue
            rid = self._next("agg")
            group.remaining.add(rid)
            group.id_to_server[rid] = name
            self._groups[rid] = group
            payload: Json = {"jsonrpc": "2.0", "id": rid, "method": method}
            if params is not None:
                payload["params"] = params
            await leg.send(payload)
        if not group.remaining:  # all backends were disconnected
            self._complete(group, state)

    async def _route_named(self, host_id: Any, msg: Json, owner: dict) -> None:
        params = msg.get("params") or {}
        name = params.get("name")
        if not name or "__" not in name:
            self._resolve_error(host_id, -32602,
                                f"name {name!r} must be namespaced as '<server>__<name>'")
            return
        if name in owner:
            server, original = owner[name]
        else:  # cache miss — fall back to splitting on the first separator
            server, _, original = name.partition("__")
        leg = self.backends.get(server)
        if leg is None or not leg.conn.connected:
            self._resolve_error(host_id, -32002, f"server {server!r} unavailable")
            return
        out = dict(msg)
        out["params"] = {**params, "name": original}
        await leg.send(out)  # keep the host id — only this backend sees it

    async def _route_resource(self, state: RelayState, host_id: Any, msg: Json) -> None:
        params = msg.get("params") or {}
        uri = params.get("uri")
        server = self._resource_owner.get(uri)
        if server and server in self.backends and self.backends[server].conn.connected:
            await self.backends[server].send(msg)  # keep host id
            return
        if msg.get("method") == "resources/read":
            await self._fanout(state, host_id, "resources/read", params, "first_success")
        else:
            self._resolve_error(host_id, -32002, f"no server owns resource {uri!r}")

    async def _broadcast(self, msg: Json) -> None:
        for leg in list(self.backends.values()):
            if leg.conn.connected:
                await leg.send(msg)

    async def _route_host_response(self, msg: Json) -> None:
        entry = self._server_initiated.pop(str(msg.get("id")), None)
        if entry is None:
            return
        server_name, original_id = entry
        leg = self.backends.get(server_name)
        if leg is not None:
            await leg.send({**msg, "id": original_id})

    # -- backends -> host ---------------------------------------------------
    def on_backend_message(self, server_name: str, payload: Json, state: RelayState) -> None:
        self.last_seen = time.time()
        if "method" in payload:
            if payload.get("id") is not None:  # server-initiated request
                self._forward_server_request(server_name, payload)
            else:  # server notification
                self._to_host(payload)
            return
        # response
        rid = str(payload.get("id"))
        group = self._groups.pop(rid, None)
        if group is not None:
            group.results[group.id_to_server[rid]] = payload
            group.remaining.discard(rid)
            if not group.remaining:
                self._complete(group, state)
            return
        if rid in self.pending:  # single-target reply (carries the host id)
            self._resolve_payload(payload)

    def _forward_server_request(self, server_name: str, payload: Json) -> None:
        sid = self._next("si")
        self._server_initiated[sid] = (server_name, payload.get("id"))
        self._to_host({**payload, "id": sid})

    def _to_host(self, payload: Json) -> None:
        try:
            self.standby.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                self.standby.get_nowait()
                self.standby.put_nowait(payload)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    # -- merge ----------------------------------------------------------------
    def _complete(self, group: FanoutGroup, state: RelayState) -> None:
        mode = group.mode
        if mode == "init":
            self._complete_init(group, state)
        elif mode == "tools":
            self._tool_owner = {}
            tools = []
            for server, res in group.results.items():
                for t in (res.get("result") or {}).get("tools", []):
                    pref = f"{server}__{t.get('name')}"
                    self._tool_owner[pref] = (server, t.get("name"))
                    tools.append({**t, "name": pref})
            self._resolve(group.host_id, {"tools": tools})
        elif mode == "prompts":
            self._prompt_owner = {}
            prompts = []
            for server, res in group.results.items():
                for p in (res.get("result") or {}).get("prompts", []):
                    pref = f"{server}__{p.get('name')}"
                    self._prompt_owner[pref] = (server, p.get("name"))
                    prompts.append({**p, "name": pref})
            self._resolve(group.host_id, {"prompts": prompts})
        elif mode == "resources":
            resources = []
            for server, res in group.results.items():
                for r in (res.get("result") or {}).get("resources", []):
                    if r.get("uri"):
                        self._resource_owner[r["uri"]] = server
                    resources.append(r)
            self._resolve(group.host_id, {"resources": resources})
        elif mode == "templates":
            tpls = []
            for res in group.results.values():
                tpls.extend((res.get("result") or {}).get("resourceTemplates", []))
            self._resolve(group.host_id, {"resourceTemplates": tpls})
        elif mode == "first_success":
            for res in group.results.values():
                if "result" in res:
                    self._resolve_payload({**res, "id": group.host_id})
                    return
            err = next((r["error"] for r in group.results.values() if "error" in r),
                       {"code": -32002, "message": "no server responded"})
            self._resolve_error(group.host_id, err.get("code", -32000), err.get("message", "error"))

    def _complete_init(self, group: FanoutGroup, state: RelayState) -> None:
        caps: Json = {}
        names: list[str] = []
        for server, res in group.results.items():
            if "result" not in res:
                self._drop_leg(server, state)
                continue
            names.append(server)
            for k, v in (res["result"].get("capabilities") or {}).items():
                if isinstance(v, dict):
                    cur = caps.setdefault(k, {})
                    if isinstance(cur, dict):
                        cur.update(v)
                else:
                    caps.setdefault(k, v)
        if not names:
            self._resolve_error(group.host_id, -32002, "no servers could initialize")
            return
        self._resolve(group.host_id, {
            "protocolVersion": self._client_protocol or PROTOCOL_VERSION,
            "capabilities": caps,
            "serverInfo": {"name": "mcp-relay-aggregate", "version": __version__},
            "instructions": (
                "Aggregated MCP servers: " + ", ".join(sorted(names))
                + ". Tool and prompt names are namespaced as '<server>__<name>'."
            ),
        })

    def _drop_leg(self, server_name: str, state: RelayState) -> None:
        leg = self.backends.pop(server_name, None)
        if leg is None:
            return
        state.detach_session(leg.id)
        leg.conn.sessions.discard(leg.id)

    # -- resolution helpers -------------------------------------------------
    def _resolve(self, host_id: Any, result: Json) -> None:
        fut = self.pending.pop(str(host_id), None)
        if fut is not None and not fut.done():
            fut.set_result({"jsonrpc": "2.0", "id": host_id, "result": result})

    def _resolve_error(self, host_id: Any, code: int, message: str) -> None:
        fut = self.pending.pop(str(host_id), None)
        if fut is not None and not fut.done():
            fut.set_result({"jsonrpc": "2.0", "id": host_id, "error": {"code": code, "message": message}})

    def _resolve_payload(self, payload: Json) -> None:
        fut = self.pending.pop(str(payload.get("id")), None)
        if fut is not None and not fut.done():
            fut.set_result(payload)

    # -- teardown -----------------------------------------------------------
    async def close(self, state: RelayState) -> None:
        self.cancel_all_pending()
        for leg in list(self.backends.values()):
            state.detach_session(leg.id)
            leg.conn.sessions.discard(leg.id)
            if leg.conn.connected:
                try:
                    leg.conn.downlink.put_nowait(
                        {"v": ENVELOPE_VERSION, "type": "close", "session": leg.id}
                    )
                except asyncio.QueueFull:
                    pass
        state.detach_session(self.id)


async def open_aggregate(state: RelayState, user_id: str) -> AggregateSession:
    """Create an aggregate session spanning all of the user's connected servers."""
    servers = state.connected_servers(user_id)
    if not servers:
        raise ServerUnavailable(user_id, "*", [])
    agg = AggregateSession(user_id)
    for conn in servers:
        sub = secrets.token_urlsafe(18)
        leg = BackendLeg(id=sub, user_id=user_id, server_name=conn.name, conn=conn, parent=agg)
        agg.backends[conn.name] = leg
        state.attach_session(sub, leg)
        conn.sessions.add(sub)
        # "user" rides the open envelope (same as registry._send_to_server)
        # so the connector scopes the spawned MCP child via GML_MCP_USER.
        await conn.push({"v": ENVELOPE_VERSION, "type": "open", "session": sub,
                         "user": user_id})
    state.attach_session(agg.id, agg)
    return agg
