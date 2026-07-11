"""Core relay state and routing.

The relay is a broker: MCP **servers** dial in via a connector and MCP **hosts**
(clients) dial in via Streamable HTTP. This module owns the in-memory tables that
pair them and route JSON-RPC messages.

Key invariants
--------------
* A host can only reach servers registered under the **same user**. That is the
  multi-tenant isolation boundary.
* Each host session gets its **own logical session** on the backing server (the
  connector spins up a dedicated server instance per session). Because no two
  hosts share a server session, JSON-RPC ids never collide and never need
  rewriting — correlation is exact.
* Everything flowing host -> server travels on the server's *downlink* queue
  (drained by the connector's GET SSE stream). Everything flowing server -> host
  arrives via the connector's uplink POST and is delivered to the host either by
  resolving a pending request future (POST response stream) or by enqueueing onto
  the host's *standby* queue (its GET SSE stream).
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

Json = dict[str, Any]

# Envelope protocol version for the relay<->connector legs.
ENVELOPE_VERSION = 1


def message_kind(msg: Json) -> str:
    """Classify a JSON-RPC message by shape: 'request', 'notification', or 'response'."""
    if "method" in msg:
        return "request" if ("id" in msg and msg.get("id") is not None) else "notification"
    return "response"


class ServerUnavailable(Exception):
    """Raised when a host targets a server that is not registered/connected."""

    def __init__(self, user_id: str, name: Optional[str], available: list[str]):
        self.user_id = user_id
        self.name = name
        self.available = available
        super().__init__(f"server {name!r} unavailable for user {user_id!r}")


@dataclass
class ServerConnection:
    """A connector that has registered a backing MCP server for a user."""

    user_id: str
    name: str
    session_token: str  # secret used by the connector to authenticate its legs
    info: Json = field(default_factory=dict)
    downlink: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1000))
    sessions: set[str] = field(default_factory=set)  # host session ids attached
    connected: bool = False  # is a GET stream currently draining the downlink?
    # When True the connector registered with `RELAY_REGISTER_GLOBAL=1` —
    # every authenticated user (not just user_id) is allowed to reach this
    # server. Without this flag the relay's per-user isolation makes a
    # shared service connector (the gml MCP server here) invisible to anyone
    # except the account whose token launched it, which is wrong: there is
    # exactly one production gml-connector and every signed-in user needs it.
    is_global: bool = False
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    async def push(self, envelope: Json) -> None:
        await self.downlink.put(envelope)


@dataclass
class ClientSession:
    """A host (MCP client) session, bound to exactly one of the user's servers."""

    id: str
    user_id: str
    server_name: str
    standby: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1000))
    # JSON-RPC id (stringified) -> future resolved when the server replies.
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def register_pending(self, msg_id: Any) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending[str(msg_id)] = fut
        return fut

    def resolve_pending(self, msg_id: Any, payload: Json) -> bool:
        fut = self.pending.pop(str(msg_id), None)
        if fut is not None and not fut.done():
            fut.set_result(payload)
            return True
        return False

    def cancel_all_pending(self) -> None:
        for fut in self.pending.values():
            if not fut.done():
                fut.cancel()
        self.pending.clear()


@dataclass
class QueuedRequest:
    """A host request waiting for its target server to connect."""

    session: ClientSession
    messages: list[tuple[Json, bool]]


class RelayState:
    """Single-process, in-memory routing tables. All access happens on one asyncio
    event loop, so no locking is required."""

    def __init__(self, auth):
        # `auth` is either a {token: user_id} dict (wrapped as StaticAuth) or an
        # auth backend exposing async user_for_token / startup / shutdown.
        if isinstance(auth, dict):
            from .auth import StaticAuth
            auth = StaticAuth(auth)
        self._auth = auth
        self._servers: dict[tuple[str, str], ServerConnection] = {}  # (user, name) -> conn
        self._server_by_token: dict[str, ServerConnection] = {}
        self._clients: dict[str, ClientSession] = {}
        self._queued: dict[tuple[str, str], list[QueuedRequest]] = {}

    # -- auth ---------------------------------------------------------------
    async def user_for_token(self, token: Optional[str]) -> Optional[str]:
        return await self._auth.user_for_token(token)

    async def startup(self) -> None:
        await self._auth.startup()

    async def shutdown(self) -> None:
        await self._auth.shutdown()

    # -- server (connector) side -------------------------------------------
    def register_server(
        self,
        user_id: str,
        name: str,
        info: Optional[Json],
        is_global: bool = False,
    ) -> ServerConnection:
        key = (user_id, name)
        old = self._servers.get(key)
        if old is not None:
            # Replace a stale registration (e.g. connector restarted).
            self._server_by_token.pop(old.session_token, None)
            old.connected = False
        conn = ServerConnection(
            user_id=user_id,
            name=name,
            session_token=secrets.token_urlsafe(24),
            info=info or {},
            is_global=is_global,
        )
        self._servers[key] = conn
        self._server_by_token[conn.session_token] = conn
        return conn

    # -- per-user view of the registry -------------------------------------
    # Helpers that any user can see — i.e. servers they own OR servers that
    # were registered globally (RELAY_REGISTER_GLOBAL=1 on the connector).
    # If two registrations share the same name (one owned, one global), the
    # owned one wins so a user can override a shared default with their own.

    def _visible(self, user_id: str) -> dict[str, ServerConnection]:
        out: dict[str, ServerConnection] = {}
        for (u, n), c in self._servers.items():
            if c.is_global:
                out[n] = c
        # User's own registrations shadow globals with the same name.
        for (u, n), c in self._servers.items():
            if u == user_id:
                out[n] = c
        return out

    def resolve_server_name(self, user_id: str, name: Optional[str]) -> str:
        visible = self._visible(user_id)
        if name:
            return name
        if len(visible) == 1:
            return next(iter(visible))
        raise ServerUnavailable(user_id, name, sorted(visible))

    def server_by_session(self, session_token: Optional[str]) -> Optional[ServerConnection]:
        if not session_token:
            return None
        return self._server_by_token.get(session_token)

    def server_connection(self, user_id: str, name: str) -> Optional[ServerConnection]:
        # Prefer the user's own — fall back to a globally-registered server
        # under the same name.
        c = self._servers.get((user_id, name))
        if c is not None:
            return c
        for (_, n), c in self._servers.items():
            if n == name and c.is_global:
                return c
        return None

    def find_server(self, user_id: str, name: Optional[str]) -> ServerConnection:
        visible = self._visible(user_id)
        if name:
            conn = visible.get(name)
        elif len(visible) == 1:
            conn = next(iter(visible.values()))  # unambiguous default
        else:
            conn = None
        if conn is None or not conn.connected:
            raise ServerUnavailable(user_id, name, sorted(visible))
        return conn

    def list_servers(self, user_id: str) -> list[Json]:
        return [
            {"name": n, "connected": c.connected, "info": c.info,
             "scope": "global" if c.is_global else "user"}
            for n, c in self._visible(user_id).items()
        ]

    def _queue_key(self, user_id: str, server_name: str) -> tuple[str, str]:
        return (user_id, server_name)

    def queue_opening_request(self, cs: ClientSession, messages: list[tuple[Json, bool]]) -> None:
        key = self._queue_key(cs.user_id, cs.server_name)
        self._queued.setdefault(key, []).append(QueuedRequest(session=cs, messages=messages))

    def drop_queued_for_session(self, session_id: str) -> None:
        for key, items in list(self._queued.items()):
            kept = [item for item in items if item.session.id != session_id]
            if kept:
                self._queued[key] = kept
            else:
                self._queued.pop(key, None)

    async def flush_queued(self, user_id: str, server_name: str) -> None:
        key = self._queue_key(user_id, server_name)
        conn = self._servers.get(key)
        if conn is None or not conn.connected:
            return
        items = self._queued.pop(key, [])
        for item in items:
            if self.get_client(item.session.id) is not item.session:
                continue
            conn.sessions.add(item.session.id)
            for payload, opening in item.messages:
                await self._send_to_server(conn, item.session, payload, opening=opening)

    # -- host (client) side -------------------------------------------------
    def open_client(self, user_id: str, server_name: str) -> ClientSession:
        sid = secrets.token_urlsafe(18)
        cs = ClientSession(id=sid, user_id=user_id, server_name=server_name)
        self._clients[sid] = cs
        return cs

    def get_client(self, session_id: Optional[str]):
        """Return the session object for an id. May be a ClientSession, an
        AggregateSession (host id), or a BackendLeg (aggregate sub-session id)."""
        if not session_id:
            return None
        return self._clients.get(session_id)

    # -- generic session table (used by the aggregator) ---------------------
    def connected_servers(self, user_id: str) -> list[ServerConnection]:
        # Returns all connections this user can reach: their own + any
        # globally-registered ones. Used by the aggregator (server=*) to
        # decide what to fan a host's initialize/tools-list/etc. across.
        # The dedupe-by-name keeps a user's own override from also showing
        # the global with the same name as a duplicate target.
        seen: dict[str, ServerConnection] = {}
        for (u, n), c in self._servers.items():
            if not c.connected:
                continue
            if c.is_global:
                seen.setdefault(n, c)
        for (u, n), c in self._servers.items():
            if u == user_id and c.connected:
                seen[n] = c
        return list(seen.values())

    def attach_session(self, session_id: str, obj) -> None:
        self._clients[session_id] = obj

    def detach_session(self, session_id: str) -> None:
        self._clients.pop(session_id, None)

    async def close_client(self, cs: ClientSession, notify_server: bool = True) -> None:
        self._clients.pop(cs.id, None)
        self.drop_queued_for_session(cs.id)
        cs.cancel_all_pending()
        conn = self._servers.get((cs.user_id, cs.server_name))
        if conn is not None:
            conn.sessions.discard(cs.id)
            if notify_server and conn.connected:
                try:
                    conn.downlink.put_nowait(
                        {"v": ENVELOPE_VERSION, "type": "close", "session": cs.id}
                    )
                except asyncio.QueueFull:
                    pass

    # -- routing ------------------------------------------------------------
    async def _send_to_server(self, conn: ServerConnection, cs: ClientSession, payload: Json,
                              *, opening: bool = False) -> None:
        if opening:
            # "user" rides the open envelope so the connector can scope the
            # spawned MCP child to the authenticated tenant (GML_MCP_USER).
            await conn.push({"v": ENVELOPE_VERSION, "type": "open", "session": cs.id,
                             "user": cs.user_id})
        # "user" rides every message too: if the connector restarted and lost
        # its session map, its feed() fallback re-opens the child — without
        # the user here that respawned child would run UNSCOPED and read/write
        # the wrong tenant's memories.
        await conn.push(
            {"v": ENVELOPE_VERSION, "type": "message", "session": cs.id,
             "payload": payload, "user": cs.user_id}
        )
        cs.last_seen = time.time()

    async def to_server(self, cs: ClientSession, payload: Json, *, opening: bool = False) -> None:
        """Forward a host-originated message down to its server."""
        conn = self.find_server(cs.user_id, cs.server_name)
        await self._send_to_server(conn, cs, payload, opening=opening)

    def to_client(self, cs: ClientSession, payload: Json) -> None:
        """Deliver a server-originated message up to its host.

        Responses to outstanding host requests resolve the matching pending future
        (so they appear on the originating POST's SSE stream). Everything else
        (server-initiated requests/notifications, or responses with no waiter)
        lands on the host's standby GET stream.
        """
        cs.last_seen = time.time()
        if message_kind(payload) == "response" and cs.resolve_pending(payload.get("id"), payload):
            return
        try:
            cs.standby.put_nowait(payload)
        except asyncio.QueueFull:
            # Drop oldest to make room rather than wedging the connector.
            try:
                cs.standby.get_nowait()
                cs.standby.put_nowait(payload)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    # -- introspection (for /health) ---------------------------------------
    def stats(self) -> Json:
        users = {u for (u, _) in self._servers}
        users.update(getattr(o, "user_id", None) for o in self._clients.values())
        users.discard(None)
        return {
            "active_users": len(users),
            "servers": len(self._servers),
            "servers_connected": sum(1 for c in self._servers.values() if c.connected),
            "client_sessions": len(self._clients),
        }
