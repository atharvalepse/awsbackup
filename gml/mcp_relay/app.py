"""Starlette ASGI app exposing the relay.

Two families of endpoints:

Host-facing (standard MCP Streamable HTTP — stock MCP clients work unchanged):
    POST   /mcp     send JSON-RPC; replies stream back as SSE (or 202 for no-reply)
    GET    /mcp     open the standby SSE stream for server-initiated messages
    DELETE /mcp     terminate the session

Connector-facing (the small agent that bridges a local MCP server in):
    POST   /relay/register   register a server for the authenticated user
    GET    /relay/stream      downlink: SSE of envelopes to push to the server
    POST   /relay/message     uplink: server-originated JSON-RPC back to hosts

Plus GET /health and GET / for liveness/info.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any, Optional

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

from . import PROTOCOL_VERSION, __version__
from .aggregator import AggregateSession, BackendLeg, open_aggregate
from .auth import EmailTaken, make_session, read_session
from .registry import RelayState, ServerUnavailable, message_kind
from .sse import format_comment, format_sse

Json = dict[str, Any]

REQUEST_TIMEOUT = float(os.environ.get("RELAY_REQUEST_TIMEOUT", "120"))
KEEPALIVE = float(os.environ.get("RELAY_KEEPALIVE", "15"))
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


# -- small helpers ----------------------------------------------------------
def _token(request: Request) -> Optional[str]:
    h = request.headers.get("authorization", "")
    if h[:7].lower() == "bearer ":
        return h[7:].strip()
    return request.query_params.get("token")  # EventSource can't set headers


def _server_session(request: Request) -> Optional[str]:
    return request.headers.get("x-relay-server-session") or request.query_params.get("s")


def jsonrpc_error(msg_id: Any, code: int, message: str, data: Any = None) -> Json:
    err: Json = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def _is_initialize(messages: list[Json]) -> bool:
    return (
        len(messages) == 1
        and message_kind(messages[0]) == "request"
        and messages[0].get("method") == "initialize"
    )


# -- host-facing endpoints --------------------------------------------------
async def mcp_post(request: Request) -> Response:
    state: RelayState = request.app.state.relay
    user_id = await state.user_for_token(_token(request))
    if user_id is None:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(jsonrpc_error(None, -32700, "parse error"), status_code=400)
    messages: list[Json] = body if isinstance(body, list) else [body]
    if not messages or not all(isinstance(m, dict) for m in messages):
        return JSONResponse(jsonrpc_error(None, -32600, "invalid request"), status_code=400)

    session_id = request.headers.get("mcp-session-id")
    opening = False

    if session_id is None:
        if not _is_initialize(messages):
            return JSONResponse(
                jsonrpc_error(messages[0].get("id"), -32600,
                              "missing Mcp-Session-Id; send 'initialize' first"),
                status_code=400,
            )
        server_name = request.query_params.get("server") or request.headers.get("x-mcp-server")
        try:
            if server_name == "*":
                cs = await open_aggregate(state, user_id)
            else:
                target_name = state.resolve_server_name(user_id, server_name)
                conn = state.server_connection(user_id, target_name)
                if conn is None:
                    raise ServerUnavailable(user_id, target_name, [])
                cs = state.open_client(user_id, target_name)
                if conn.connected:
                    conn.sessions.add(cs.id)
        except ServerUnavailable as exc:
            return JSONResponse(
                jsonrpc_error(messages[0].get("id"), -32002,
                              f"server {exc.name or '(default)'} unavailable",
                              {"available": exc.available}),
                status_code=503,
            )
        opening = True
    else:
        cs = state.get_client(session_id)
        if cs is None or cs.user_id != user_id or isinstance(cs, BackendLeg):
            return JSONResponse(
                jsonrpc_error(messages[0].get("id"), -32001, "session not found"),
                status_code=404,
            )

    # Register pending futures *before* forwarding, then forward each message.
    pending: dict[str, asyncio.Future] = {}
    original_ids: dict[str, Any] = {}
    queued_messages: list[tuple[Json, bool]] | None = None
    if opening and not isinstance(cs, AggregateSession):
        conn = state.server_connection(user_id, cs.server_name)
        if conn is None or not conn.connected:
            queued_messages = [(msg, i == 0) for i, msg in enumerate(messages)]
    try:
        for i, msg in enumerate(messages):
            if message_kind(msg) == "request":
                rid = msg.get("id")
                fut = cs.register_pending(rid)
                pending[str(rid)] = fut
                original_ids[str(rid)] = rid
        if queued_messages is not None:
            state.queue_opening_request(cs, queued_messages)
        else:
            for i, msg in enumerate(messages):
                if isinstance(cs, AggregateSession):
                    await cs.send_from_host(msg, state)
                else:
                    await state.to_server(cs, msg, opening=(opening and i == 0))
    except ServerUnavailable as exc:
        cs.cancel_all_pending()
        if opening:
            await state.close_client(cs, notify_server=False)
        return JSONResponse(
            jsonrpc_error(messages[0].get("id"), -32002,
                          f"server {exc.name or '(default)'} unavailable",
                          {"available": exc.available}),
            status_code=503,
        )

    headers = dict(SSE_HEADERS)
    headers["Mcp-Session-Id"] = cs.id

    if not pending:
        # Only notifications / responses were sent — nothing to stream back.
        return Response(status_code=202, headers={"Mcp-Session-Id": cs.id})

    loop = asyncio.get_event_loop()

    async def gen():
        remaining = dict(pending)
        deadline = loop.time() + REQUEST_TIMEOUT
        try:
            while remaining:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    for sid, fut in list(remaining.items()):
                        if not fut.done():
                            fut.cancel()
                        cs.pending.pop(sid, None)
                        yield format_sse(jsonrpc_error(original_ids[sid], -32000,
                                                        "relay timeout waiting for server"))
                    state.drop_queued_for_session(cs.id)
                    return
                done, _ = await asyncio.wait(
                    set(remaining.values()), timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for sid in list(remaining):
                    fut = remaining[sid]
                    if fut.done():
                        del remaining[sid]
                        try:
                            yield format_sse(fut.result())
                        except asyncio.CancelledError:
                            return
                        except Exception:
                            yield format_sse(jsonrpc_error(original_ids[sid], -32000, "relay error"))
        except asyncio.CancelledError:
            # Host disconnected mid-stream; drop the waiters.
            for sid, fut in remaining.items():
                if not fut.done():
                    fut.cancel()
                cs.pending.pop(sid, None)
            state.drop_queued_for_session(cs.id)
            raise

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


async def mcp_get(request: Request) -> Response:
    state: RelayState = request.app.state.relay
    user_id = await state.user_for_token(_token(request))
    if user_id is None:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    session_id = request.headers.get("mcp-session-id") or request.query_params.get("session")
    cs = state.get_client(session_id)
    if cs is None or cs.user_id != user_id or isinstance(cs, BackendLeg):
        return JSONResponse({"error": "session not found"}, status_code=404)

    async def gen():
        yield format_comment("ready")
        while True:
            if await request.is_disconnected():
                break
            try:
                payload = await asyncio.wait_for(cs.standby.get(), timeout=KEEPALIVE)
                yield format_sse(payload)
            except asyncio.TimeoutError:
                yield format_comment("keepalive")

    return StreamingResponse(gen(), media_type="text/event-stream", headers=dict(SSE_HEADERS))


async def mcp_delete(request: Request) -> Response:
    state: RelayState = request.app.state.relay
    user_id = await state.user_for_token(_token(request))
    if user_id is None:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    session_id = request.headers.get("mcp-session-id") or request.query_params.get("session")
    cs = state.get_client(session_id)
    if cs is None or cs.user_id != user_id or isinstance(cs, BackendLeg):
        return JSONResponse({"error": "session not found"}, status_code=404)
    if isinstance(cs, AggregateSession):
        await cs.close(state)
    else:
        await state.close_client(cs)
    return Response(status_code=204)


# -- connector-facing endpoints ---------------------------------------------
async def relay_register(request: Request) -> Response:
    state: RelayState = request.app.state.relay
    user_id = await state.user_for_token(_token(request))
    if user_id is None:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = (body.get("server") or "default").strip()
    # `RELAY_REGISTER_GLOBAL=1` on the connector sends "global": true here.
    # Honouring it makes the connector visible to every authenticated user,
    # not just the account whose token launched it — the right behaviour for
    # a shared service connector (e.g. gml-connector). The default stays
    # per-user so personal connectors keep their isolation.
    is_global = bool(body.get("global"))
    conn = state.register_server(user_id, name, body.get("info"), is_global=is_global)
    return JSONResponse(
        {"server_session": conn.session_token, "server": name,
         "user": user_id, "scope": "global" if is_global else "user"}
    )


async def relay_stream(request: Request) -> Response:
    state: RelayState = request.app.state.relay
    conn = state.server_by_session(_server_session(request))
    if conn is None:
        return JSONResponse({"error": "unknown server session"}, status_code=401)
    conn.connected = True

    async def gen():
        await state.flush_queued(conn.user_id, conn.name)
        yield format_comment("connected")
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    env = await asyncio.wait_for(conn.downlink.get(), timeout=KEEPALIVE)
                    yield format_sse(env)
                except asyncio.TimeoutError:
                    yield format_comment("keepalive")
        finally:
            conn.connected = False

    return StreamingResponse(gen(), media_type="text/event-stream", headers=dict(SSE_HEADERS))


async def relay_message(request: Request) -> Response:
    state: RelayState = request.app.state.relay
    conn = state.server_by_session(_server_session(request))
    if conn is None:
        return JSONResponse({"error": "unknown server session"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "parse error"}, status_code=400)
    items = body if isinstance(body, list) else [body]
    for item in items:
        if not isinstance(item, dict):
            continue
        cs = state.get_client(item.get("session"))
        payload = item.get("payload")
        if cs is None or payload is None:
            continue
        # Isolation guard: deliver only to a host that is *talking to this
        # exact server*. The server-name must always match. The owner-user
        # match is enforced only for per-user connectors — a globally-
        # registered connector intentionally serves every authenticated
        # user, so cross-user delivery is correct there (the host session
        # already has its own per-host subprocess inside the connector).
        if cs.server_name != conn.name:
            continue
        if not conn.is_global and cs.user_id != conn.user_id:
            continue
        if isinstance(cs, BackendLeg):
            cs.parent.on_backend_message(cs.server_name, payload, state)
        else:
            state.to_client(cs, payload)
    return Response(status_code=202)


# -- auth endpoints (only mounted when a DB auth backend is configured) -----
def _session_user(request: Request) -> Optional[str]:
    h = request.headers.get("authorization", "")
    if h[:7].lower() == "bearer ":
        return read_session(h[7:].strip(), request.app.state.session_secret)
    return None


async def auth_signup(request: Request) -> Response:
    store = request.app.state.auth_store
    body = await request.json()
    email = (body.get("email") or "").strip()
    password = body.get("password") or ""
    if "@" not in email or len(password) < 8:
        return JSONResponse({"error": "valid email and password (>=8 chars) required"},
                            status_code=400)
    try:
        user_id = await store.create_user(email, password)
    except EmailTaken:
        return JSONResponse({"error": "email already registered"}, status_code=409)
    session = make_session(user_id, request.app.state.session_secret)
    return JSONResponse({"user_id": user_id, "email": email.lower(), "session": session},
                        status_code=201)


async def auth_login(request: Request) -> Response:
    store = request.app.state.auth_store
    body = await request.json()
    user_id = await store.verify_login(body.get("email") or "", body.get("password") or "")
    if user_id is None:
        return JSONResponse({"error": "invalid credentials"}, status_code=401)
    session = make_session(user_id, request.app.state.session_secret)
    return JSONResponse({"user_id": user_id, "session": session})


async def auth_tokens(request: Request) -> Response:
    store = request.app.state.auth_store
    user_id = _session_user(request)
    if user_id is None:
        return JSONResponse({"error": "login required"}, status_code=401)
    if request.method == "POST":
        body = await request.json() if await request.body() else {}
        created = await store.create_token(user_id, (body or {}).get("label"))
        return JSONResponse({**created,
                             "note": "copy this token now — it is not shown again"},
                            status_code=201)
    return JSONResponse({"tokens": await store.list_tokens(user_id)})


async def auth_revoke(request: Request) -> Response:
    store = request.app.state.auth_store
    user_id = _session_user(request)
    if user_id is None:
        return JSONResponse({"error": "login required"}, status_code=401)
    ok = await store.revoke_token(user_id, request.path_params["token_id"])
    return Response(status_code=204) if ok else JSONResponse({"error": "not found"}, status_code=404)


async def dashboard(request: Request) -> Response:
    return HTMLResponse(_DASHBOARD_HTML)


# -- meta -------------------------------------------------------------------
async def health(request: Request) -> Response:
    state: RelayState = request.app.state.relay
    return JSONResponse({"status": "ok", "version": __version__, "stats": state.stats()})


async def index(request: Request) -> Response:
    return PlainTextResponse(
        f"mcp-relay {__version__}\n"
        f"protocol: {PROTOCOL_VERSION}\n\n"
        "host endpoint:      /mcp (POST/GET/DELETE, Streamable HTTP)\n"
        "connector endpoints: /relay/register, /relay/stream, /relay/message\n"
        "health:             /health\n"
    )


def create_app(state: RelayState, *, auth_store=None, session_secret: Optional[str] = None) -> Starlette:
    routes = [
        Route("/", index, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/mcp", mcp_post, methods=["POST"]),
        Route("/mcp", mcp_get, methods=["GET"]),
        Route("/mcp", mcp_delete, methods=["DELETE"]),
        Route("/relay/register", relay_register, methods=["POST"]),
        Route("/relay/stream", relay_stream, methods=["GET"]),
        Route("/relay/message", relay_message, methods=["POST"]),
    ]
    if auth_store is not None:
        routes += [
            Route("/auth/signup", auth_signup, methods=["POST"]),
            Route("/auth/login", auth_login, methods=["POST"]),
            Route("/auth/tokens", auth_tokens, methods=["GET", "POST"]),
            Route("/auth/tokens/{token_id:int}", auth_revoke, methods=["DELETE"]),
            Route("/dashboard", dashboard, methods=["GET"]),
        ]
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )
    ]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await state.startup()
        try:
            yield
        finally:
            await state.shutdown()

    app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
    app.state.relay = state
    app.state.auth_store = auth_store
    app.state.session_secret = session_secret
    return app


_DASHBOARD_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>mcp-relay</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem}
input,button{font:inherit;padding:.5rem;margin:.2rem 0}input{width:100%;box-sizing:border-box}
button{cursor:pointer}.card{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
code{background:#f4f4f4;padding:.1rem .3rem;border-radius:4px}.tok{word-break:break-all}
.muted{color:#777}</style></head><body>
<h1>mcp-relay</h1>
<div class=card id=auth>
  <h3>Sign up / Log in</h3>
  <input id=email type=email placeholder=email autocomplete=username>
  <input id=password type=password placeholder="password (min 8 chars)" autocomplete=current-password>
  <button onclick=signup()>Sign up</button> <button onclick=login()>Log in</button>
  <p id=authmsg class=muted></p>
</div>
<div class=card id=tokens style=display:none>
  <h3>API tokens</h3>
  <input id=label placeholder="label (e.g. cursor-laptop)">
  <button onclick=mktoken()>Create token</button>
  <div id=newtoken></div>
  <ul id=list></ul>
  <button onclick=logout()>Log out</button>
</div>
<script>
let s=localStorage.getItem('relay_session');
const j=(m)=>document.getElementById('authmsg').textContent=m;
function show(){document.getElementById('auth').style.display=s?'none':'';
  document.getElementById('tokens').style.display=s?'':'none'; if(s) listTokens();}
async function api(path,opts={}){opts.headers=Object.assign({'Content-Type':'application/json'},
  opts.headers||{}); if(s)opts.headers.Authorization='Bearer '+s; return fetch(path,opts);}
async function signup(){const r=await api('/auth/signup',{method:'POST',body:JSON.stringify(
  {email:email.value,password:password.value})});const d=await r.json();
  if(r.ok){s=d.session;localStorage.setItem('relay_session',s);show();}else j(d.error);}
async function login(){const r=await api('/auth/login',{method:'POST',body:JSON.stringify(
  {email:email.value,password:password.value})});const d=await r.json();
  if(r.ok){s=d.session;localStorage.setItem('relay_session',s);show();}else j(d.error);}
function logout(){s=null;localStorage.removeItem('relay_session');show();}
async function mktoken(){const r=await api('/auth/tokens',{method:'POST',body:JSON.stringify(
  {label:label.value})});const d=await r.json();if(r.ok){
  document.getElementById('newtoken').innerHTML='<p>New token (copy now):</p><p class=tok><code>'
  +d.token+'</code></p>';listTokens();}else if(r.status==401){logout();}}
async function listTokens(){const r=await api('/auth/tokens');if(r.status==401){logout();return;}
  const d=await r.json();document.getElementById('list').innerHTML=d.tokens.map(t=>
  '<li>#'+t.id+' '+(t.label||'')+(t.revoked?' <span class=muted>(revoked)</span>':
  ' <button onclick=revoke('+t.id+')>revoke</button>')+'</li>').join('');}
async function revoke(id){await api('/auth/tokens/'+id,{method:'DELETE'});listTokens();}
show();
</script></body></html>"""
