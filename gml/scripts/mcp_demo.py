"""End-to-end MCP demo — pretends to be Claude Desktop calling GML.

Spawns ``gml mcp`` as a subprocess, talks to it over stdio (the same way
Claude Desktop / Cursor / VS Code do), and exercises every tool with real
inputs. Print results to the console so you can SEE the protocol in
action without setting up an MCP client.

Run from the repo root:

    .venv/bin/python scripts/mcp_demo.py
"""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SECTION = "\n" + "─" * 70 + "\n"


async def demo() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "orchestration.mcp_server"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(SECTION + "1. list_tools — what does the server expose?")
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"  • {t.name}({', '.join(t.inputSchema.get('properties', {}).keys())})")
                if t.description:
                    first_line = t.description.strip().splitlines()[0]
                    print(f"      {first_line}")

            print(SECTION + "2. status — pipeline wiring + memory count")
            r = await session.call_tool("status", {})
            print(_first_text(r))

            # --- PRIMARY PATH: query() runs the full pipeline -------------
            print(SECTION + "3. query('how is auth_service implemented?')")
            print("   Runs: classify → embed → retriever.search (probe) →")
            print("         YES-hits branch → top50 → rerank → SAM.resolve_conflicts →")
            print("         assembler → translator. Watch stderr for stage logs.")
            r = await session.call_tool(
                "query", {"text": "how is auth_service implemented?"},
            )
            print(_first_text(r))

            print(SECTION + "4. query('what is sdp?')  — should hit NO-match branch")
            print("   Runs: classify → embed → search returns [] →")
            print("         SAM.reason_from_scratch → assembler → translator.")
            r = await session.call_tool(
                "query", {"text": "what is sdp?"},
            )
            print(_first_text(r))

            print(SECTION + "5. ingest(user_query, assistant_reply) — full extractor path")
            print("   MemoryExtractor runs DeepSeek to pull facts, persists,")
            print("   live-ingests into the retriever. Next query should see them.")
            r = await session.call_tool(
                "ingest",
                {
                    "user_query": "we use Stripe for payments and the prod webhook lives at webhooks.example.com/stripe",
                    "assistant_reply": (
                        "Got it — noted that the payments stack is Stripe and the "
                        "production webhook endpoint is webhooks.example.com/stripe."
                    ),
                },
            )
            print(_first_text(r))

            print(SECTION + "6. query('what do we use for payments?')  — should now find #5's fact")
            r = await session.call_tool(
                "query", {"text": "what do we use for payments?"},
            )
            print(_first_text(r))

            # --- LOW-LEVEL TOOLS: still work for direct/debug use --------
            print(SECTION + "7. recall — low-level retrieval (bypasses pipeline)")
            r = await session.call_tool(
                "recall",
                {"query": "auth_service", "top_k": 3},
            )
            print(_first_text(r))

            print(SECTION + "8. list_memories(entity='auth_service')")
            r = await session.call_tool(
                "list_memories",
                {"entity": "auth_service", "limit": 5},
            )
            print(_first_text(r))

            print(SECTION + "Done. Tool calls 3-6 are what Claude Desktop should exercise per turn.")
            return 0


def _first_text(call_tool_result) -> str:
    for c in call_tool_result.content:
        if hasattr(c, "text"):
            return c.text
    return str(call_tool_result)


if __name__ == "__main__":
    sys.exit(asyncio.run(demo()))
