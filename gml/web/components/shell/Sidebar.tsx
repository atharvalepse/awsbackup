"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Network, List, Search, Activity, KeyRound, Settings, RotateCcw, Plus, Plug, LogOut, Globe, ShieldAlert, MessagesSquare } from "lucide-react";
import { Logomark } from "@/components/ui/logomark";
import { useAppStore } from "@/stores/app-store";
import { useHealth, useMe } from "@/hooks/use-gml";
import { clearToken } from "@/lib/auth";
import { cn } from "@/lib/utils";

const CLUSTER_NAME = "akhrot-eu";

type Item = { icon: typeof Network; label: string; href: string; view?: "graph" | "list" };
const ITEMS: Item[] = [
  { icon: Network, label: "Graph", href: "/app", view: "graph" },
  { icon: List, label: "Memories", href: "/app", view: "list" },
  { icon: Search, label: "Recall", href: "/app" },
  { icon: MessagesSquare, label: "Conversations", href: "/app/conversations" },
  { icon: Activity, label: "Trace", href: "/app/trace" },
  { icon: KeyRound, label: "API keys", href: "/app/keys" },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const qc = useQueryClient();
  const { view, setView, select, replayTour, setAddOpen, setMcpOpen, setClipOpen } = useAppStore();
  const { data, isError } = useHealth();
  const { data: me } = useMe();
  const [menu, setMenu] = useState(false);
  // Master key or a users-row plan of 'admin' both count as admin.
  const isAdmin = me?.is_master === true || me?.plan === "admin";

  const dot = isError
    ? "var(--cluster-3)" // red-ish: down
    : data?.status === "ok"
      ? "var(--accent)" // healthy
      : "var(--cluster-4)"; // amber: connecting/degraded

  return (
    <nav className="flex w-[72px] shrink-0 flex-col items-center border-r border-border bg-bg-1 py-4">
      <Link href="/" className="mb-5" aria-label="Akhrot home">
        <Logomark size={24} />
      </Link>

      <button
        onClick={() => setAddOpen(true)}
        title="Add memory (N)"
        className="mb-2 flex h-9 w-9 items-center justify-center rounded-md border border-border-strong text-text-1 transition-colors duration-[180ms] ease-out hover:border-accent hover:text-accent"
      >
        <Plus className="h-[18px] w-[18px]" />
      </button>

      <button
        onClick={() => setMcpOpen(true)}
        title="Connect your AI (MCP)"
        className="mb-2 flex h-9 w-9 items-center justify-center rounded-md border border-border-strong text-text-1 transition-colors duration-[180ms] ease-out hover:border-accent hover:text-accent"
      >
        <Plug className="h-[18px] w-[18px]" />
      </button>

      <button
        onClick={() => setClipOpen(true)}
        title="Download Memory Clipper extension"
        className="mb-4 flex h-9 w-9 items-center justify-center rounded-md border border-border-strong text-text-1 transition-colors duration-[180ms] ease-out hover:border-accent hover:text-accent"
      >
        <Globe className="h-[18px] w-[18px]" />
      </button>

      <ul className="flex flex-1 flex-col gap-1">
        {ITEMS.map((item, i) => {
          const isActive =
            item.href === "/app"
              ? pathname === "/app" && (item.view ? view === item.view : false)
              : pathname === item.href;
          const Icon = item.icon;
          return (
            <li key={i} className="relative">
              {isActive && (
                <span className="absolute -left-2 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-accent" />
              )}
              <Link
                href={item.href}
                data-coach={item.label === "Trace" ? "trace" : undefined}
                onClick={() => {
                  if (item.view) setView(item.view);
                  if (item.label === "Recall") select(null);
                }}
                title={item.label}
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-md transition-colors duration-[180ms] ease-out",
                  isActive ? "text-text-0" : "text-text-2 hover:bg-bg-3 hover:text-text-1",
                )}
              >
                <Icon className="h-[18px] w-[18px]" />
              </Link>
            </li>
          );
        })}
      </ul>

      <div className="relative mt-2 flex flex-col items-center gap-3">
        {isAdmin && (
          <Link
            href="/app/admin"
            title="Admin — invite codes"
            className={cn(
              "flex h-10 w-10 items-center justify-center rounded-md transition-colors duration-[180ms] ease-out",
              pathname === "/app/admin"
                ? "text-text-0"
                : "text-text-2 hover:bg-bg-3 hover:text-text-1",
            )}
          >
            <ShieldAlert className="h-[18px] w-[18px]" />
          </Link>
        )}
        {menu && (
          <div className="absolute bottom-0 left-[calc(100%+0.5rem)] z-50 w-44 rounded-md border border-border bg-bg-2 p-1 shadow-glow">
            <button
              onClick={() => {
                replayTour();
                setMenu(false);
              }}
              className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-xs text-text-1 hover:bg-bg-3 hover:text-text-0"
            >
              <RotateCcw className="h-3.5 w-3.5" /> Replay tour
            </button>
            <button
              onClick={() => {
                clearToken();
                // Drop the React Query cache so the next signed-in user
                // doesn't see this account's memories/graph flash on screen
                // for the staleTime window (60s). Without this, ["graph", 2]
                // and friends linger for gcTime (5min default) and get
                // re-served instantly on the next mount under a new user.
                qc.clear();
                router.replace("/login");
              }}
              className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-xs text-text-1 hover:bg-bg-3 hover:text-text-0"
            >
              <LogOut className="h-3.5 w-3.5" /> Log out
            </button>
          </div>
        )}
        <button
          onClick={() => setMenu((m) => !m)}
          title="Settings"
          className="flex h-10 w-10 items-center justify-center rounded-md text-text-2 transition-colors duration-[180ms] ease-out hover:bg-bg-3 hover:text-text-1"
        >
          <Settings className="h-[18px] w-[18px]" />
        </button>
        <button
          onClick={() => {
            clearToken();
            qc.clear();
            router.replace("/login");
          }}
          title="Log out"
          aria-label="Log out"
          className="flex h-10 w-10 items-center justify-center rounded-md text-text-2 transition-colors duration-[180ms] ease-out hover:bg-red-500/15 hover:text-red-400"
        >
          <LogOut className="h-[18px] w-[18px]" />
        </button>
        <div
          title={`${CLUSTER_NAME} · ${data?.status ?? "connecting"}`}
          className="flex flex-col items-center gap-1"
        >
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: dot, boxShadow: `0 0 8px ${dot}` }}
          />
        </div>
      </div>
    </nav>
  );
}
