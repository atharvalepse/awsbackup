"use client";

import { useMemo } from "react";
import { Zap } from "lucide-react";
import {
  OnlineDot,
  Sparkline,
  fmtDuration,
  relativeTime,
  useAdminStats,
} from "@/components/admin/kit";

export default function ActivityPage() {
  const { data, isLoading } = useAdminStats();
  const recent = data?.recent ?? [];
  const users = data?.users ?? [];

  const leaderboard = useMemo(
    () => [...users].filter((u) => u.total_seconds > 0).sort((a, b) => b.total_seconds - a.total_seconds).slice(0, 10),
    [users],
  );

  return (
    <div className="px-6 py-6">
      <header className="mb-5">
        <h1 className="font-clash text-2xl font-semibold tracking-tight text-text-0">
          Activity<span className="text-accent">.</span>
        </h1>
        <p className="mt-1 text-sm text-text-1">Live request stream, daily volume and time-spent leaders.</p>
      </header>

      <section className="mb-4 rounded-xl border border-border bg-bg-2 p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-text-2">
            <Zap className="h-3.5 w-3.5" /> Activity — last 24h
          </span>
          <span className="text-xs text-text-2">{data ? `${data.overview.events_24h} events` : ""}</span>
        </div>
        {data ? <Sparkline series={data.timeseries} /> : <div className="h-16 animate-pulse rounded bg-bg-3" />}
      </section>

      <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-border bg-bg-2">
          <div className="border-b border-border px-4 py-3 text-xs font-medium uppercase tracking-wide text-text-2">
            Live feed
          </div>
          <ul className="max-h-[460px] divide-y divide-border/60 overflow-y-auto">
            {recent.length === 0 ? (
              <li className="px-4 py-6 text-center text-sm text-text-2">{isLoading ? "Loading…" : "No activity yet."}</li>
            ) : (
              recent.map((e, i) => (
                <li key={`${e.ts}-${i}`} className="px-4 py-2.5 text-sm">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-text-0">{e.email ?? e.user_id}</span>
                    <span className="shrink-0 text-xs text-text-2">{relativeTime(e.ts)}</span>
                  </div>
                  <div className="mt-0.5 truncate font-mono text-xs text-text-2">{e.method} {e.path}</div>
                </li>
              ))
            )}
          </ul>
        </div>

        <div className="rounded-xl border border-border bg-bg-2">
          <div className="border-b border-border px-4 py-3 text-xs font-medium uppercase tracking-wide text-text-2">
            Time-spent leaders
          </div>
          <ul className="divide-y divide-border/60">
            {leaderboard.length === 0 ? (
              <li className="px-4 py-6 text-center text-sm text-text-2">No measured sessions yet.</li>
            ) : (
              leaderboard.map((u, i) => (
                <li key={u.user_id} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                  <span className="w-5 shrink-0 text-right text-xs text-text-2 tnum">{i + 1}</span>
                  <OnlineDot lastSeen={u.last_seen} />
                  <span className="min-w-0 flex-1 truncate text-text-0">{u.email ?? u.user_id}</span>
                  <span className="shrink-0 tabular-nums text-text-1 tnum">{fmtDuration(u.total_seconds)}</span>
                  <span className="w-16 shrink-0 text-right text-xs text-text-2 tnum">{u.session_count} sess</span>
                </li>
              ))
            )}
          </ul>
        </div>
      </section>
    </div>
  );
}
