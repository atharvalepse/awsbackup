"use client";

import { useMemo } from "react";
import {
  Activity,
  CalendarDays,
  Clock,
  Radio,
  UserPlus,
  Users,
  Zap,
} from "lucide-react";
import {
  Sparkline,
  StatCard,
  fmtDuration,
  relativeTime,
  useAdminStats,
} from "@/components/admin/kit";

export default function OverviewPage() {
  const { data, isLoading, isFetching, isError, error } = useAdminStats();
  const ov = data?.overview;
  const users = data?.users ?? [];
  const recent = data?.recent ?? [];

  const avgSession = useMemo(() => {
    const totalS = users.reduce((a, u) => a + u.total_seconds, 0);
    const totalSess = users.reduce((a, u) => a + u.session_count, 0);
    return totalSess ? totalS / totalSess : 0;
  }, [users]);

  return (
    <div className="px-6 py-6">
      <header className="mb-5 flex items-start justify-between gap-4">
        <div>
          <h1 className="font-clash text-2xl font-semibold tracking-tight text-text-0">
            Overview<span className="text-accent">.</span>
          </h1>
          <p className="mt-1 text-sm text-text-1">
            Live usage across Akhrots — auto-refreshing every 10s.
          </p>
        </div>
        <span className="flex shrink-0 items-center gap-2 rounded-full border border-border bg-bg-2 px-3 py-1.5 text-xs text-text-2">
          <span className="relative flex h-2 w-2">
            <span className={`absolute inline-flex h-full w-full rounded-full ${isFetching ? "animate-ping bg-accent/60" : ""}`} />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
          </span>
          {isLoading ? "connecting…" : isError ? "reconnecting…" : "live"}
        </span>
      </header>

      {isError && (
        <div className="mb-4 rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          Couldn’t reach the stats endpoint:{" "}
          {error instanceof Error ? error.message : "unknown error"}. Retrying…
        </div>
      )}

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard glow icon={<Radio className="h-3.5 w-3.5" />} label="Online now" value={ov ? ov.online_now : "—"} sub="last 5 min" />
        <StatCard icon={<Activity className="h-3.5 w-3.5" />} label="Active today" value={ov ? ov.dau : "—"} sub="DAU" />
        <StatCard icon={<CalendarDays className="h-3.5 w-3.5" />} label="This week" value={ov ? ov.wau : "—"} sub="WAU" />
        <StatCard icon={<CalendarDays className="h-3.5 w-3.5" />} label="This month" value={ov ? ov.mau : "—"} sub="MAU" />
        <StatCard icon={<Users className="h-3.5 w-3.5" />} label="Total users" value={ov ? ov.total_users : "—"} sub={ov ? `${ov.active_accounts} active` : undefined} />
        <StatCard icon={<UserPlus className="h-3.5 w-3.5" />} label="New (7d)" value={ov ? ov.new_7d : "—"} sub={ov ? `${ov.admins} admin${ov.admins === 1 ? "" : "s"}` : undefined} />
      </section>

      <section className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 rounded-xl border border-border bg-bg-2 p-4">
          <div className="mb-3 flex items-center justify-between">
            <span className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-text-2">
              <Zap className="h-3.5 w-3.5" /> Activity — last 24h
            </span>
            <span className="text-xs text-text-2">
              {ov ? `${ov.events_24h} events` : ""}
            </span>
          </div>
          {data ? <Sparkline series={data.timeseries} /> : <div className="h-16 animate-pulse rounded bg-bg-3" />}
        </div>
        <StatCard icon={<Clock className="h-3.5 w-3.5" />} label="Avg session" value={fmtDuration(avgSession)} sub="across all sessions" />
      </section>

      <section className="mt-4 rounded-xl border border-border bg-bg-2">
        <div className="border-b border-border px-4 py-3 text-xs font-medium uppercase tracking-wide text-text-2">
          Recent activity
        </div>
        <ul className="divide-y divide-border/60">
          {recent.length === 0 ? (
            <li className="px-4 py-6 text-center text-sm text-text-2">
              {isLoading ? "Loading…" : "No activity recorded yet."}
            </li>
          ) : (
            recent.slice(0, 12).map((e, i) => (
              <li key={`${e.ts}-${i}`} className="flex items-center justify-between gap-3 px-4 py-2.5 text-sm">
                <span className="truncate text-text-0">{e.email ?? e.user_id}</span>
                <span className="truncate font-mono text-xs text-text-2">{e.method} {e.path}</span>
                <span className="shrink-0 text-xs text-text-2">{relativeTime(e.ts)}</span>
              </li>
            ))
          )}
        </ul>
      </section>
    </div>
  );
}
