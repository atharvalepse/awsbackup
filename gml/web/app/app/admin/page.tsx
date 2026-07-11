"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  Activity,
  CalendarDays,
  Clock,
  Copy,
  Plus,
  Radio,
  Users,
  Zap,
} from "lucide-react";
import {
  api,
  ApiError,
  type AdminStats,
  type AdminUserRow,
} from "@/lib/api";

const STATS_QK = ["admin-stats"] as const;
const INVITES_QK = ["admin-invites"] as const;

/* ── formatting helpers ─────────────────────────────────────────────── */

function fmtDuration(seconds: number): string {
  if (!seconds || seconds < 1) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function relativeTime(iso: string | null): string {
  if (!iso) return "never";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "—";
  const diff = Date.now() - t;
  const s = Math.floor(diff / 1000);
  if (s < 10) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

/** Online if seen within the last 5 minutes (matches the server's window). */
function isOnline(iso: string | null): boolean {
  if (!iso) return false;
  return Date.now() - new Date(iso).getTime() < 5 * 60 * 1000;
}

const PLAN_TONE: Record<string, string> = {
  admin: "border-accent/40 bg-accent/10 text-accent",
  pro: "border-cluster-2/40 bg-cluster-2/10 text-cluster-2",
  team: "border-cluster-1/40 bg-cluster-1/10 text-cluster-1",
  free: "border-border-strong bg-bg-3 text-text-2",
};

/* ── small presentational pieces ────────────────────────────────────── */

function StatCard({
  icon,
  label,
  value,
  sub,
  glow,
}: {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  sub?: string;
  glow?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border bg-bg-2 p-4 ${
        glow ? "border-accent/40" : "border-border"
      }`}
    >
      <div className="flex items-center gap-2 text-text-2">
        {icon}
        <span className="text-xs font-medium uppercase tracking-wide">
          {label}
        </span>
      </div>
      <div className="mt-2 font-clash text-3xl font-semibold tabular-nums text-text-0 tnum">
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-text-2">{sub}</div>}
    </div>
  );
}

/** 24-slot hourly bar chart built from the timeseries (idle hours = 0). */
function ActivitySparkline({
  series,
}: {
  series: AdminStats["timeseries"];
}) {
  const slots = useMemo(() => {
    const HOUR = 3600_000;
    const nowHour = Math.floor(Date.now() / HOUR);
    const byHour = new Map<number, number>();
    for (const b of series) {
      const h = Math.floor(new Date(b.bucket).getTime() / HOUR);
      byHour.set(h, b.events);
    }
    return Array.from({ length: 24 }, (_, i) => {
      const h = nowHour - 23 + i;
      return { hour: h, events: byHour.get(h) ?? 0 };
    });
  }, [series]);

  const max = Math.max(1, ...slots.map((s) => s.events));
  const W = 480;
  const H = 64;
  const gap = 3;
  const bw = (W - gap * (slots.length - 1)) / slots.length;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="h-16 w-full"
      role="img"
      aria-label="Activity over the last 24 hours"
    >
      {slots.map((s, i) => {
        const h = Math.max(2, (s.events / max) * (H - 4));
        const recent = i >= slots.length - 1;
        return (
          <rect
            key={s.hour}
            x={i * (bw + gap)}
            y={H - h}
            width={bw}
            height={h}
            rx={1.5}
            className={recent ? "fill-accent" : "fill-border-strong"}
          >
            <title>
              {`${s.events} event${s.events === 1 ? "" : "s"}`}
            </title>
          </rect>
        );
      })}
    </svg>
  );
}

/* ── page ───────────────────────────────────────────────────────────── */

export default function AdminPage() {
  const router = useRouter();
  const qc = useQueryClient();
  const [newCode, setNewCode] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  const stats = useQuery({
    queryKey: STATS_QK,
    queryFn: api.admin.stats,
    refetchInterval: 10_000,
    refetchOnWindowFocus: true,
  });

  const invites = useQuery({
    queryKey: INVITES_QK,
    queryFn: api.admin.listInviteCodes,
  });

  // A 403 from either query means "not an admin" — bounce to the app.
  useEffect(() => {
    const err = stats.error ?? invites.error;
    if (err instanceof ApiError && err.status === 403) router.replace("/app");
  }, [stats.error, invites.error, router]);

  const genCode = useMutation({
    mutationFn: api.admin.generateInviteCode,
    onSuccess: (res) => {
      setNewCode(res.code);
      qc.invalidateQueries({ queryKey: INVITES_QK });
    },
  });

  function copy(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(text);
    setTimeout(() => setCopied((c) => (c === text ? null : c)), 1500);
  }

  const data = stats.data;
  const ov = data?.overview;
  const users: AdminUserRow[] = data?.users ?? [];
  const recent = data?.recent ?? [];

  const avgSession = useMemo(() => {
    if (!users.length) return 0;
    const totalS = users.reduce((a, u) => a + u.total_seconds, 0);
    const totalSess = users.reduce((a, u) => a + u.session_count, 0);
    return totalSess ? totalS / totalSess : 0;
  }, [users]);

  const codes = invites.data?.codes ?? [];
  const unusedCodes = codes.filter((c) => !c.used_by_email);

  const isError =
    stats.isError &&
    !(stats.error instanceof ApiError && stats.error.status === 403);

  return (
    <div className="mx-auto h-full max-w-6xl overflow-y-auto px-6 py-8">
      {/* Header */}
      <header className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="font-clash text-2xl font-semibold tracking-tight text-text-0">
            Admin<span className="text-accent">.</span>
          </h1>
          <p className="mt-1 text-sm text-text-1">
            Live usage &amp; activity across Akhrots. Auto-refreshes every 10s.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2 rounded-full border border-border bg-bg-2 px-3 py-1.5 text-xs text-text-2">
          <span className="relative flex h-2 w-2">
            <span
              className={`absolute inline-flex h-full w-full rounded-full ${
                stats.isFetching ? "animate-ping bg-accent/60" : ""
              }`}
            />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
          </span>
          {stats.isLoading
            ? "connecting…"
            : isError
              ? "reconnecting…"
              : "live"}
        </div>
      </header>

      {isError && (
        <div className="mb-4 rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          Couldn’t reach the stats endpoint:{" "}
          {stats.error instanceof Error ? stats.error.message : "unknown error"}
          . Retrying…
        </div>
      )}

      {/* Stat cards */}
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard
          glow
          icon={<Radio className="h-3.5 w-3.5" />}
          label="Online now"
          value={ov ? ov.online_now : "—"}
          sub="last 5 min"
        />
        <StatCard
          icon={<Activity className="h-3.5 w-3.5" />}
          label="Active today"
          value={ov ? ov.dau : "—"}
          sub="DAU"
        />
        <StatCard
          icon={<CalendarDays className="h-3.5 w-3.5" />}
          label="This week"
          value={ov ? ov.wau : "—"}
          sub="WAU"
        />
        <StatCard
          icon={<CalendarDays className="h-3.5 w-3.5" />}
          label="This month"
          value={ov ? ov.mau : "—"}
          sub="MAU"
        />
        <StatCard
          icon={<Users className="h-3.5 w-3.5" />}
          label="Total users"
          value={ov ? ov.total_users : "—"}
          sub={ov ? `+${ov.new_7d} this week` : undefined}
        />
        <StatCard
          icon={<Clock className="h-3.5 w-3.5" />}
          label="Avg session"
          value={fmtDuration(avgSession)}
          sub={ov ? `${ov.events_24h} events / 24h` : undefined}
        />
      </section>

      {/* Activity chart */}
      <section className="mt-4 rounded-xl border border-border bg-bg-2 p-4">
        <div className="mb-3 flex items-center gap-2 text-text-2">
          <Zap className="h-3.5 w-3.5" />
          <span className="text-xs font-medium uppercase tracking-wide">
            Activity — last 24h
          </span>
        </div>
        {data ? (
          <ActivitySparkline series={data.timeseries} />
        ) : (
          <div className="h-16 animate-pulse rounded bg-bg-3" />
        )}
      </section>

      {/* Users + activity feed */}
      <section className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Users table */}
        <div className="lg:col-span-2 rounded-xl border border-border bg-bg-2">
          <div className="border-b border-border px-4 py-3 text-xs font-medium uppercase tracking-wide text-text-2">
            Users ({users.length})
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-xs text-text-2">
                  <th className="px-4 py-2 font-medium">User</th>
                  <th className="px-4 py-2 font-medium">Plan</th>
                  <th className="px-4 py-2 font-medium">Last seen</th>
                  <th className="px-4 py-2 text-right font-medium">Today</th>
                  <th className="px-4 py-2 text-right font-medium">Total</th>
                  <th className="px-4 py-2 text-right font-medium">Sessions</th>
                </tr>
              </thead>
              <tbody>
                {stats.isLoading ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-6 text-center text-text-2">
                      Loading…
                    </td>
                  </tr>
                ) : users.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-6 text-center text-text-2">
                      No users yet.
                    </td>
                  </tr>
                ) : (
                  users.map((u) => {
                    const online = isOnline(u.last_seen);
                    return (
                      <tr
                        key={u.user_id}
                        className="border-t border-border/60 hover:bg-bg-3/50"
                      >
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2">
                            <span
                              title={online ? "online" : "offline"}
                              className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                                online ? "bg-accent" : "bg-border-strong"
                              }`}
                            />
                            <span className="truncate text-text-0">
                              {u.email ?? u.user_id}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-2.5">
                          <span
                            className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
                              PLAN_TONE[u.plan ?? "free"] ?? PLAN_TONE.free
                            }`}
                          >
                            {u.plan ?? "free"}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-text-1">
                          {relativeTime(u.last_seen)}
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-text-1 tnum">
                          {fmtDuration(u.seconds_24h)}
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-text-1 tnum">
                          {fmtDuration(u.total_seconds)}
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-text-2 tnum">
                          {u.session_count}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Recent activity feed */}
        <div className="rounded-xl border border-border bg-bg-2">
          <div className="border-b border-border px-4 py-3 text-xs font-medium uppercase tracking-wide text-text-2">
            Recent activity
          </div>
          <ul className="max-h-[420px] divide-y divide-border/60 overflow-y-auto">
            {recent.length === 0 ? (
              <li className="px-4 py-6 text-center text-sm text-text-2">
                {stats.isLoading ? "Loading…" : "No activity recorded yet."}
              </li>
            ) : (
              recent.map((e, i) => (
                <li key={`${e.ts}-${i}`} className="px-4 py-2.5 text-sm">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-text-0">
                      {e.email ?? e.user_id}
                    </span>
                    <span className="shrink-0 text-xs text-text-2">
                      {relativeTime(e.ts)}
                    </span>
                  </div>
                  <div className="mt-0.5 truncate font-mono text-xs text-text-2">
                    {e.method} {e.path}
                  </div>
                </li>
              ))
            )}
          </ul>
        </div>
      </section>

      {/* Invite codes */}
      <section className="mt-6 rounded-xl border border-border bg-bg-2 p-5">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-sm font-medium text-text-1">
            Invite codes{" "}
            <span className="text-text-2">({unusedCodes.length} unused)</span>
          </h2>
          <button
            onClick={() => genCode.mutate()}
            disabled={genCode.isPending}
            className="flex items-center gap-1.5 rounded-md border border-border-strong px-3 py-1.5 text-sm text-text-1 transition-colors duration-[180ms] ease-out hover:border-accent hover:text-accent disabled:opacity-50"
          >
            <Plus className="h-3.5 w-3.5" />
            {genCode.isPending ? "Generating…" : "New code"}
          </button>
        </div>

        {newCode && (
          <button
            onClick={() => copy(newCode)}
            className="mb-3 flex items-center gap-2 rounded-md border border-accent/40 bg-accent/10 px-3 py-2 font-mono text-sm text-accent transition-colors hover:bg-accent/20"
          >
            <span className="tracking-wider">{newCode}</span>
            <Copy className="h-3.5 w-3.5" />
            <span className="text-xs">
              {copied === newCode ? "copied" : "copy"}
            </span>
          </button>
        )}

        {invites.isLoading ? (
          <p className="text-sm text-text-2">Loading codes…</p>
        ) : unusedCodes.length === 0 ? (
          <p className="text-sm italic text-text-2">
            No unused codes. Generate one above.
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {unusedCodes.map((c) => (
              <button
                key={c.code}
                onClick={() => copy(c.code)}
                className="flex items-center gap-2 rounded-md border border-border-strong px-3 py-1.5 font-mono text-xs tracking-wider text-text-1 transition-colors hover:border-accent/40 hover:text-accent"
              >
                {c.code}
                <span className="text-text-2">
                  {copied === c.code ? "✓" : <Copy className="h-3 w-3" />}
                </span>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
