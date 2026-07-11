"use client";

import type { ReactNode } from "react";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type AdminStats } from "@/lib/api";

/* ── shared live-stats query ────────────────────────────────────────── */

export const ADMIN_STATS_QK = ["admin", "stats"] as const;

/** One shared, 10s-polling query for the whole console — react-query
 *  dedupes it across pages so the nav feels instant and stays live. */
export function useAdminStats() {
  return useQuery({
    queryKey: ADMIN_STATS_QK,
    queryFn: api.admin.stats,
    refetchInterval: 10_000,
    refetchOnWindowFocus: true,
  });
}

/* ── formatters ─────────────────────────────────────────────────────── */

export function fmtDuration(seconds: number): string {
  if (!seconds || seconds < 1) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function relativeTime(iso: string | null): string {
  if (!iso) return "never";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "—";
  const s = Math.floor((Date.now() - t) / 1000);
  if (s < 10) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

export function isOnline(iso: string | null): boolean {
  if (!iso) return false;
  return Date.now() - new Date(iso).getTime() < 5 * 60 * 1000;
}

export function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleDateString();
}

export function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
}

export const PLAN_TONE: Record<string, string> = {
  admin: "border-accent/40 bg-accent/10 text-accent",
  pro: "border-cluster-2/40 bg-cluster-2/10 text-cluster-2",
  team: "border-cluster-1/40 bg-cluster-1/10 text-cluster-1",
  free: "border-border-strong bg-bg-3 text-text-2",
};

/* ── presentational ─────────────────────────────────────────────────── */

export function StatCard({
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
      className={`rounded-xl border bg-bg-2 p-4 ${glow ? "border-accent/40" : "border-border"}`}
    >
      <div className="flex items-center gap-2 text-text-2">
        {icon}
        <span className="text-xs font-medium uppercase tracking-wide">{label}</span>
      </div>
      <div className="mt-2 font-clash text-3xl font-semibold text-text-0 tnum">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-text-2">{sub}</div>}
    </div>
  );
}

export function PlanBadge({ plan }: { plan: string | null }) {
  const p = plan ?? "free";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${PLAN_TONE[p] ?? PLAN_TONE.free}`}
    >
      {p}
    </span>
  );
}

export function OnlineDot({ lastSeen }: { lastSeen: string | null }) {
  const online = isOnline(lastSeen);
  return (
    <span
      title={online ? "online" : "offline"}
      className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${online ? "bg-accent" : "bg-border-strong"}`}
    />
  );
}

/** 24-slot hourly bar chart from the timeseries (idle hours render as 0). */
export function Sparkline({ series }: { series: AdminStats["timeseries"] }) {
  const slots = useMemo(() => {
    const HOUR = 3600_000;
    const nowHour = Math.floor(Date.now() / HOUR);
    const byHour = new Map<number, number>();
    for (const b of series) byHour.set(Math.floor(new Date(b.bucket).getTime() / HOUR), b.events);
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
        return (
          <rect
            key={s.hour}
            x={i * (bw + gap)}
            y={H - h}
            width={bw}
            height={h}
            rx={1.5}
            className={i >= slots.length - 1 ? "fill-accent" : "fill-border-strong"}
          >
            <title>{`${s.events} event${s.events === 1 ? "" : "s"}`}</title>
          </rect>
        );
      })}
    </svg>
  );
}
