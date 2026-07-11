"use client";

import { useQuery } from "@tanstack/react-query";
import { api, type AdminAuditEntry } from "@/lib/api";
import { relativeTime } from "@/components/admin/kit";

const ACTION_LABEL: Record<string, string> = {
  set_plan: "changed plan",
  set_active: "changed access",
  delete_user: "deleted user",
  generate_invite: "generated invite",
};

function describe(e: AdminAuditEntry): string {
  const d = e.detail ?? {};
  if (e.action === "set_plan") return `${d.from ?? "?"} → ${d.to ?? "?"}`;
  if (e.action === "set_active") return d.to === false ? "suspended" : "restored";
  if (e.action === "delete_user") return `was ${(d.plan as string) ?? "free"}`;
  return "";
}

export default function AuditPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["admin", "audit"],
    queryFn: () => api.admin.audit(200),
    refetchInterval: 15_000,
  });
  const entries = data?.entries ?? [];

  return (
    <div className="px-6 py-6">
      <header className="mb-5">
        <h1 className="font-clash text-2xl font-semibold tracking-tight text-text-0">
          Audit log<span className="text-accent">.</span>
        </h1>
        <p className="mt-1 text-sm text-text-1">Every admin mutation — who did what, to whom, when.</p>
      </header>

      <div className="overflow-x-auto rounded-xl border border-border bg-bg-2">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-xs text-text-2">
              <th className="px-4 py-2.5 font-medium">When</th>
              <th className="px-4 py-2.5 font-medium">Admin</th>
              <th className="px-4 py-2.5 font-medium">Action</th>
              <th className="px-4 py-2.5 font-medium">Target</th>
              <th className="px-4 py-2.5 font-medium">Detail</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr><td colSpan={5} className="px-4 py-8 text-center text-text-2">Loading…</td></tr>
            ) : entries.length === 0 ? (
              <tr><td colSpan={5} className="px-4 py-8 text-center text-text-2">No admin actions recorded yet.</td></tr>
            ) : (
              entries.map((e, i) => (
                <tr key={i} className="border-t border-border/60">
                  <td className="whitespace-nowrap px-4 py-2.5 text-text-2">{relativeTime(e.ts)}</td>
                  <td className="px-4 py-2.5 text-text-1">{e.actor_email ?? e.actor_id}</td>
                  <td className="px-4 py-2.5 text-text-0">{ACTION_LABEL[e.action] ?? e.action}</td>
                  <td className="px-4 py-2.5 text-text-1">{e.target_email ?? e.target_id ?? "—"}</td>
                  <td className="px-4 py-2.5 font-mono text-xs text-text-2">{describe(e)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
