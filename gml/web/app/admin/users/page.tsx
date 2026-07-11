"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Search, Trash2, X } from "lucide-react";
import { api, ApiError, type AdminUserRow } from "@/lib/api";
import {
  ADMIN_STATS_QK,
  OnlineDot,
  PlanBadge,
  fmtBytes,
  fmtDate,
  fmtDuration,
  relativeTime,
  useAdminStats,
} from "@/components/admin/kit";

const PLANS = ["free", "pro", "team", "admin"];

export default function UsersPage() {
  const qc = useQueryClient();
  const { data, isLoading } = useAdminStats();
  const me = useQuery({ queryKey: ["admin", "me"], queryFn: api.auth.me });
  const myId = me.data?.user_id;

  const [q, setQ] = useState("");
  const [detailId, setDetailId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<AdminUserRow | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const invalidate = () => qc.invalidateQueries({ queryKey: ADMIN_STATS_QK });
  const onError = (e: unknown) =>
    setActionError(e instanceof ApiError ? e.message : "Action failed");

  const update = useMutation({
    mutationFn: (v: { userId: string; patch: { plan?: string; is_active?: boolean } }) =>
      api.admin.updateUser(v.userId, v.patch),
    onSuccess: () => { setActionError(null); invalidate(); },
    onError,
  });
  const del = useMutation({
    mutationFn: (userId: string) => api.admin.deleteUser(userId),
    onSuccess: () => { setActionError(null); setConfirmDelete(null); invalidate(); },
    onError,
  });

  const users = data?.users ?? [];
  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!t) return users;
    return users.filter(
      (u) => (u.email ?? "").toLowerCase().includes(t) || u.user_id.toLowerCase().includes(t),
    );
  }, [users, q]);

  return (
    <div className="px-6 py-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-clash text-2xl font-semibold tracking-tight text-text-0">
            Users<span className="text-accent">.</span>
          </h1>
          <p className="mt-1 text-sm text-text-1">
            {users.length} accounts · manage plan, access and lifecycle.
          </p>
        </div>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-text-2" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search email or id…"
            className="w-64 rounded-md border border-border-strong bg-bg-2 py-2 pl-8 pr-3 text-sm text-text-0 outline-none focus:border-accent"
          />
        </div>
      </header>

      {actionError && (
        <div className="mb-3 flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-sm text-red-400">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          {actionError}
        </div>
      )}

      <div className="overflow-x-auto rounded-xl border border-border bg-bg-2">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-xs text-text-2">
              <th className="px-4 py-2.5 font-medium">User</th>
              <th className="px-3 py-2.5 font-medium">Plan / role</th>
              <th className="px-3 py-2.5 font-medium">Status</th>
              <th className="px-3 py-2.5 font-medium">Joined</th>
              <th className="px-3 py-2.5 font-medium">Last seen</th>
              <th className="px-3 py-2.5 text-right font-medium">Total time</th>
              <th className="px-3 py-2.5 text-right font-medium">Sessions</th>
              <th className="px-4 py-2.5 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-text-2">Loading…</td></tr>
            ) : filtered.length === 0 ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-text-2">No matching users.</td></tr>
            ) : (
              filtered.map((u) => {
                const isSelf = u.user_id === myId;
                const busy = update.isPending && update.variables?.userId === u.user_id;
                return (
                  <tr key={u.user_id} className={`border-t border-border/60 hover:bg-bg-3/40 ${!u.is_active ? "opacity-60" : ""}`}>
                    <td className="px-4 py-2.5">
                      <button onClick={() => setDetailId(u.user_id)} className="flex items-center gap-2 text-left hover:text-accent">
                        <OnlineDot lastSeen={u.last_seen} />
                        <span className="truncate text-text-0">{u.email ?? u.user_id}</span>
                        {isSelf && <span className="text-[10px] text-text-2">(you)</span>}
                      </button>
                    </td>
                    <td className="px-3 py-2.5">
                      <select
                        value={u.plan ?? "free"}
                        disabled={busy}
                        onChange={(e) => update.mutate({ userId: u.user_id, patch: { plan: e.target.value } })}
                        className="rounded-md border border-border-strong bg-bg-1 px-2 py-1 text-xs text-text-0 outline-none focus:border-accent disabled:opacity-50"
                      >
                        {PLANS.map((p) => <option key={p} value={p}>{p}</option>)}
                      </select>
                    </td>
                    <td className="px-3 py-2.5">
                      {u.is_active ? (
                        <span className="text-xs text-accent">active</span>
                      ) : (
                        <span className="text-xs text-text-2">suspended</span>
                      )}
                    </td>
                    <td className="px-3 py-2.5 text-text-1">{fmtDate(u.created_at)}</td>
                    <td className="px-3 py-2.5 text-text-1">{relativeTime(u.last_seen)}</td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-text-1 tnum">{fmtDuration(u.total_seconds)}</td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-text-2 tnum">{u.session_count}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => update.mutate({ userId: u.user_id, patch: { is_active: !u.is_active } })}
                          disabled={busy || isSelf}
                          title={isSelf ? "You can’t suspend yourself" : u.is_active ? "Suspend access" : "Restore access"}
                          className="rounded border border-border-strong px-2 py-1 text-xs text-text-1 transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
                        >
                          {u.is_active ? "Suspend" : "Restore"}
                        </button>
                        <button
                          onClick={() => { setActionError(null); setConfirmDelete(u); }}
                          disabled={isSelf}
                          title={isSelf ? "You can’t delete yourself" : "Delete user"}
                          className="rounded border border-border-strong p-1 text-text-2 transition-colors hover:border-red-500/50 hover:text-red-400 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {detailId && <UserDrawer userId={detailId} onClose={() => setDetailId(null)} />}

      {confirmDelete && (
        <DeleteModal
          user={confirmDelete}
          pending={del.isPending}
          onCancel={() => setConfirmDelete(null)}
          onConfirm={() => del.mutate(confirmDelete.user_id)}
        />
      )}
    </div>
  );
}

function DeleteModal({
  user,
  pending,
  onCancel,
  onConfirm,
}: {
  user: AdminUserRow;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const label = user.email ?? user.user_id;
  const [typed, setTyped] = useState("");
  const armed = typed.trim().toLowerCase() === label.toLowerCase();
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
      <div className="w-full max-w-md rounded-xl border border-red-500/30 bg-bg-1 p-5">
        <div className="mb-3 flex items-center gap-2 text-red-400">
          <AlertTriangle className="h-5 w-5" />
          <h2 className="font-clash text-lg font-semibold">Delete user</h2>
        </div>
        <p className="text-sm text-text-1">
          This permanently deletes <span className="font-medium text-text-0">{label}</span> and
          all of their memories and conversations. This cannot be undone.
        </p>
        <p className="mt-3 text-xs text-text-2">Type the email/id to confirm:</p>
        <input
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder={label}
          className="mt-1 w-full rounded-md border border-border-strong bg-bg-2 px-3 py-2 text-sm text-text-0 outline-none focus:border-red-500/50"
        />
        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onCancel} className="rounded-md border border-border-strong px-3 py-1.5 text-sm text-text-1 hover:text-text-0">
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={!armed || pending}
            className="rounded-md bg-red-500 px-3 py-1.5 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {pending ? "Deleting…" : "Delete permanently"}
          </button>
        </div>
      </div>
    </div>
  );
}

function UserDrawer({ userId, onClose }: { userId: string; onClose: () => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ["admin", "user", userId],
    queryFn: () => api.admin.userDetail(userId),
  });
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/40" onClick={onClose}>
      <div
        className="flex h-full w-full max-w-md flex-col overflow-y-auto border-l border-border bg-bg-1 p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h2 className="truncate font-clash text-lg font-semibold text-text-0">
              {data?.email ?? userId}
            </h2>
            <p className="truncate font-mono text-xs text-text-2">{userId}</p>
          </div>
          <button onClick={onClose} className="rounded p-1 text-text-2 hover:text-text-0">
            <X className="h-4 w-4" />
          </button>
        </div>

        {isLoading || !data ? (
          <p className="text-sm text-text-2">Loading…</p>
        ) : (
          <>
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <PlanBadge plan={data.plan} />
              <span className={`text-xs ${data.is_active ? "text-accent" : "text-text-2"}`}>
                {data.is_active ? "active" : "suspended"}
              </span>
              {data.has_google && <span className="text-xs text-text-2">· Google</span>}
              {data.has_password && <span className="text-xs text-text-2">· password</span>}
            </div>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <Stat label="Joined" value={fmtDate(data.created_at)} />
              <Stat label="Last seen" value={relativeTime(data.last_seen)} />
              <Stat label="Memories" value={String(data.memory_count)} />
              <Stat label="Conversations" value={String(data.conversation_count)} />
              <Stat label="API keys" value={String(data.key_count)} />
              <Stat label="Total events" value={String(data.event_count)} />
              <Stat label="Storage" value={fmtBytes(data.bytes_used)} />
            </dl>
            <div className="mt-5">
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-text-2">Recent activity</p>
              <ul className="space-y-1">
                {data.recent.length === 0 ? (
                  <li className="text-sm text-text-2">No activity recorded.</li>
                ) : (
                  data.recent.map((e, i) => (
                    <li key={i} className="flex items-center justify-between gap-2 text-xs">
                      <span className="truncate font-mono text-text-1">{e.method} {e.path}</span>
                      <span className="shrink-0 text-text-2">{relativeTime(e.ts)}</span>
                    </li>
                  ))
                )}
              </ul>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-bg-2 px-3 py-2">
      <dt className="text-xs text-text-2">{label}</dt>
      <dd className="mt-0.5 text-text-0 tnum">{value}</dd>
    </div>
  );
}
