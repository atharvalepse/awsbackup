"use client";

import { useState, type FormEvent, type ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  LayoutDashboard,
  LogOut,
  ScrollText,
  ShieldCheck,
  Ticket,
  Users,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { clearToken, getToken, setToken } from "@/lib/auth";

const NAV = [
  { href: "/admin", label: "Overview", icon: LayoutDashboard },
  { href: "/admin/users", label: "Users", icon: Users },
  { href: "/admin/activity", label: "Activity", icon: Activity },
  { href: "/admin/invites", label: "Invites", icon: Ticket },
  { href: "/admin/audit", label: "Audit log", icon: ScrollText },
];

const ME_QK = ["admin", "me"] as const;

export default function AdminLayout({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [authed, setAuthed] = useState<boolean>(() => getToken() !== null);

  const me = useQuery({
    queryKey: ME_QK,
    queryFn: api.auth.me,
    enabled: authed,
    retry: false,
  });

  const isAdmin = !!me.data && (me.data.is_master || me.data.plan === "admin");

  function signOut() {
    clearToken();
    setAuthed(false);
    qc.clear();
  }

  // 1) No token, or token rejected → login screen.
  if (!authed || (me.isError && me.error instanceof ApiError && me.error.status === 401)) {
    return <AdminLogin onSuccess={() => { setAuthed(true); qc.invalidateQueries({ queryKey: ME_QK }); }} />;
  }

  // 2) Resolving identity.
  if (me.isLoading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-bg-0 text-sm text-text-2">
        Checking access…
      </div>
    );
  }

  // 3) Authenticated but not an admin → denied.
  if (!isAdmin) {
    return (
      <div className="flex h-screen w-screen flex-col items-center justify-center gap-4 bg-bg-0 text-center">
        <ShieldCheck className="h-8 w-8 text-text-2" />
        <div>
          <p className="font-clash text-lg text-text-0">Admins only</p>
          <p className="mt-1 text-sm text-text-2">
            {me.data?.email ?? "This account"} doesn’t have admin access.
          </p>
        </div>
        <button
          onClick={signOut}
          className="rounded-md border border-border-strong px-3 py-1.5 text-sm text-text-1 hover:border-accent hover:text-accent"
        >
          Sign out
        </button>
      </div>
    );
  }

  // 4) Admin → the console shell.
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-0 text-text-0">
      <Sidebar email={me.data?.email ?? null} onSignOut={signOut} />
      <main className="min-w-0 flex-1 overflow-y-auto">{children}</main>
    </div>
  );
}

function Sidebar({ email, onSignOut }: { email: string | null; onSignOut: () => void }) {
  const pathname = usePathname();
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-bg-1">
      <div className="flex items-center gap-2 border-b border-border px-4 py-4">
        <ShieldCheck className="h-5 w-5 text-accent" />
        <span className="font-clash text-base font-semibold tracking-tight">
          Akhrot Admin
        </span>
      </div>
      <nav className="flex-1 space-y-1 p-2">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/admin" ? pathname === "/admin" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors ${
                active
                  ? "bg-bg-3 text-text-0"
                  : "text-text-1 hover:bg-bg-2 hover:text-text-0"
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-border p-3">
        <p className="truncate px-1 text-xs text-text-2" title={email ?? undefined}>
          {email ?? "admin"}
        </p>
        <button
          onClick={onSignOut}
          className="mt-2 flex w-full items-center gap-2 rounded-md px-1 py-1.5 text-xs text-text-1 transition-colors hover:text-accent"
        >
          <LogOut className="h-3.5 w-3.5" />
          Sign out
        </button>
      </div>
    </aside>
  );
}

function AdminLogin({ onSuccess }: { onSuccess: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await api.auth.login(email.trim(), password);
      setToken(res.access_token);
      onSuccess();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-bg-0 px-4">
      <form
        onSubmit={submit}
        className="w-full max-w-sm rounded-xl border border-border bg-bg-1 p-6"
      >
        <div className="mb-5 flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-accent" />
          <h1 className="font-clash text-lg font-semibold text-text-0">
            Akhrot Admin
          </h1>
        </div>
        {error && (
          <div className="mb-3 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}
        <label className="mb-1 block text-xs text-text-2">Email</label>
        <input
          type="email"
          autoComplete="username"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="mb-3 w-full rounded-md border border-border-strong bg-bg-2 px-3 py-2 text-sm text-text-0 outline-none focus:border-accent"
        />
        <label className="mb-1 block text-xs text-text-2">Password</label>
        <input
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          className="mb-4 w-full rounded-md border border-border-strong bg-bg-2 px-3 py-2 text-sm text-text-0 outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={busy}
          className="w-full rounded-md bg-accent px-4 py-2 text-sm font-medium text-black transition-opacity hover:opacity-90 disabled:opacity-50"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
