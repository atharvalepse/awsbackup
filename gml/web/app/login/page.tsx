"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { setToken } from "@/lib/auth";
import { ParticleField } from "@/components/ui/ParticleField";

type Mode = "login" | "signup";

// Google Identity Services is loaded as an external script and attaches a
// `google` global; declare the slice of its surface we touch.
declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (cfg: {
            client_id: string;
            callback: (resp: { credential?: string }) => void;
          }) => void;
          renderButton: (el: HTMLElement, opts: Record<string, unknown>) => void;
        };
      };
    };
  }
}

const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

// Only allow returning to same-origin paths starting with "/". Anything else
// (full URLs, "//evil.com", javascript: schemes) is silently ignored so a
// malicious deep link can't bounce a freshly-signed-in user to another site.
function safeReturnTo(raw: string | null): string {
  if (!raw) return "/app";
  if (!raw.startsWith("/") || raw.startsWith("//")) return "/app";
  return raw;
}

const STEPS = [
  { n: "1", title: "Capture", desc: "Your AI remembers what matters — automatically." },
  { n: "2", title: "Recall", desc: "Semantic search across every past conversation." },
  { n: "3", title: "Inject", desc: "Into Claude, Cursor, anything — over MCP." },
];

export default function LoginPage() {
  const router = useRouter();
  const [returnTo, setReturnTo] = useState("/app");
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setReturnTo(safeReturnTo(params.get("returnTo")));
  }, []);
  const qc = useQueryClient();
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [inviteCode, setInviteCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res =
        mode === "login"
          ? await api.auth.login(email, password)
          : await api.auth.signup(email, password, inviteCode);
      qc.clear();
      setToken(res.access_token);
      router.replace(returnTo);
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "Something went wrong. Try again.";
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  // Latest mode/inviteCode/returnTo, read inside the (long-lived) Google
  // credential callback so it never closes over stale state.
  const googleBtnRef = useRef<HTMLDivElement>(null);
  const liveRef = useRef({ mode, inviteCode, returnTo });
  liveRef.current = { mode, inviteCode, returnTo };

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) return;

    async function onGoogleCredential(resp: { credential?: string }) {
      if (!resp.credential) return;
      setError(null);
      setBusy(true);
      try {
        const { mode: m, inviteCode: code, returnTo: ret } = liveRef.current;
        // Pass the invite code only in signup mode — login of an existing
        // account never needs one, and unknown emails get a clear 403.
        const res = await api.auth.google(
          resp.credential,
          m === "signup" ? code : undefined,
        );
        qc.clear();
        setToken(res.access_token);
        router.replace(ret);
      } catch (err) {
        const msg =
          err instanceof ApiError ? err.message : "Google sign-in failed. Try again.";
        setError(msg);
      } finally {
        setBusy(false);
      }
    }

    function render(): boolean {
      const g = window.google;
      if (!g?.accounts?.id || !googleBtnRef.current) return false;
      g.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID!,
        callback: onGoogleCredential,
      });
      googleBtnRef.current.innerHTML = "";
      g.accounts.id.renderButton(googleBtnRef.current, {
        type: "standard",
        theme: "outline",
        size: "large",
        text: "continue_with",
        shape: "pill",
        logo_alignment: "center",
        width: 340,
      });
      return true;
    }

    if (render()) return;
    const existing = document.getElementById("gis-client");
    if (existing) {
      existing.addEventListener("load", render);
      return;
    }
    const s = document.createElement("script");
    s.src = "https://accounts.google.com/gsi/client";
    s.async = true;
    s.defer = true;
    s.id = "gis-client";
    s.onload = render;
    document.body.appendChild(s);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const inputCls =
    "w-full rounded-xl border border-border bg-bg-1/70 px-4 py-3 text-sm text-text-0 " +
    "placeholder:text-text-2 outline-none transition " +
    "focus:border-accent/50 focus:bg-bg-1 focus:ring-1 focus:ring-accent/30";

  return (
    <div className="akhrot flex min-h-screen w-screen bg-bg-0 p-3 text-text-0 sm:p-4">
      <style>{`
        @keyframes gml-rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
        @keyframes gml-glow{0%,100%{transform:translate(-50%,0) scale(1);opacity:.85}50%{transform:translate(-46%,4%) scale(1.12);opacity:1}}
        .gml-rise{animation:gml-rise .5s cubic-bezier(.2,.7,.2,1) both}
      `}</style>

      {/* ───────── LEFT — brand panel ───────── */}
      <aside className="relative hidden w-[44%] shrink-0 flex-col justify-between overflow-hidden rounded-3xl p-11 lg:flex">
        {/* brand gradient: signal-green glow fading into near-black */}
        <div className="absolute inset-0" style={{ background: "linear-gradient(165deg,#13180c 0%,#0b0d08 45%,#070707 100%)" }} />
        <div className="absolute left-1/2 top-[-10%] h-[34rem] w-[34rem] rounded-full blur-[110px]"
          style={{ background: "radial-gradient(circle, rgba(184,255,61,0.45), transparent 65%)", animation: "gml-glow 12s ease-in-out infinite" }} />
        <ParticleField />
        <div className="absolute inset-0 opacity-[0.05]"
          style={{ backgroundImage: "linear-gradient(#fff 1px,transparent 1px),linear-gradient(90deg,#fff 1px,transparent 1px)", backgroundSize: "46px 46px" }} />

        <div className="relative z-10 flex items-center gap-2.5">
          <img src="/LOGO.svg" alt="Akhrot" className="h-6 w-auto" />
        </div>

        <div className="relative z-10">
          <h2 className="font-clash text-[40px] font-semibold leading-[1.05] tracking-[-0.02em] text-white">
            Your memory,
            <br />
            remembered<span className="text-accent">.</span>
          </h2>
          <p className="mt-3 max-w-sm text-sm leading-relaxed text-white/55">
            A persistent memory layer for your AI. Three steps, then it just works.
          </p>

          <div className="mt-8 space-y-2.5">
            {STEPS.map((s, i) => (
              <div
                key={s.n}
                className={
                  "flex items-center gap-3.5 rounded-2xl border px-4 py-3.5 backdrop-blur-sm transition " +
                  (i === 0
                    ? "border-accent/40 bg-accent/[0.12]"
                    : "border-white/10 bg-white/[0.04]")
                }
              >
                <span
                  className={
                    "flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold " +
                    (i === 0 ? "bg-accent text-bg-0" : "bg-white/10 text-white/70")
                  }
                >
                  {s.n}
                </span>
                <div>
                  <div className="text-sm font-medium text-white">{s.title}</div>
                  <div className="text-xs text-white/45">{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* ───────── RIGHT — form panel ───────── */}
      <main className="gml-rise relative flex flex-1 items-center justify-center px-4 py-8">
        <div className="w-full max-w-[380px]">
          {/* logo on mobile (brand panel hidden) */}
          <img src="/LOGO.svg" alt="Akhrot" className="mb-8 h-6 w-auto lg:hidden" />

          <h1 className="font-clash text-[30px] font-semibold leading-tight tracking-[-0.02em] text-text-0">
            {mode === "login" ? "Welcome back" : "Create your account"}
          </h1>
          <p className="mb-7 mt-1.5 text-sm text-text-2">
            {mode === "login"
              ? "Enter your details to sign in."
              : "Enter your details and invite code to join."}
          </p>

          <form onSubmit={onSubmit} className="space-y-3.5">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-text-1">Email</label>
              <input type="email" required autoComplete="email" placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} className={inputCls} />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-text-1">Password</label>
              <input type="password" required minLength={mode === "signup" ? 8 : undefined} autoComplete={mode === "login" ? "current-password" : "new-password"} placeholder="••••••••" value={password} onChange={(e) => setPassword(e.target.value)} className={inputCls} />
              {mode === "signup" && (
                <p className="mt-1.5 text-xs text-text-2">Must be at least 8 characters.</p>
              )}
            </div>
            {mode === "signup" && (
              <div>
                <label className="mb-1.5 block text-xs font-medium text-text-1">Invite code</label>
                <input type="text" required autoComplete="off" placeholder="AKH-1A2B3C4D" value={inviteCode} onChange={(e) => setInviteCode(e.target.value.toUpperCase())} className={`${inputCls} font-mono tracking-[0.15em]`} />
              </div>
            )}

            {error && (
              <p className="rounded-lg border border-red-500/25 bg-red-500/10 px-3 py-2 text-sm text-red-400">{error}</p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="mt-1 w-full rounded-xl bg-accent px-4 py-3 text-sm font-semibold text-bg-0 shadow-[0_0_30px_-8px_var(--accent-glow)] transition hover:brightness-110 active:brightness-95 disabled:opacity-50"
            >
              {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Sign up"}
            </button>
          </form>

          {GOOGLE_CLIENT_ID && (
            <>
              <div className="my-6 flex items-center gap-3">
                <span className="h-px flex-1 bg-border" />
                <span className="text-xs uppercase tracking-wider text-text-2">or</span>
                <span className="h-px flex-1 bg-border" />
              </div>
              {/* Google Identity Services renders its button into this div. */}
              <div ref={googleBtnRef} className="flex justify-center" />
              {mode === "signup" && (
                <p className="mt-2 text-center text-xs text-text-2">
                  Signing up with Google still needs a valid invite code above.
                </p>
              )}
            </>
          )}

          <p className="mt-7 text-center text-sm text-text-2">
            {mode === "login" ? "Don't have an account? " : "Already have an account? "}
            <button
              type="button"
              onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(null); }}
              className="font-medium text-accent hover:underline"
            >
              {mode === "login" ? "Sign up" : "Log in"}
            </button>
          </p>
        </div>
      </main>
    </div>
  );
}
