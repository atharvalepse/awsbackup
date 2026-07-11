"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, Plus } from "lucide-react";
import { api } from "@/lib/api";
import { fmtDate } from "@/components/admin/kit";

const INVITES_QK = ["admin", "invites"] as const;

export default function InvitesPage() {
  const qc = useQueryClient();
  const [copied, setCopied] = useState<string | null>(null);

  const invites = useQuery({ queryKey: INVITES_QK, queryFn: api.admin.listInviteCodes });
  const gen = useMutation({
    mutationFn: api.admin.generateInviteCode,
    onSuccess: () => qc.invalidateQueries({ queryKey: INVITES_QK }),
  });

  function copy(code: string) {
    navigator.clipboard.writeText(code);
    setCopied(code);
    setTimeout(() => setCopied((c) => (c === code ? null : c)), 1500);
  }

  const codes = invites.data?.codes ?? [];
  const unused = codes.filter((c) => !c.used_by_email);
  const used = codes.filter((c) => c.used_by_email);

  return (
    <div className="px-6 py-6">
      <header className="mb-5 flex items-end justify-between gap-3">
        <div>
          <h1 className="font-clash text-2xl font-semibold tracking-tight text-text-0">
            Invites<span className="text-accent">.</span>
          </h1>
          <p className="mt-1 text-sm text-text-1">
            Single-use codes that grant new users access. {unused.length} unused.
          </p>
        </div>
        <button
          onClick={() => gen.mutate()}
          disabled={gen.isPending}
          className="flex items-center gap-1.5 rounded-md border border-border-strong px-3 py-2 text-sm text-text-1 transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
        >
          <Plus className="h-3.5 w-3.5" />
          {gen.isPending ? "Generating…" : "New code"}
        </button>
      </header>

      <section className="mb-6">
        <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-text-2">Unused ({unused.length})</h2>
        {invites.isLoading ? (
          <p className="text-sm text-text-2">Loading…</p>
        ) : unused.length === 0 ? (
          <p className="text-sm italic text-text-2">No unused codes. Generate one above.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {unused.map((c) => (
              <button
                key={c.code}
                onClick={() => copy(c.code)}
                className="flex items-center gap-2 rounded-md border border-border-strong bg-bg-2 px-3 py-1.5 font-mono text-sm tracking-wider text-text-0 transition-colors hover:border-accent/40 hover:text-accent"
              >
                {c.code}
                {copied === c.code ? <Check className="h-3.5 w-3.5 text-accent" /> : <Copy className="h-3.5 w-3.5 text-text-2" />}
              </button>
            ))}
          </div>
        )}
      </section>

      {used.length > 0 && (
        <section>
          <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-text-2">Redeemed ({used.length})</h2>
          <div className="overflow-x-auto rounded-xl border border-border bg-bg-2">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-xs text-text-2">
                  <th className="px-4 py-2 font-medium">Code</th>
                  <th className="px-4 py-2 font-medium">Used by</th>
                  <th className="px-4 py-2 font-medium">When</th>
                </tr>
              </thead>
              <tbody>
                {used.map((c) => (
                  <tr key={c.code} className="border-t border-border/60">
                    <td className="px-4 py-2 font-mono text-xs text-text-2 line-through">{c.code}</td>
                    <td className="px-4 py-2 text-text-1">{c.used_by_email}</td>
                    <td className="px-4 py-2 text-text-2">{fmtDate(c.used_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
