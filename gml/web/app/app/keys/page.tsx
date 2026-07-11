"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, KeyRound, Plus, Trash2 } from "lucide-react";
import { createMyKey, listMyKeys, revokeMyKey, type NewKey } from "@/lib/api";

const KEYS_QK = ["my-keys"] as const;

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString();
}

export default function KeysPage() {
  const qc = useQueryClient();
  const { data, isLoading, isError, error } = useQuery({
    queryKey: KEYS_QK,
    queryFn: listMyKeys,
  });
  const [created, setCreated] = useState<NewKey | null>(null);
  const [copied, setCopied] = useState(false);

  const create = useMutation({
    mutationFn: (label?: string) => createMyKey(label),
    onSuccess: (k) => {
      setCreated(k);
      setCopied(false);
      qc.invalidateQueries({ queryKey: KEYS_QK });
    },
  });

  const revoke = useMutation({
    mutationFn: (keyOrPreview: string) => revokeMyKey(keyOrPreview),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEYS_QK }),
  });

  const onCreate = () => {
    const label = window.prompt(
      "Label for this key (e.g. “Claude Desktop”, “laptop”):",
    );
    if (label === null) return; // cancelled
    create.mutate(label.trim() || undefined);
  };

  const onCopy = async () => {
    if (!created) return;
    await navigator.clipboard.writeText(created.key);
    setCopied(true);
  };

  const onRevoke = (keyOrPreview: string, label: string | null) => {
    if (
      window.confirm(
        `Revoke ${label ? `“${label}”` : "this key"}? Anything using it will stop working.`,
      )
    ) {
      revoke.mutate(keyOrPreview);
    }
  };

  const keys = data?.keys ?? [];

  return (
    <div className="mx-auto h-full max-w-3xl overflow-y-auto px-8 py-10">
      <header className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="font-clash text-2xl font-semibold tracking-tight text-text-0">
            API keys<span className="text-accent">.</span>
          </h1>
          <p className="mt-1 max-w-lg text-sm text-text-1">
            Personal <code className="text-text-0">gml_</code> bearer keys for the
            Claude Desktop extension and the API. Scoped to your account and your
            memories.
          </p>
        </div>
        <button
          onClick={onCreate}
          disabled={create.isPending}
          className="flex shrink-0 items-center gap-1.5 rounded-md border border-border-strong px-3 py-2 text-sm text-text-1 transition-colors duration-[180ms] ease-out hover:border-accent hover:text-accent disabled:opacity-50"
        >
          <Plus className="h-4 w-4" /> Create key
        </button>
      </header>

      {/* The full secret — shown exactly once, right after creation. */}
      {created && (
        <div className="mb-6 rounded-md border border-accent bg-bg-1 p-4 shadow-glow">
          <div className="mb-2 flex items-center gap-2 text-sm text-text-0">
            <KeyRound className="h-4 w-4 text-accent" />
            New key{created.label ? ` · ${created.label}` : ""}
          </div>
          <div className="flex items-center gap-2">
            <code className="min-w-0 flex-1 truncate rounded-sm bg-bg-3 px-2 py-1.5 font-mono text-sm text-text-0">
              {created.key}
            </code>
            <button
              onClick={onCopy}
              title="Copy"
              className="flex items-center gap-1 rounded-md border border-border-strong px-2 py-1.5 text-xs text-text-1 transition-colors hover:border-accent hover:text-accent"
            >
              {copied ? <Check className="h-4 w-4 text-accent" /> : <Copy className="h-4 w-4" />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <p className="mt-2 text-xs text-cluster-4">
            Save this now — it won’t be shown again.
          </p>
          <button
            onClick={() => setCreated(null)}
            className="mt-3 text-xs text-text-2 hover:text-text-1"
          >
            Done
          </button>
        </div>
      )}

      {isLoading && <p className="text-sm text-text-2">Loading…</p>}
      {isError && (
        <p className="text-sm text-cluster-3">
          Couldn’t load keys: {(error as Error).message}
        </p>
      )}

      {!isLoading && !isError && keys.length === 0 && (
        <p className="rounded-md border border-border border-dashed bg-bg-1 p-6 text-center text-sm text-text-2">
          No API keys yet. Create one to connect the desktop extension.
        </p>
      )}

      {keys.length > 0 && (
        <ul className="divide-y divide-border overflow-hidden rounded-md border border-border bg-bg-1">
          {keys.map((k) => (
            <li
              key={k.key_preview}
              className="flex items-center justify-between gap-4 px-4 py-3"
            >
              <div className="min-w-0">
                <div className="truncate text-sm text-text-0">
                  {k.label || <span className="text-text-2">Unlabeled key</span>}
                </div>
                <div className="mt-0.5 flex items-center gap-3 text-xs text-text-2">
                  <span className="font-mono">{k.key_preview}</span>
                  <span>created {fmtDate(k.created_at)}</span>
                </div>
              </div>
              <button
                onClick={() => onRevoke(k.key_preview, k.label)}
                disabled={revoke.isPending}
                title="Revoke"
                className="flex shrink-0 items-center gap-1 rounded-md px-2 py-1.5 text-xs text-text-2 transition-colors hover:bg-bg-3 hover:text-cluster-3 disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" /> Revoke
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
