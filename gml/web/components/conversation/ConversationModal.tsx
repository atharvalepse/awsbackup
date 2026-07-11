"use client";

import { useEffect } from "react";
import { X, ExternalLink, Trash2 } from "lucide-react";
import { useConversation, useDeleteConversation } from "@/hooks/use-gml";
import { useToasts } from "@/stores/toast-store";
import { relativeTime } from "@/lib/utils";

export function ConversationModal({
  id,
  onClose,
}: {
  id: string;
  onClose: () => void;
}) {
  const { data: card, isLoading } = useConversation(id);
  const del = useDeleteConversation();
  const push = useToasts((s) => s.push);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function handleDelete() {
    try {
      await del.mutateAsync(id);
      push({ title: "Conversation deleted", kind: "success" });
      onClose();
    } catch {
      push({ title: "Delete failed", kind: "error" });
    }
  }

  return (
    <div
      className="fixed inset-0 z-[55] flex items-center justify-center bg-[rgba(5,5,7,0.7)] p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
        className="grid max-h-[86vh] w-full max-w-4xl grid-cols-1 overflow-hidden rounded-xl border border-border-strong bg-bg-1 md:grid-cols-[1.4fr_1fr]"
      >
        {/* Left: the turn */}
        <div className="overflow-y-auto p-6">
          <div className="mb-4 flex items-start gap-3">
            <h2 className="flex-1 text-lg font-semibold leading-snug text-text-0">
              {card?.title || "Conversation"}
            </h2>
            <button
              onClick={onClose}
              aria-label="Close"
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border text-text-2 hover:bg-bg-3 hover:text-text-0"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {isLoading ? (
            <p className="text-sm text-text-2">Loading…</p>
          ) : (
            <>
              <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-text-2">
                You asked
              </div>
              <div className="mb-4 whitespace-pre-wrap break-words rounded-md border border-border bg-bg-0 p-3 text-sm leading-relaxed text-text-0">
                {card?.user_prompt || "—"}
              </div>
              <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-text-2">
                AI replied
              </div>
              <div className="whitespace-pre-wrap break-words text-sm leading-relaxed text-text-1">
                {card?.ai_response || "—"}
              </div>
            </>
          )}
        </div>

        {/* Right: summary + memories + meta */}
        <div className="overflow-y-auto border-t border-border bg-bg-2 p-5 md:border-l md:border-t-0">
          <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-text-2">
            Summary
          </div>
          <p className="mb-5 text-xs leading-relaxed text-text-1">
            {card?.summary || "No summary yet."}
          </p>

          <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-text-2">
            Memories ({card?.facts.length ?? 0})
          </div>
          {card && card.facts.length > 0 ? (
            <ul className="flex flex-col gap-2">
              {card.facts.map((f, i) => (
                <li
                  key={f.id ?? i}
                  className="rounded-md border border-border bg-bg-0 p-2.5 text-xs leading-relaxed text-text-1"
                >
                  {f.entity && (
                    <span className="mr-1 font-mono text-[10px] text-text-2">
                      {f.entity}:
                    </span>
                  )}
                  {f.value || f.content}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-text-2">No atomic memories extracted yet.</p>
          )}

          <div className="mt-5 flex flex-wrap items-center gap-3 text-[10px] text-text-2">
            <span>{card?.source_model}</span>
            <span>{card?.created_at ? relativeTime(card.created_at) : ""}</span>
            {card?.source_url && (
              <a
                href={card.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-text-1 hover:text-accent"
              >
                Visit source <ExternalLink className="h-3 w-3" />
              </a>
            )}
            <button
              onClick={handleDelete}
              className="ml-auto inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-text-2 hover:border-[var(--cluster-3)] hover:text-[var(--cluster-3)]"
            >
              <Trash2 className="h-3 w-3" /> Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
