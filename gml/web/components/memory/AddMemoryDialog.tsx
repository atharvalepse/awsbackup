"use client";

import { useEffect, useState, type FormEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X, Zap, Brain } from "lucide-react";
import { useAppStore } from "@/stores/app-store";
import { useIngest } from "@/hooks/use-gml";
import { toast } from "@/stores/toast-store";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Add memories by replaying a conversation turn through GML's extraction.
 * Surfaces the system's core duality: a fast regex pass (SDP) vs a deep
 * LLM pass — the brief's "fast save" vs "deep save".
 */
export function AddMemoryDialog() {
  const open = useAppStore((s) => s.addOpen);
  const setOpen = useAppStore((s) => s.setAddOpen);
  const [userQuery, setUserQuery] = useState("");
  const [assistantReply, setAssistantReply] = useState("");
  const [fast, setFast] = useState(true);
  const ingest = useIngest();

  // Reset fields whenever the dialog is opened fresh.
  useEffect(() => {
    if (open) {
      setUserQuery("");
      setAssistantReply("");
      ingest.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!assistantReply.trim()) return;
    try {
      const res = await ingest.mutateAsync({ userQuery, assistantReply, fast });
      if (res.count === 0) {
        toast.info("Nothing extracted", res.detail ?? "No durable facts in that turn.");
      } else {
        toast.success(
          `Saved ${res.count} ${res.count === 1 ? "memory" : "memories"}`,
          `${fast ? "Fast (SDP)" : "Deep (LLM)"} ingest${res.detail ? ` · ${res.detail}` : ""}`,
        );
      }
      setOpen(false);
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        toast.error("Deep save unavailable", "The LLM (Ollama) isn't running. Use Fast save instead.");
        setFast(true);
      } else {
        toast.error("Ingest failed", (err as Error).message);
      }
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[55] flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onClick={() => setOpen(false)}
        >
          <div className="absolute inset-0 bg-[rgba(5,5,7,0.7)] backdrop-blur-sm" />
          <motion.div
            role="dialog"
            aria-modal="true"
            initial={{ opacity: 0, scale: 0.97, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 8 }}
            transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
            onClick={(e) => e.stopPropagation()}
            className="relative z-10 w-full max-w-lg rounded-xl border border-border-strong bg-bg-1 p-6"
          >
            <div className="mb-5 flex items-start justify-between">
              <div>
                <h2 className="font-clash text-lg font-semibold tracking-tight text-text-0">Add a memory</h2>
                <p className="mt-1 text-sm text-text-1">
                  Replay a conversation turn — Akhrot extracts the durable facts.
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="flex h-7 w-7 items-center justify-center rounded-sm text-text-2 hover:bg-bg-3 hover:text-text-0"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <form onSubmit={submit} className="space-y-4">
              <Field label="User said">
                <textarea
                  value={userQuery}
                  onChange={(e) => setUserQuery(e.target.value)}
                  rows={2}
                  placeholder="What database do we use?"
                  className="w-full resize-none rounded-md border border-border bg-bg-2 p-2.5 text-sm text-text-0 placeholder:text-text-2 focus:border-border-strong focus:outline-none"
                />
              </Field>
              <Field label="Assistant replied" required>
                <textarea
                  value={assistantReply}
                  onChange={(e) => setAssistantReply(e.target.value)}
                  rows={3}
                  placeholder="We use PostgreSQL 16 on port 5432 for the orders database."
                  className="w-full resize-none rounded-md border border-border bg-bg-2 p-2.5 text-sm text-text-0 placeholder:text-text-2 focus:border-border-strong focus:outline-none"
                />
              </Field>

              {/* fast vs deep */}
              <div className="grid grid-cols-2 gap-2">
                <ModeCard
                  active={fast}
                  onClick={() => setFast(true)}
                  icon={<Zap className="h-4 w-4" />}
                  title="Fast save"
                  desc="Regex SDP · <50ms · pattern facts"
                />
                <ModeCard
                  active={!fast}
                  onClick={() => setFast(false)}
                  icon={<Brain className="h-4 w-4" />}
                  title="Deep save"
                  desc="LLM · ~7s · catches nuance"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  className="rounded-sm px-3 py-2 text-sm text-text-1 hover:text-text-0"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!assistantReply.trim() || ingest.isPending}
                  className="rounded-sm bg-accent px-4 py-2 text-sm font-medium text-bg-0 transition disabled:opacity-40"
                >
                  {ingest.isPending
                    ? fast
                      ? "Saving…"
                      : "Extracting (~7s)…"
                    : "Save memory"}
                </button>
              </div>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function Field({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block font-mono text-[10px] uppercase tracking-wider text-text-2">
        {label}
        {required && <span className="text-accent"> *</span>}
      </span>
      {children}
    </label>
  );
}

function ModeCard({
  active,
  onClick,
  icon,
  title,
  desc,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  title: string;
  desc: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col gap-1 rounded-md border p-3 text-left transition-colors duration-[180ms] ease-out",
        active
          ? "border-accent bg-bg-2 text-text-0"
          : "border-border bg-bg-1 text-text-1 hover:bg-bg-2",
      )}
    >
      <span className="flex items-center gap-2" style={{ color: active ? "var(--accent)" : undefined }}>
        {icon}
        <span className="text-sm font-medium">{title}</span>
      </span>
      <span className="text-[11px] text-text-2">{desc}</span>
    </button>
  );
}
