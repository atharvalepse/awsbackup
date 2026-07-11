"use client";

import { useRef, useState, type FormEvent } from "react";
import { ArrowUp } from "lucide-react";
import { api } from "@/lib/api";
import type { TraceResponse } from "@/lib/types";
import { StageRow } from "@/components/trace/StageRow";
import { PipelineDots, canonicalStageIndex } from "@/components/inspector/PipelineDots";

type Status = "idle" | "running" | "done" | "error";

export default function TracePage() {
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [lit, setLit] = useState(0);
  const [data, setData] = useState<TraceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const text = q.trim();
    if (!text) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setStatus("running");
    setLit(0);
    setData(null);
    setError(null);
    try {
      const res = await api.traceStream(
        text,
        (s) => {
          const idx = canonicalStageIndex(s.stage);
          if (idx >= 0) setLit((l) => Math.max(l, idx + 1));
        },
        ac.signal,
      );
      setLit(7);
      setData(res);
      setStatus("done");
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError((err as Error).message);
      setStatus("error");
    }
  };

  const stages = data?.stages ?? [];
  const maxMs = Math.max(1, ...stages.map((s) => s.duration_ms));
  const total = stages.reduce((s, x) => s + x.duration_ms, 0);

  return (
    <div className="mx-auto h-full max-w-3xl overflow-y-auto px-8 py-10">
      <header className="mb-6">
        <h1 className="font-clash text-2xl font-semibold tracking-tight text-text-0">
          Trace<span className="text-accent">.</span>
        </h1>
        <p className="mt-1 max-w-lg text-sm text-text-1">
          Every pipeline stage for a query — timing, I/O size, and the data each
          stage returned. When something feels off, start here.
        </p>
      </header>

      <form onSubmit={submit}>
        <div className="flex items-center gap-2 rounded-md border border-border bg-bg-2 px-3 focus-within:border-border-strong">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Trace a query, e.g. “what port does auth_service run on?”"
            className="flex-1 bg-transparent py-3 text-sm text-text-0 placeholder:text-text-2 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!q.trim() || status === "running"}
            className="flex h-7 w-7 items-center justify-center rounded-sm bg-accent text-bg-0 disabled:opacity-30"
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        </div>
      </form>

      {(status === "running" || status === "done") && (
        <div className="mt-5 flex items-center gap-3">
          <PipelineDots lit={lit} running={status === "running"} />
          <span className="font-mono text-[11px] text-text-2">
            {status === "running" ? "running pipeline…" : `${stages.length} stages · ${total}ms`}
          </span>
        </div>
      )}
      {status === "error" && (
        <p className="mt-5 text-sm text-cluster-3">{error}</p>
      )}

      {status === "done" && data && (
        <div className="mt-6">
          <div className="rounded-lg border border-border bg-bg-1 px-4">
            {stages.map((s, i) => (
              <StageRow key={`${s.stage}-${i}`} stage={s} maxMs={maxMs} />
            ))}
          </div>

          {data.annotations.improved_query && (
            <Section label="Improved query (SAM)">{data.annotations.improved_query}</Section>
          )}
          {data.annotations.sam_reasoning && (
            <Section label="SAM reasoning">{data.annotations.sam_reasoning}</Section>
          )}
          <Section label="Assembled context" mono>
            {data.formatted_context}
          </Section>
        </div>
      )}
    </div>
  );
}

function Section({
  label,
  children,
  mono,
}: {
  label: string;
  children: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="mt-6">
      <h2 className="mb-2 font-mono text-[10px] uppercase tracking-wider text-text-2">
        {label}
      </h2>
      <div
        className={`whitespace-pre-wrap rounded-md border border-border bg-bg-0 p-3 text-sm text-text-1 ${
          mono ? "font-mono text-[11px] leading-relaxed" : ""
        }`}
      >
        {children}
      </div>
    </div>
  );
}
