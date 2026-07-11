"use client";

import { useState, type FormEvent } from "react";
import { ArrowUp } from "lucide-react";
import { useRecallStream } from "@/hooks/use-recall-stream";
import { PipelineDots } from "./PipelineDots";
import { RecallResults } from "./RecallResults";

/** Empty-state of the inspector: ask GML, watch the real pipeline, get hits. */
export function RecallPanel() {
  const [q, setQ] = useState("");
  const { status, lit, data, error, run } = useRecallStream();

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const query = q.trim();
    if (query) run(query, 8);
  };

  return (
    <div data-coach="recall" className="flex h-full flex-col p-5">
      <h2 className="font-mono text-[10px] uppercase tracking-wider text-text-2">Recall</h2>
      <p className="mt-1 text-sm text-text-1">
        Ask anything. Akhrot retrieves, reranks, and explains why.
      </p>

      <form onSubmit={submit} className="mt-4">
        <div className="flex items-center gap-2 rounded-md border border-border bg-bg-2 px-3 focus-within:border-border-strong">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="what port does auth_service run on?"
            className="flex-1 bg-transparent py-2.5 text-sm text-text-0 placeholder:text-text-2 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!q.trim() || status === "running"}
            className="flex h-6 w-6 items-center justify-center rounded-sm bg-accent text-bg-0 disabled:opacity-30"
          >
            <ArrowUp className="h-3.5 w-3.5" />
          </button>
        </div>
      </form>

      {(status === "running" || status === "done") && (
        <div className="mt-3 flex items-center gap-3">
          <PipelineDots lit={lit} running={status === "running"} />
          <span className="font-mono text-[11px] text-text-2">
            {status === "running" ? "running pipeline…" : `${data?.results.length ?? 0} hits`}
          </span>
        </div>
      )}

      {status === "done" && data?.sam_reasoning && (
        <p className="mt-3 rounded-md border border-border bg-bg-2 p-2.5 text-[11px] leading-relaxed text-text-1">
          <span className="text-text-2">SAM: </span>
          {data.sam_reasoning}
        </p>
      )}

      <div className="mt-4 flex-1 overflow-y-auto">
        {status === "done" && data && <RecallResults results={data.results} />}
        {status === "error" && <p className="text-xs text-cluster-3">{error}</p>}
      </div>
    </div>
  );
}
