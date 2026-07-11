"use client";

import { useEffect } from "react";
import { X } from "lucide-react";
import { useRecallStream } from "@/hooks/use-recall-stream";
import { PipelineDots } from "./PipelineDots";
import { RecallResults } from "./RecallResults";

/** Overlay drawer launched by "Recall similar" — streamed full pipeline. */
export function RecallDrawer({ query, onClose }: { query: string; onClose: () => void }) {
  const { status, lit, data, error, run } = useRecallStream();

  useEffect(() => {
    run(query, 8);
    // run once per query
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  return (
    <div className="absolute inset-0 z-30 flex flex-col bg-bg-1/97 backdrop-blur">
      <div className="flex items-center justify-between border-b border-border p-4">
        <div className="min-w-0">
          <div className="font-mono text-[10px] uppercase tracking-wider text-text-2">
            Recall similar
          </div>
          <div className="truncate text-sm text-text-0">{query}</div>
        </div>
        <button
          onClick={onClose}
          className="flex h-7 w-7 items-center justify-center rounded-sm text-text-2 hover:bg-bg-3 hover:text-text-0"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="flex items-center gap-3 px-4 py-3">
        <PipelineDots lit={lit} running={status === "running"} />
        <span className="font-mono text-[11px] text-text-2">
          {status === "running" ? "running…" : `${data?.results.length ?? 0} hits`}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto p-4 pt-0">
        {status === "done" && data && <RecallResults results={data.results} onPick={onClose} />}
        {status === "error" && <p className="text-xs text-cluster-3">{error}</p>}
      </div>
    </div>
  );
}
