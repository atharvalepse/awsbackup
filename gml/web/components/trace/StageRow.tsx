"use client";

import { useState } from "react";
import { ChevronRight } from "lucide-react";
import type { TraceStage } from "@/lib/types";
import { cn } from "@/lib/utils";

// Stage health by duration: the brief's thresholds.
function healthColor(ms: number): string {
  if (ms < 200) return "var(--cluster-5)"; // green/mint
  if (ms < 1000) return "var(--cluster-4)"; // amber
  return "var(--cluster-3)"; // red/coral
}

export function StageRow({ stage, maxMs }: { stage: TraceStage; maxMs: number }) {
  const [open, setOpen] = useState(false);
  const pct = maxMs > 0 ? (stage.duration_ms / maxMs) * 100 : 0;
  const bytes = JSON.stringify(stage.output).length;

  return (
    <div className="border-b border-border last:border-0">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 py-2.5 text-left hover:bg-bg-2"
      >
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-text-2 transition-transform duration-[180ms]",
            open && "rotate-90",
          )}
        />
        <span className="w-44 shrink-0 truncate font-mono text-xs text-text-0">
          {stage.stage}
        </span>

        {/* duration bar */}
        <div className="flex h-4 flex-1 items-center">
          <div
            className="h-1.5 rounded-full transition-[width] duration-300 ease-out"
            style={{ width: `${Math.max(pct, 2)}%`, background: healthColor(stage.duration_ms) }}
          />
        </div>

        <span className="tnum w-16 shrink-0 text-right font-mono text-xs text-text-1">
          {stage.duration_ms}ms
        </span>
        <span className="w-20 shrink-0 text-right font-mono text-[10px] text-text-2">
          {bytes}B out
        </span>
      </button>

      {open && (
        <pre className="mb-2 ml-7 max-h-56 overflow-auto rounded-md border border-border bg-bg-0 p-3 font-mono text-[11px] leading-relaxed text-text-1">
          {JSON.stringify(stage.output, null, 2)}
        </pre>
      )}
    </div>
  );
}
