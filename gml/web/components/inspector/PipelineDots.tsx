"use client";

import { cn } from "@/lib/utils";

// The 7 canonical pipeline stages, in order.
export const PIPELINE_STAGES = [
  "classifier",
  "embedder",
  "retriever",
  "reranker",
  "sam",
  "assembler",
  "translator",
] as const;

/**
 * Map a backend stage name (which is finer-grained — retriever_probe,
 * retriever_top_k, sam_skipped, sam_resolve_conflicts, …) to its canonical
 * 0–6 dot index. Returns -1 for unknown stages.
 */
export function canonicalStageIndex(stage: string): number {
  if (stage.startsWith("classifier")) return 0;
  if (stage.startsWith("embedder")) return 1;
  if (stage.startsWith("retriever")) return 2;
  if (stage.startsWith("reranker")) return 3;
  if (stage.startsWith("sam")) return 4;
  if (stage.startsWith("assembler")) return 5;
  if (stage.startsWith("translator")) return 6;
  return -1;
}

/**
 * Seven dots, lit by *real* backend progress. `lit` is the number of stages
 * completed so far (0–7), derived from streamed SSE `stage` events.
 */
export function PipelineDots({ lit, running }: { lit: number; running: boolean }) {
  return (
    <div className="flex items-center gap-1.5" title={PIPELINE_STAGES.join(" → ")}>
      {PIPELINE_STAGES.map((s, i) => (
        <span
          key={s}
          className={cn(
            "h-1.5 w-1.5 rounded-full transition-colors duration-[160ms] ease-out",
            i < lit ? "bg-accent" : "bg-bg-3",
            running && i === lit && "animate-pulse bg-accent/50",
          )}
        />
      ))}
    </div>
  );
}
