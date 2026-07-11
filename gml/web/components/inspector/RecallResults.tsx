"use client";

import type { RecallResult } from "@/lib/types";
import { useAppStore } from "@/stores/app-store";
import { clusterHex } from "@/lib/cluster-colors";

/** Result cards shared by the inspector recall panel and the "recall similar" drawer. */
export function RecallResults({
  results,
  onPick,
}: {
  results: RecallResult[];
  onPick?: () => void;
}) {
  const select = useAppStore((s) => s.select);
  if (results.length === 0)
    return <p className="px-1 text-xs text-text-2">No matches above threshold.</p>;

  return (
    <ul className="space-y-2">
      {results.map((r) => (
        <li key={r.memory.id}>
          <button
            onClick={() => {
              select(r.memory.id);
              onPick?.();
            }}
            className="w-full rounded-md border border-border bg-bg-2 p-3 text-left transition-colors duration-[180ms] ease-out hover:border-border-strong hover:bg-bg-3"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex min-w-0 items-center gap-2">
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ background: clusterHex(r.memory.cluster_id) }}
                />
                <span className="truncate text-sm text-text-0">
                  {r.memory.entity ?? r.memory.content.slice(0, 48)}
                </span>
              </div>
              <span className="tnum shrink-0 font-mono text-xs text-text-1">
                {r.score.toFixed(3)}
              </span>
            </div>
            {r.memory.value && (
              <p className="mt-1 truncate font-mono text-xs text-text-2">{r.memory.value}</p>
            )}
            {r.why && <p className="mt-1 text-[11px] text-text-2">{r.why}</p>}
          </button>
        </li>
      ))}
    </ul>
  );
}
