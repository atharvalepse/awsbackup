"use client";

import { useMemories } from "@/hooks/use-gml";
import { useAppStore } from "@/stores/app-store";
import { clusterHex } from "@/lib/cluster-colors";
import { relativeTime, cn } from "@/lib/utils";

/** Tabular alternative to the graph. Reads /api/memories, honours cluster isolation. */
export function MemoryList() {
  const isolated = useAppStore((s) => s.isolatedCluster);
  const selectedId = useAppStore((s) => s.selectedId);
  const select = useAppStore((s) => s.select);
  const { data, isLoading } = useMemories(isolated ?? undefined);

  if (isLoading)
    return <div className="p-8 text-sm text-text-2">Loading memories…</div>;

  return (
    <div className="h-full overflow-y-auto px-6 py-5">
      <div className="mb-3 flex items-baseline gap-2">
        <h1 className="text-sm font-medium text-text-0">Memories</h1>
        <span className="tnum font-mono text-xs text-text-2">{data?.total ?? 0}</span>
      </div>
      <div className="overflow-hidden rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left font-mono text-[10px] uppercase tracking-wider text-text-2">
              <th className="px-3 py-2 font-normal">Entity</th>
              <th className="px-3 py-2 font-normal">Value</th>
              <th className="px-3 py-2 text-right font-normal">Conf</th>
              <th className="px-3 py-2 text-right font-normal">Imp</th>
              <th className="px-3 py-2 text-right font-normal">Created</th>
            </tr>
          </thead>
          <tbody>
            {data?.memories.map((m) => (
              <tr
                key={m.id}
                onClick={() => select(m.id)}
                className={cn(
                  "cursor-pointer border-b border-border/60 transition-colors hover:bg-bg-2",
                  selectedId === m.id && "bg-bg-2",
                )}
              >
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ background: clusterHex(m.cluster_id) }}
                    />
                    <span className="truncate text-text-0">
                      {m.entity ?? m.content.slice(0, 40)}
                    </span>
                  </div>
                </td>
                <td className="max-w-[200px] truncate px-3 py-2 font-mono text-xs text-text-1">
                  {m.value ?? "—"}
                </td>
                <td className="tnum px-3 py-2 text-right font-mono text-xs text-text-1">
                  {m.confidence.toFixed(2)}
                </td>
                <td className="tnum px-3 py-2 text-right font-mono text-xs text-text-1">
                  {m.importance.toFixed(2)}
                </td>
                <td className="px-3 py-2 text-right text-xs text-text-2">
                  {relativeTime(m.timestamp)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
