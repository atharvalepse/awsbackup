"use client";

import { useClusters } from "@/hooks/use-gml";
import { useAppStore } from "@/stores/app-store";
import { clusterHex } from "@/lib/cluster-colors";
import { cn } from "@/lib/utils";

/** Floating legend, top-right of the canvas. Click a row to isolate its cluster. */
export function ClusterLegend() {
  const { data } = useClusters();
  const { isolatedCluster, isolateCluster } = useAppStore();
  if (!data?.clusters?.length) return null;

  return (
    <div
      data-coach="legend"
      className="absolute right-3 top-3 z-10 w-52 rounded-md border border-border bg-bg-2/90 p-3 backdrop-blur"
    >
      <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-text-2">
        Clusters
      </div>
      <ul className="space-y-0.5">
        {data.clusters.map((c) => {
          const active = isolatedCluster === c.id;
          return (
            <li key={c.id}>
              <button
                onClick={() => isolateCluster(c.id)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-sm px-1.5 py-1 text-left text-xs transition-colors duration-[180ms] ease-out hover:bg-bg-3",
                  active ? "text-text-0" : "text-text-1",
                  isolatedCluster != null && !active && "opacity-50",
                )}
              >
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ background: clusterHex(c.id) }}
                />
                <span className="truncate">{c.label}</span>
                <span className="tnum ml-auto font-mono text-[11px] text-text-2">
                  {c.size}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
