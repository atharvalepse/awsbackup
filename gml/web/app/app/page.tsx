"use client";

import dynamic from "next/dynamic";
import { AlertTriangle, Loader2 } from "lucide-react";
import { useGraph } from "@/hooks/use-gml";
import { useAppStore } from "@/stores/app-store";
import { CanvasToolbar } from "@/components/graph/CanvasToolbar";
import { ClusterLegend } from "@/components/graph/ClusterLegend";
import { Inspector } from "@/components/inspector/Inspector";
import { MemoryList } from "@/components/MemoryList";
import { Coachmarks } from "@/components/onboarding/Coachmarks";
import { API_BASE } from "@/lib/api";

// react-force-graph-3d touches window — load client-only, after first paint.
const MemoryGraph = dynamic(() => import("@/components/graph/MemoryGraph"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-text-2">
      <Loader2 className="h-5 w-5 animate-spin" />
    </div>
  ),
});

export default function AppPage() {
  const { data, isError, error } = useGraph(2);
  const view = useAppStore((s) => s.view);

  return (
    <div className="flex h-full">
      <div data-coach="graph" className="relative min-w-0 flex-1 overflow-hidden bg-bg-0">
        <CanvasToolbar />
        {view === "graph" && data && <ClusterLegend />}

        {isError ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
            <AlertTriangle className="h-6 w-6 text-cluster-3" />
            <p className="max-w-sm text-sm text-text-1">
              Couldn’t reach the Akhrot backend at{" "}
              <span className="font-mono text-text-0">{API_BASE}</span>.
            </p>
            <p className="font-mono text-xs text-text-2">
              {(error as Error)?.message} — start it with{" "}
              <span className="text-text-1">gml serve --port 8000</span>
            </p>
          </div>
        ) : view === "graph" ? (
          data && <MemoryGraph data={data} />
        ) : (
          <MemoryList />
        )}
      </div>

      <Inspector />
      <Coachmarks />
    </div>
  );
}
