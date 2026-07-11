"use client";

import { ChevronRight } from "lucide-react";
import { useAppStore } from "@/stores/app-store";
import { useMemory } from "@/hooks/use-gml";
import { MemoryDetail } from "./MemoryDetail";
import { RecallPanel } from "./RecallPanel";
import { cn } from "@/lib/utils";

const WIDTH = 360;

export function Inspector() {
  const { selectedId, inspectorOpen, toggleInspector } = useAppStore();
  const { data } = useMemory(selectedId);

  return (
    <aside
      className="relative shrink-0 border-l border-border bg-bg-1 transition-[width] duration-[180ms] ease-out"
      style={{ width: inspectorOpen ? WIDTH : 0 }}
    >
      {/* Hairline grab handle — collapses the inspector to 0. */}
      <button
        onClick={toggleInspector}
        title={inspectorOpen ? "Collapse" : "Expand"}
        className="group absolute -left-3 top-1/2 z-20 flex h-12 w-3 -translate-y-1/2 items-center justify-center"
      >
        <span className="h-8 w-px bg-border-strong transition-colors group-hover:bg-accent" />
        <ChevronRight
          className={cn(
            "absolute h-3 w-3 text-text-2 transition-transform duration-[180ms] ease-out group-hover:text-text-0",
            inspectorOpen ? "" : "rotate-180",
          )}
        />
      </button>

      {inspectorOpen && (
        <div className="h-full overflow-y-auto" style={{ width: WIDTH }}>
          {selectedId && data ? <MemoryDetail detail={data} /> : <RecallPanel />}
        </div>
      )}
    </aside>
  );
}
