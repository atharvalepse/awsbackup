"use client";

import { useEffect, useRef, useState } from "react";
import { Search, Plus } from "lucide-react";
import { useAppStore, type CanvasView } from "@/stores/app-store";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const VIEWS: CanvasView[] = ["graph", "list"];

/** Top-left of the canvas: Graph|List segmented control + debounced live recall. */
export function CanvasToolbar() {
  const { view, setView, search, setSearch, setSearchMatches, setAddOpen } = useAppStore();
  const [local, setLocal] = useState(search);
  const timer = useRef<ReturnType<typeof setTimeout>>();

  // Debounced (250ms) live recall — matched node ids glow, others dim.
  useEffect(() => {
    clearTimeout(timer.current);
    // Guard against out-of-order resolution: a slow earlier recall must not
    // overwrite a newer query's matches (and a resolve after unmount must be
    // ignored). The cleanup invalidates this run before the next one starts.
    let cancelled = false;
    timer.current = setTimeout(async () => {
      setSearch(local);
      const q = local.trim();
      if (!q) {
        if (!cancelled) setSearchMatches([]);
        return;
      }
      try {
        const r = await api.recall(q, 12);
        if (!cancelled) setSearchMatches(r.results.map((x) => x.memory.id));
      } catch {
        if (!cancelled) setSearchMatches([]);
      }
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer.current);
    };
  }, [local, setSearch, setSearchMatches]);

  return (
    <div className="absolute left-3 top-3 z-10 flex items-center gap-2">
      <div className="flex rounded-md border border-border bg-bg-2/90 p-0.5 backdrop-blur">
        {VIEWS.map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            className={cn(
              "rounded-[5px] px-3 py-1 text-xs capitalize transition-colors duration-[180ms] ease-out",
              view === v ? "bg-bg-3 text-text-0" : "text-text-2 hover:text-text-1",
            )}
          >
            {v}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2 rounded-md border border-border bg-bg-2/90 px-2.5 backdrop-blur">
        <Search className="h-3.5 w-3.5 text-text-2" />
        <input
          value={local}
          onChange={(e) => setLocal(e.target.value)}
          placeholder="Search memories…"
          className="w-44 bg-transparent py-1.5 text-xs text-text-0 placeholder:text-text-2 focus:outline-none"
        />
      </div>
      <button
        onClick={() => setAddOpen(true)}
        data-coach="add"
        title="Add a memory (saves a conversation turn)"
        className="flex items-center gap-1.5 rounded-md border border-border bg-bg-2/90 px-2.5 py-1.5 text-xs text-text-1 backdrop-blur transition-colors duration-[180ms] hover:bg-bg-3 hover:text-text-0"
      >
        <Plus className="h-3.5 w-3.5" /> Add memory
      </button>
    </div>
  );
}
