"use client";

import { create } from "zustand";

export type CanvasView = "graph" | "list";

interface AppState {
  /** Currently selected memory node id (drives the inspector + camera). */
  selectedId: string | null;
  /** Cluster isolated in the graph (others fade); null = show all. */
  isolatedCluster: number | null;
  /** Inspector collapsed to 0 width. */
  inspectorOpen: boolean;
  /** Main canvas view mode. */
  view: CanvasView;
  /** Live search query (debounced upstream) — matched nodes glow. */
  search: string;
  /** Ids matched by the live recall search; empty = no active search. */
  searchMatches: string[];
  /** Bumped to force the coachmark tour to replay (Settings → Replay tour). */
  tourNonce: number;
  /** Add-memory dialog visibility. */
  addOpen: boolean;
  /** MCP-install dialog visibility. */
  mcpOpen: boolean;
  /** Memory-Clipper download dialog visibility. */
  clipOpen: boolean;

  select: (id: string | null) => void;
  isolateCluster: (id: number | null) => void;
  toggleInspector: () => void;
  setView: (v: CanvasView) => void;
  setSearch: (s: string) => void;
  setSearchMatches: (ids: string[]) => void;
  replayTour: () => void;
  setAddOpen: (v: boolean) => void;
  setMcpOpen: (v: boolean) => void;
  setClipOpen: (v: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedId: null,
  isolatedCluster: null,
  inspectorOpen: true,
  view: "graph",
  search: "",
  searchMatches: [],
  tourNonce: 0,
  addOpen: false,
  mcpOpen: false,
  clipOpen: false,

  select: (id) =>
    set((s) => ({ selectedId: id, inspectorOpen: id ? true : s.inspectorOpen })),
  isolateCluster: (id) =>
    set((s) => ({ isolatedCluster: s.isolatedCluster === id ? null : id })),
  toggleInspector: () => set((s) => ({ inspectorOpen: !s.inspectorOpen })),
  setView: (view) => set({ view }),
  setSearch: (search) => set({ search }),
  setSearchMatches: (searchMatches) => set({ searchMatches }),
  replayTour: () => {
    if (typeof window !== "undefined") localStorage.removeItem("gml.onboarded");
    set((s) => ({ tourNonce: s.tourNonce + 1 }));
  },
  setAddOpen: (addOpen) => set({ addOpen }),
  setMcpOpen: (mcpOpen) => set({ mcpOpen }),
  setClipOpen: (clipOpen: boolean) => set({ clipOpen }),
}));
