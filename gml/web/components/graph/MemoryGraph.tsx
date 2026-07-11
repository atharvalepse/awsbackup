"use client";

/**
 * The memory graph — a themed wrapper around react-force-graph-3d (never a
 * hand-rolled three.js graph). Loaded via next/dynamic({ssr:false}) by the
 * page, so it only ever runs client-side.
 *
 * IMPORTANT: we do NOT import `three` here. react-force-graph-3d bundles its
 * own three.js instance; importing a second one (e.g. for UnrealBloomPass)
 * crashes its render loop ("Cannot read properties of undefined (reading
 * 'tick')") and loses the WebGL context. So no custom postprocessing/fog —
 * the star-cluster look comes from cluster-colored nodes on the dark canvas.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph3D, {
  type ForceGraphMethods,
  type NodeObject,
  type LinkObject,
} from "react-force-graph-3d";
import type { GraphResponse, GraphNode } from "@/lib/types";
import { ACCENT_WALNUT, BG0_WALNUT, clusterHex } from "@/lib/cluster-colors";
import { useAppStore } from "@/stores/app-store";

const DIM = "#cdbfa6"; // light warm faded node/link when out of focus

// react-force-graph renders `nodeLabel` as raw HTML. Memory entity/value are
// user-ingested content, so they MUST be escaped to avoid stored XSS (e.g. a
// memory value of `<img src=x onerror=...>` would otherwise run script — and
// the JWT lives in localStorage). Escape every interpolated field.
function esc(s: string | null | undefined): string {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

type LinkExtra = { weight: number };
type NodeT = NodeObject<GraphNode>;
type LinkT = LinkObject<GraphNode, LinkExtra>;
type FG = ForceGraphMethods<NodeT, LinkT>;

export default function MemoryGraph({ data }: { data: GraphResponse }) {
  const fgRef = useRef<FG | undefined>(undefined);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  const { selectedId, isolatedCluster, searchMatches, select } = useAppStore();

  // Clone nodes/links so the force engine never mutates the query cache.
  const graphData = useMemo(
    () => ({
      nodes: data.nodes.map((n) => ({ ...n })),
      links: data.edges.map((e) => ({ ...e })),
    }),
    [data],
  );

  // Adjacency for 1-hop neighbour highlighting.
  const adjacency = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of data.edges) {
      (m.get(e.source) ?? m.set(e.source, new Set()).get(e.source)!).add(e.target);
      (m.get(e.target) ?? m.set(e.target, new Set()).get(e.target)!).add(e.source);
    }
    return m;
  }, [data.edges]);

  const neighbors = useMemo(() => {
    if (!selectedId) return null;
    const s = new Set<string>([selectedId]);
    adjacency.get(selectedId)?.forEach((id) => s.add(id));
    return s;
  }, [selectedId, adjacency]);

  const matchSet = useMemo(() => new Set(searchMatches), [searchMatches]);

  // ResizeObserver → fill the canvas pane.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // Seed synchronously so the graph mounts on first paint (ResizeObserver
    // can fire a frame late, and doesn't fire at all in some headless runs).
    const r = el.getBoundingClientRect();
    if (r.width > 0) setSize({ w: Math.round(r.width), h: Math.round(r.height) });
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Cap DPR for perf once the canvas has a size. We deliberately do NOT call
  // d3ReheatSimulation() / tweak forces here: doing so flips engineRunning=true
  // before the force-graph has digested graphData and assigned its layout, so
  // the first tick reads `state.layout.tick` of undefined and crashes the
  // render loop (black canvas). The default d3 forces lay the graph out fine.
  const tunedRef = useRef(false);
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || tunedRef.current || size.w === 0) return;
    tunedRef.current = true;
    try {
      fg.renderer().setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    } catch {
      /* renderer not ready — harmless */
    }
  }, [size.w]);

  const nodeColor = useCallback(
    (node: GraphNode) => {
      if (node.id === selectedId) return ACCENT_WALNUT;
      if (matchSet.size > 0) return matchSet.has(node.id) ? clusterHex(node.cluster_id) : DIM;
      if (neighbors) return neighbors.has(node.id) ? clusterHex(node.cluster_id) : DIM;
      if (isolatedCluster != null)
        return node.cluster_id === isolatedCluster ? clusterHex(node.cluster_id) : DIM;
      return clusterHex(node.cluster_id);
    },
    [selectedId, matchSet, neighbors, isolatedCluster],
  );

  const linkColor = useCallback(
    (link: LinkT) => {
      const s = typeof link.source === "object" ? link.source?.id : link.source;
      const t = typeof link.target === "object" ? link.target?.id : link.target;
      if (selectedId && (s === selectedId || t === selectedId)) return ACCENT_WALNUT;
      return "#9c8a6a"; // warm walnut hairline edge (visible on beige)
    },
    [selectedId],
  );

  // Orient the camera to a named side, or fit the whole graph in view.
  const orientCamera = useCallback((view: "top" | "front" | "side" | "fit") => {
    const fg = fgRef.current;
    if (!fg) return;
    if (view === "fit") {
      fg.zoomToFit(700, 60);
      return;
    }
    const cam = fg.camera();
    const d = Math.hypot(cam.position.x, cam.position.y, cam.position.z) || 300;
    const pos =
      view === "top"
        ? { x: 0, y: d, z: 0.001 }
        : view === "side"
          ? { x: d, y: 0, z: 0.001 }
          : { x: 0, y: 0, z: d };
    fg.cameraPosition(pos, { x: 0, y: 0, z: 0 }, 800);
  }, []);

  const flyTo = useCallback((node: GraphNode) => {
    const fg = fgRef.current;
    if (!fg) return;
    const { x = 0, y = 0, z = 0 } = node as GraphNode & { z?: number };
    const dist = 110;
    const hyp = Math.hypot(x, y, z) || 1;
    const r = 1 + dist / hyp;
    fg.cameraPosition({ x: x * r, y: y * r, z: z * r }, { x, y, z }, 1200);
  }, []);

  // Selecting from the inspector / list / recall also flies the camera.
  const lastFlown = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedId || selectedId === lastFlown.current) return;
    const node = graphData.nodes.find((n) => n.id === selectedId);
    if (node) {
      lastFlown.current = selectedId;
      flyTo(node);
    }
  }, [selectedId, graphData, flyTo]);

  return (
    <div ref={wrapRef} className="absolute inset-0">
      {size.w > 0 && (
        <ForceGraph3D<GraphNode, LinkExtra>
          ref={fgRef}
          width={size.w}
          height={size.h}
          graphData={graphData}
          backgroundColor={BG0_WALNUT}
          showNavInfo={false}
          // Lay the graph out synchronously (warmupTicks) and DON'T run the
          // async tick loop (cooldownTicks=0). This sidesteps a 3d-force-graph
          // init race where the loop ticks before graphData is digested and
          // `state.layout` exists → crash. With cooldownTicks=0, layoutTick
          // hits its stop-check before ever calling state.layout.tick().
          warmupTicks={100}
          cooldownTicks={0}
          // Orbit controls: drag = rotate around the cloud, scroll = zoom,
          // right-drag = pan. Most intuitive for viewing from any side.
          controlType="orbit"
          enableNodeDrag={false}
          nodeRelSize={4}
          nodeVal={(n: GraphNode) => (n.id === selectedId ? n.val * 1.6 : n.val)}
          nodeColor={nodeColor}
          nodeOpacity={0.95}
          nodeResolution={14}
          nodeLabel={(n: GraphNode) =>
            `<div style="background:#fcf9f2;border:1px solid #c2b095;border-radius:8px;padding:6px 9px;font:12px/1.3 Satoshi,ui-sans-serif;color:#2e2418;max-width:240px"><b>${esc(n.entity ?? n.label)}</b>${n.value ? `<div style='color:#5c4e3a;margin-top:2px'>${esc(n.value.slice(0, 80))}</div>` : ""}</div>`
          }
          linkColor={linkColor}
          linkOpacity={0.55}
          linkWidth={(l: { weight: number }) => 0.6 + l.weight * 1.6}
          // Particles flowing along each edge make the connections legible.
          linkDirectionalParticles={2}
          linkDirectionalParticleWidth={1.8}
          linkDirectionalParticleSpeed={0.006}
          linkDirectionalParticleColor={() => "#9c8a76"}
          onNodeClick={(n: GraphNode) => {
            select(n.id);
            flyTo(n);
          }}
          onBackgroundClick={() => select(null)}
        />
      )}

      {/* View controls — orbit from any side / fit to screen */}
      <div className="absolute bottom-3 left-3 z-10 flex items-center gap-1 rounded-md border border-border bg-bg-2/90 p-1 backdrop-blur">
        {(["fit", "top", "front", "side"] as const).map((v) => (
          <button
            key={v}
            onClick={() => orientCamera(v)}
            className="rounded-[5px] px-2.5 py-1 text-xs capitalize text-text-2 transition-colors duration-[180ms] ease-out hover:bg-bg-3 hover:text-text-0"
          >
            {v}
          </button>
        ))}
      </div>
      <div className="pointer-events-none absolute bottom-3 right-3 z-10 font-mono text-[10px] text-text-2">
        drag · rotate &nbsp; scroll · zoom &nbsp; right-drag · pan
      </div>
    </div>
  );
}
