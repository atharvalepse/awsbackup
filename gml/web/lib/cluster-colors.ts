/**
 * Hex mirrors of cluster colors, for contexts that can't read CSS vars
 * (three.js / canvas materials).
 *
 * CLUSTER_HEX = the original landing-hero palette (kept so the landing,
 * which the user wants left as-is, doesn't change).
 *
 * WALNUT_HEX = the Akhrot walnut theme used across the /app product. Keep in
 * sync with the .akhrot --cluster-N tokens in styles/tokens.css.
 */
export const CLUSTER_HEX = [
  "#7dd3fc", // sky
  "#c4b5fd", // violet
  "#fca5a5", // coral
  "#fcd34d", // amber
  "#6ee7b7", // mint
  "#f0abfc", // fuchsia
] as const;

export const WALNUT_HEX = [
  "#4f8f5e", // forest green
  "#b5793e", // caramel walnut
  "#c0552f", // terracotta
  "#c99a2e", // ochre
  "#6e8b3d", // olive moss
  "#9c6b4f", // clay
] as const;

// Landing hero (three.js) — original signal green + page bg.
export const ACCENT_HEX = "#b8ff3d";
export const BG0_HEX = "#0a0a0b";

// Akhrot /app graph — forest green + warm beige ground.
export const ACCENT_WALNUT = "#3f8f5b";
export const BG0_WALNUT = "#efe7d8";

/** Cluster color for /app components (legend, list, inspector, graph nodes). */
export function clusterHex(clusterId: number | null | undefined): string {
  const n = (((clusterId ?? 0) % WALNUT_HEX.length) + WALNUT_HEX.length) % WALNUT_HEX.length;
  return WALNUT_HEX[n];
}
