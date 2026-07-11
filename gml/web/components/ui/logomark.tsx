/** GML logomark — three linked nodes, the smallest possible memory graph. */
export function Logomark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-label="GML">
      <path d="M6 7 L18 17 M6 7 L17 6.5 M18 17 L17 6.5" stroke="var(--border-strong)" strokeWidth="1.2" />
      <circle cx="6" cy="7" r="2.4" fill="var(--accent)" />
      <circle cx="18" cy="17" r="2.1" fill="var(--cluster-2)" />
      <circle cx="17" cy="6.5" r="1.8" fill="var(--cluster-1)" />
    </svg>
  );
}
