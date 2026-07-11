import { cn } from "@/lib/utils";

/** A 0–1 horizontal bar filled with accent. Used for confidence / importance. */
export function Meter({
  value,
  label,
  className,
}: {
  value: number;
  label: string;
  className?: string;
}) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className={cn("space-y-1.5", className)}>
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-text-2">{label}</span>
        <span className="tnum font-mono text-xs text-text-1">{value.toFixed(2)}</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-bg-3">
        <div
          className="h-full rounded-full bg-accent transition-[width] duration-[180ms] ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
