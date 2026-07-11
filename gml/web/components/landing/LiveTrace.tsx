"use client";

import { useEffect, useRef, useState } from "react";
import { useInView } from "framer-motion";

// A real-looking pipeline trace. The credibility moment — show, don't tell.
const SCRIPT = [
  '$ gml trace "what port does auth_service run on?"',
  "› classifier        intent=question · conf 0.82        0ms",
  "› embedder          fastembed:bge-small · dim 384      87ms",
  "› retriever.probe   12 hits above threshold            13ms",
  "› retriever.top_k   50 candidates                      24ms",
  "› reranker          top score 0.922                    214ms",
  "› sam               skipped — unambiguous              0ms",
  "› assembler         selected 10 · budget 1024          1ms",
  "› translator        1.2k chars · 10 items              3ms",
  "✓ context assembled · auth_service runs on port 8000",
].join("\n");

export function LiveTrace() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.5 });
  const [n, setN] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let typer: ReturnType<typeof setInterval>;
    const start = () => {
      setN(0);
      typer = setInterval(() => {
        setN((c) => {
          if (c >= SCRIPT.length) {
            clearInterval(typer);
            return c;
          }
          return c + 1;
        });
      }, 14);
    };
    start();
    const loop = setInterval(start, 12000); // re-reveal every 12s
    return () => {
      clearInterval(typer);
      clearInterval(loop);
    };
  }, [inView]);

  return (
    <section ref={ref} className="border-t border-border px-8 py-24 sm:px-16">
      <div className="mx-auto max-w-3xl">
        <p className="mb-4 font-mono text-xs tracking-wider text-text-2">
          LIVE TRACE
        </p>
        <div className="overflow-hidden rounded-lg border border-border bg-bg-1">
          <div className="flex items-center gap-1.5 border-b border-border px-4 py-2.5">
            <span className="h-2.5 w-2.5 rounded-full bg-bg-3" />
            <span className="h-2.5 w-2.5 rounded-full bg-bg-3" />
            <span className="h-2.5 w-2.5 rounded-full bg-bg-3" />
            <span className="ml-2 font-mono text-[11px] text-text-2">gml — trace</span>
          </div>
          <pre className="min-h-[260px] overflow-x-auto whitespace-pre-wrap p-5 font-mono text-[12.5px] leading-relaxed text-text-1">
            {SCRIPT.slice(0, n)}
            <span className="ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 bg-accent" />
          </pre>
        </div>
      </div>
    </section>
  );
}
