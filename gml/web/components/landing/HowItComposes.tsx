"use client";

import { useRef } from "react";
import { motion, useInView, useReducedMotion } from "framer-motion";

/** Three pipeline stages, each with a looping SVG figure that pauses off-screen. */
export function HowItComposes() {
  return (
    <section className="border-t border-border px-8 py-24 sm:px-16">
      <div className="mx-auto grid max-w-5xl grid-cols-1 gap-12 md:grid-cols-3">
        <Stage
          label="01 / Ingest"
          title="Fragments become facts"
          body="Text collapses into one atomic entity·attribute·value node — scored for importance and confidence."
          figure={<IngestFigure />}
        />
        <Stage
          label="02 / Recall"
          title="A question lights the graph"
          body="A query finds the right node, then traces outward along its strongest edges to bring back what's related."
          figure={<RecallFigure />}
        />
        <Stage
          label="03 / Assemble"
          title="Context, composed"
          body="Neighbours are reranked, conflicts resolved, and the survivors flow into the context the model receives."
          figure={<AssembleFigure />}
        />
      </div>
    </section>
  );
}

function Stage({
  label,
  title,
  body,
  figure,
}: {
  label: string;
  title: string;
  body: string;
  figure: React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.4 });
  return (
    <div ref={ref}>
      <div className="mb-5 h-28 [&_*]:will-change-transform">
        {inView ? figure : <div className="h-full" />}
      </div>
      <p className="font-mono text-xs tracking-wider text-text-2">{label}</p>
      <h2 className="mt-2 text-[28px] font-medium leading-tight text-text-0">{title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-text-1">{body}</p>
    </div>
  );
}

const loop = (duration: number) => ({
  duration,
  repeat: Infinity,
  ease: "easeInOut" as const,
});

function IngestFigure() {
  const reduce = useReducedMotion();
  const frags = [
    { x: 2, y: 8 },
    { x: 6, y: 60 },
    { x: 2, y: 100 },
  ];
  return (
    <svg viewBox="0 0 160 112" className="h-full w-full" aria-hidden>
      {frags.map((f, i) => (
        <motion.rect
          key={i}
          x={f.x}
          y={f.y}
          width="46"
          height="10"
          rx="2"
          fill="var(--bg-3)"
          animate={reduce ? {} : { x: [f.x, 96], y: [f.y, 51], opacity: [0.9, 0] }}
          transition={{ ...loop(3), delay: i * 0.35 }}
        />
      ))}
      <circle cx="120" cy="56" r="9" fill="var(--accent)" />
      <text x="120" y="80" fill="var(--text-2)" fontSize="8" textAnchor="middle" fontFamily="monospace">
        node
      </text>
    </svg>
  );
}

function RecallFigure() {
  const reduce = useReducedMotion();
  const targets = [
    [120, 18],
    [128, 56],
    [118, 94],
  ];
  return (
    <svg viewBox="0 0 160 112" className="h-full w-full" aria-hidden>
      {targets.map(([x, y], i) => (
        <g key={i}>
          <motion.line
            x1="40"
            y1="56"
            x2={x}
            y2={y}
            stroke="var(--accent)"
            strokeWidth="1"
            animate={reduce ? {} : { pathLength: [0, 1], opacity: [0.2, 0.8] }}
            transition={{ ...loop(3), delay: i * 0.3 }}
          />
          <circle cx={x} cy={y} r="5" fill="var(--cluster-2)" />
        </g>
      ))}
      <motion.circle
        cx="40"
        cy="56"
        r="9"
        fill="var(--accent)"
        animate={reduce ? {} : { scale: [1, 1.15, 1] }}
        transition={loop(2)}
        style={{ transformOrigin: "40px 56px" }}
      />
    </svg>
  );
}

function AssembleFigure() {
  const reduce = useReducedMotion();
  const dots = [30, 65, 100, 135];
  return (
    <svg viewBox="0 0 160 112" className="h-full w-full" aria-hidden>
      {dots.map((x, i) => (
        <motion.circle
          key={i}
          cx={x}
          cy="20"
          r="5"
          fill={`var(--cluster-${(i % 6) + 1})`}
          animate={reduce ? {} : { cy: [20, 86], opacity: [1, 0.3] }}
          transition={{ ...loop(3), delay: i * 0.25 }}
        />
      ))}
      <rect x="20" y="92" width="120" height="14" rx="3" fill="var(--bg-2)" stroke="var(--border-strong)" />
      <rect x="20" y="92" width="120" height="14" rx="3" fill="none" stroke="var(--accent)" strokeOpacity="0.3" />
    </svg>
  );
}
