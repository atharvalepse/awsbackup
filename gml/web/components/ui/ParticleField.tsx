"use client";

import { useEffect, useRef } from "react";

/**
 * Interactive particle-network background (ReactBits "Particles" aesthetic),
 * canvas-based so it adds zero deps and can't break the build. Drifting nodes
 * link with hairlines; the cursor pulls signal-green threads toward nearby
 * nodes — on-brand for a memory *graph*. Respects prefers-reduced-motion.
 */
export function ParticleField() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const ACCENT = "184,255,61";
    let w = 0, h = 0, raf = 0;
    type P = { x: number; y: number; vx: number; vy: number; r: number; accent: boolean };
    let ps: P[] = [];
    const mouse = { x: -9999, y: -9999 };

    function resize() {
      w = canvas!.clientWidth;
      h = canvas!.clientHeight;
      canvas!.width = w * dpr;
      canvas!.height = h * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.min(95, Math.max(36, Math.floor((w * h) / 15000)));
      ps = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.22,
        vy: (Math.random() - 0.5) * 0.22,
        r: Math.random() * 1.5 + 0.6,
        accent: Math.random() < 0.12,
      }));
    }

    function frame() {
      ctx!.clearRect(0, 0, w, h);
      for (let i = 0; i < ps.length; i++) {
        const a = ps[i];
        for (let j = i + 1; j < ps.length; j++) {
          const b = ps[j];
          const dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
          if (d2 < 19600) {
            const o = (1 - Math.sqrt(d2) / 140) * 0.15;
            ctx!.strokeStyle = `rgba(255,255,255,${o})`;
            ctx!.lineWidth = 0.6;
            ctx!.beginPath();
            ctx!.moveTo(a.x, a.y);
            ctx!.lineTo(b.x, b.y);
            ctx!.stroke();
          }
        }
        const mdx = a.x - mouse.x, mdy = a.y - mouse.y, md2 = mdx * mdx + mdy * mdy;
        if (md2 < 25600) {
          const o = (1 - Math.sqrt(md2) / 160) * 0.45;
          ctx!.strokeStyle = `rgba(${ACCENT},${o})`;
          ctx!.lineWidth = 0.7;
          ctx!.beginPath();
          ctx!.moveTo(a.x, a.y);
          ctx!.lineTo(mouse.x, mouse.y);
          ctx!.stroke();
        }
      }
      for (const p of ps) {
        if (!reduce) {
          p.x += p.vx;
          p.y += p.vy;
          if (p.x < 0) p.x += w;
          if (p.x > w) p.x -= w;
          if (p.y < 0) p.y += h;
          if (p.y > h) p.y -= h;
        }
        ctx!.beginPath();
        ctx!.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        if (p.accent) {
          ctx!.fillStyle = `rgba(${ACCENT},0.9)`;
          ctx!.shadowColor = `rgba(${ACCENT},0.8)`;
          ctx!.shadowBlur = 8;
        } else {
          ctx!.fillStyle = "rgba(216,216,228,0.5)";
          ctx!.shadowBlur = 0;
        }
        ctx!.fill();
      }
      ctx!.shadowBlur = 0;
      raf = requestAnimationFrame(frame);
    }

    const onMove = (e: MouseEvent) => {
      const r = canvas!.getBoundingClientRect();
      mouse.x = e.clientX - r.left;
      mouse.y = e.clientY - r.top;
    };
    const onLeave = () => {
      mouse.x = -9999;
      mouse.y = -9999;
    };

    resize();
    frame();
    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseout", onLeave);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseout", onLeave);
    };
  }, []);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" aria-hidden />;
}
