"use client";

import { useCallback, useEffect, useState } from "react";
import { useAppStore } from "@/stores/app-store";
import { Button } from "@/components/ui/button";

const KEY = "gml.onboarded";

type Step = { selector: string; title: string; prep?: () => void };

export function Coachmarks() {
  const tourNonce = useAppStore((s) => s.tourNonce);
  const select = useAppStore((s) => s.select);

  const STEPS: Step[] = [
    { selector: '[data-coach="graph"]', title: "This is your memory. Each star is one thing GML remembers." },
    { selector: '[data-coach="legend"]', title: "Stars cluster by what they’re about. Click a cluster to isolate it." },
    { selector: '[data-coach="recall"]', title: "Ask anything. GML retrieves, reranks, and explains why.", prep: () => select(null) },
    { selector: '[data-coach="trace"]', title: "When something feels off, the trace shows every pipeline stage." },
  ];

  const [active, setActive] = useState(false);
  const [step, setStep] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);

  // Start on first run, or whenever the user replays the tour.
  // `?tour=skip` deep-links past onboarding (handy for demos / screenshots).
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (new URLSearchParams(window.location.search).get("tour") === "skip") {
      localStorage.setItem(KEY, "v1");
      return;
    }
    if (tourNonce > 0 || localStorage.getItem(KEY) == null) {
      setStep(0);
      setActive(true);
    }
  }, [tourNonce]);

  const measure = useCallback((selector: string) => {
    const el = document.querySelector(selector);
    setRect(el ? el.getBoundingClientRect() : null);
  }, []);

  useEffect(() => {
    if (!active) return;
    STEPS[step]?.prep?.();
    const t = setTimeout(() => measure(STEPS[step].selector), 240); // wait for transitions
    const onResize = () => measure(STEPS[step].selector);
    window.addEventListener("resize", onResize);
    return () => {
      clearTimeout(t);
      window.removeEventListener("resize", onResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, step]);

  const finish = () => {
    localStorage.setItem(KEY, "v1");
    setActive(false);
  };
  const next = () => (step >= STEPS.length - 1 ? finish() : setStep((s) => s + 1));

  if (!active) return null;
  const pad = 8;
  const tipTop = rect ? Math.min(rect.bottom + 12, window.innerHeight - 120) : 120;
  const tipLeft = rect
    ? Math.min(Math.max(rect.left, 16), window.innerWidth - 320)
    : window.innerWidth / 2 - 160;

  return (
    <div className="fixed inset-0 z-50" onClick={next}>
      {/* spotlight: a transparent window with a giant shadow dimming everything else */}
      {rect ? (
        <div
          className="pointer-events-none absolute rounded-md"
          style={{
            top: rect.top - pad,
            left: rect.left - pad,
            width: rect.width + pad * 2,
            height: rect.height + pad * 2,
            boxShadow: "0 0 0 9999px rgba(5,5,7,0.72)",
            outline: "1px solid var(--accent)",
            outlineOffset: 0,
          }}
        />
      ) : (
        <div className="absolute inset-0 bg-[rgba(5,5,7,0.72)]" />
      )}

      {/* tooltip */}
      <div
        className="absolute w-[300px] rounded-lg border border-border-strong bg-bg-2 p-4"
        style={{ top: tipTop, left: tipLeft }}
        onClick={(e) => e.stopPropagation()}
      >
        <p className="text-sm leading-relaxed text-text-0">{STEPS[step].title}</p>
        <div className="mt-4 flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((_, i) => (
              <span
                key={i}
                className="h-1.5 w-1.5 rounded-full"
                style={{ background: i === step ? "var(--accent)" : "var(--bg-3)" }}
              />
            ))}
          </div>
          <Button size="sm" onClick={next}>
            {step >= STEPS.length - 1 ? "Done" : "Next"}
          </Button>
        </div>
      </div>

      <button
        onClick={(e) => {
          e.stopPropagation();
          finish();
        }}
        className="absolute bottom-5 right-5 text-xs text-text-2 hover:text-text-0"
      >
        Skip
      </button>
    </div>
  );
}
