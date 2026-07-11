"use client";

import { useCallback, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { RecallStreamDone } from "@/lib/types";
import { canonicalStageIndex } from "@/components/inspector/PipelineDots";

type Status = "idle" | "running" | "done" | "error";

/**
 * Drives the full-pipeline recall flow from the SSE stream: `lit` advances as
 * real `stage` events arrive (feeding PipelineDots), and `data` holds the
 * reranked results once the pipeline finishes. Aborts any in-flight request
 * when a new one starts.
 */
export function useRecallStream() {
  const [status, setStatus] = useState<Status>("idle");
  const [lit, setLit] = useState(0);
  const [data, setData] = useState<RecallStreamDone | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async (query: string, topK = 8) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setStatus("running");
    setLit(0);
    setData(null);
    setError(null);
    try {
      const done = await api.recallStream(
        query,
        topK,
        (s) => {
          const idx = canonicalStageIndex(s.stage);
          if (idx >= 0) setLit((l) => Math.max(l, idx + 1));
        },
        ac.signal,
      );
      setLit(7);
      setData(done);
      setStatus("done");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError((e as Error).message);
      setStatus("error");
    }
  }, []);

  return { status, lit, data, error, run };
}
