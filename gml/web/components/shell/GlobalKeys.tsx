"use client";

import { useEffect } from "react";
import { useAppStore } from "@/stores/app-store";

/**
 * App-wide keyboard shortcuts:
 *   Esc — close the add dialog, else deselect the current memory
 *   N   — add a memory
 *   G/L — switch graph / list view
 */
export function GlobalKeys() {
  const { addOpen, setAddOpen, selectedId, select, setView } = useAppStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const typing =
        el?.tagName === "INPUT" || el?.tagName === "TEXTAREA" || el?.isContentEditable;

      if (e.key === "Escape") {
        if (addOpen) setAddOpen(false);
        else if (selectedId) select(null);
        return;
      }
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === "n" || e.key === "N") {
        e.preventDefault();
        setAddOpen(true);
      } else if (e.key === "g" || e.key === "G") {
        setView("graph");
      } else if (e.key === "l" || e.key === "L") {
        setView("list");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [addOpen, setAddOpen, selectedId, select, setView]);

  return null;
}
