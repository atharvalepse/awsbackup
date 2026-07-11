"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Check, AlertTriangle, Info, X } from "lucide-react";
import { useToasts, type ToastKind } from "@/stores/toast-store";

const ICON = { success: Check, error: AlertTriangle, info: Info };
const ACCENT: Record<ToastKind, string> = {
  success: "var(--accent)",
  error: "var(--cluster-3)",
  info: "var(--cluster-1)",
};

export function Toaster() {
  const { toasts, dismiss } = useToasts();
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-80 flex-col gap-2">
      <AnimatePresence initial={false}>
        {toasts.map((t) => {
          const Icon = ICON[t.kind];
          return (
            <motion.div
              key={t.id}
              layout
              initial={{ opacity: 0, x: 24, scale: 0.98 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 24, scale: 0.98 }}
              transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
              onClick={() => dismiss(t.id)}
              className="pointer-events-auto flex cursor-pointer items-start gap-3 rounded-md border border-border bg-bg-2 p-3 shadow-glow"
              style={{ boxShadow: "0 8px 24px -12px rgba(0,0,0,0.6)" }}
            >
              <span
                className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full"
                style={{ background: `color-mix(in srgb, ${ACCENT[t.kind]} 18%, transparent)` }}
              >
                <Icon className="h-3 w-3" style={{ color: ACCENT[t.kind] }} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-text-0">{t.title}</p>
                {t.desc && <p className="mt-0.5 text-xs leading-relaxed text-text-1">{t.desc}</p>}
              </div>
              <X className="mt-0.5 h-3.5 w-3.5 shrink-0 text-text-2" />
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
