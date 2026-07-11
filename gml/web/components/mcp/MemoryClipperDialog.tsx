"use client";

import { AnimatePresence, motion } from "framer-motion";
import { X, Download, Globe, CheckCircle2 } from "lucide-react";
import { useAppStore } from "@/stores/app-store";

const ZIP_URL = "https://akhrots.com/static/akhrots-clip.zip";

/**
 * Memory Clipper — browser extension download + install guide.
 * Clips ChatGPT / Claude / Gemini conversations straight into Akhrot memory.
 */
export function MemoryClipperDialog() {
  const open = useAppStore((s) => s.clipOpen);
  const setOpen = useAppStore((s) => s.setClipOpen);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[55] flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onClick={() => setOpen(false)}
        >
          <div className="absolute inset-0 bg-[rgba(5,5,7,0.7)] backdrop-blur-sm" />

          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="clip-dialog-title"
            initial={{ opacity: 0, scale: 0.97, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 8 }}
            transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
            onClick={(e) => e.stopPropagation()}
            className="relative z-10 w-full max-w-md rounded-xl border border-border-strong bg-bg-1 p-6"
          >
            {/* Header */}
            <div className="mb-5 flex items-start justify-between">
              <div>
                <h2 id="clip-dialog-title" className="font-clash text-lg font-semibold tracking-tight text-text-0">
                  Memory Clipper
                </h2>
                <p className="mt-1 text-sm text-text-1">
                  Save any AI conversation to Akhrot in one click.
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="flex h-7 w-7 items-center justify-center rounded-sm text-text-2 hover:bg-bg-3 hover:text-text-0"
                aria-label="Close"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* What it does */}
            <div className="mb-4 rounded-md border border-border bg-bg-2 p-4">
              <div className="flex items-center gap-2 mb-2">
                <Globe className="h-4 w-4 text-accent" />
                <span className="text-sm font-medium text-text-0">Chrome Extension</span>
                <span className="ml-auto rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-semibold text-accent">
                  v0.3.1
                </span>
              </div>
              <p className="text-xs leading-relaxed text-text-1">
                Works on <strong className="text-text-0">ChatGPT</strong>,{" "}
                <strong className="text-text-0">Claude</strong>, and{" "}
                <strong className="text-text-0">Gemini</strong>. Detects conversations
                automatically — hit <em>Save</em> and every exchange is ingested into your
                Akhrot memory layer. Recall anything from the{" "}
                <strong className="text-text-0">Recall</strong> tab instantly.
              </p>
            </div>

            {/* Feature pills */}
            <div className="mb-4 flex flex-wrap gap-2">
              {[
                "Auto-detects AI chats",
                "One-click save",
                "Instant recall",
                "No copy-paste",
              ].map((f) => (
                <span
                  key={f}
                  className="flex items-center gap-1.5 rounded-full border border-border bg-bg-2 px-2.5 py-1 text-[11px] text-text-1"
                >
                  <CheckCircle2 className="h-3 w-3 text-accent" />
                  {f}
                </span>
              ))}
            </div>

            {/* Download button */}
            <a
              href={ZIP_URL}
              download
              className="mb-4 flex w-full items-center justify-between gap-2 rounded-md bg-accent px-4 py-2.5 text-sm font-medium text-bg-0 transition-opacity hover:opacity-90"
            >
              <span className="flex items-center gap-2">
                <Download className="h-4 w-4" />
                Download Memory Clipper
              </span>
              <span className="text-xs opacity-70">.zip · 34 KB</span>
            </a>

            {/* Install steps */}
            <div className="rounded-md border border-border-strong bg-bg-2 p-4">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-text-2">
                Install in 4 steps
              </p>
              <ol className="ml-4 list-decimal space-y-2 text-xs text-text-1">
                <li>
                  Unzip the downloaded file to any folder.
                </li>
                <li>
                  Open Chrome and go to{" "}
                  <span className="font-mono text-text-0">chrome://extensions</span> —
                  enable <strong className="text-text-0">Developer mode</strong> (top-right toggle).
                </li>
                <li>
                  Click <strong className="text-text-0">Load unpacked</strong> and select the
                  unzipped folder.
                </li>
                <li>
                  Open the extension popup → <strong className="text-text-0">Settings</strong> →
                  paste your Akhrot API token → click{" "}
                  <strong className="text-text-0">Test connection</strong>.
                </li>
              </ol>
              <p className="mt-3 text-[11px] leading-relaxed text-text-2">
                Your API token is under{" "}
                <strong className="text-text-1">API keys</strong> in the sidebar. Once
                connected, the header dot turns green and you&rsquo;re ready to clip.
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
