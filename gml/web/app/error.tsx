"use client";

/**
 * Root error boundary. Without this, an uncaught throw in any client
 * component — notably the WebGL/three.js render loop in MemoryGraph, which
 * the graph code explicitly guards against — white-screens the whole app.
 * Here it degrades to a recoverable panel with a reset.
 */
import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Surface to the console (and any attached error reporter) for debugging.
    console.error("App error boundary caught:", error);
  }, [error]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-bg-0 px-6 text-center">
      <h2 className="text-lg font-medium text-text-0">Something went wrong</h2>
      <p className="max-w-md text-sm text-text-2">
        The view hit an unexpected error and was stopped to keep the rest of the
        app responsive. You can try again.
      </p>
      <button
        onClick={() => reset()}
        className="rounded-md border border-border bg-bg-2 px-4 py-2 text-sm text-text-1 transition-colors hover:bg-bg-3 hover:text-text-0"
      >
        Try again
      </button>
    </div>
  );
}
