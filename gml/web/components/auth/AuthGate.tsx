"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { isAuthed } from "@/lib/auth";

/**
 * Client-side route guard for the authenticated app shell. Redirects to
 * /login when no token is stored. (Auth is enforced server-side by the API;
 * this is just UX routing.) Sign-out lives in the sidebar Settings menu.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!isAuthed()) {
      router.replace("/login");
    } else {
      setReady(true);
    }
  }, [router]);

  if (!ready) {
    return (
      <div className="akhrot flex h-screen w-screen items-center justify-center bg-bg-0 text-sm text-text-2">
        Loading…
      </div>
    );
  }

  return <>{children}</>;
}
