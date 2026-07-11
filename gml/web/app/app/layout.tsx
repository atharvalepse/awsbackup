import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Sidebar } from "@/components/shell/Sidebar";
import { GlobalKeys } from "@/components/shell/GlobalKeys";
import { AddMemoryDialog } from "@/components/memory/AddMemoryDialog";
import { McpInstallDialog } from "@/components/mcp/McpInstallDialog";
import { MemoryClipperDialog } from "@/components/mcp/MemoryClipperDialog";
import { Toaster } from "@/components/ui/toaster";
import { AuthGate } from "@/components/auth/AuthGate";

export const metadata: Metadata = {
  title: "Akhrot — Memory",
};

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <AuthGate>
      <div className="akhrot flex h-screen w-screen overflow-hidden bg-bg-0 text-text-0">
        <Sidebar />
        <main className="min-w-0 flex-1">{children}</main>
        <AddMemoryDialog />
        <McpInstallDialog />
        <MemoryClipperDialog />
        <Toaster />
        <GlobalKeys />
      </div>
    </AuthGate>
  );
}
