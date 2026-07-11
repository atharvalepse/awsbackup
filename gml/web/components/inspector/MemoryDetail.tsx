"use client";

import { useState } from "react";
import { Trash2, Sparkles } from "lucide-react";
import type { MemoryDetailResponse } from "@/lib/types";
import { useForget } from "@/hooks/use-gml";
import { toast } from "@/stores/toast-store";
import { useAppStore } from "@/stores/app-store";
import { Meter } from "@/components/ui/meter";
import { Button } from "@/components/ui/button";
import { clusterHex } from "@/lib/cluster-colors";
import { relativeTime } from "@/lib/utils";
import { RecallDrawer } from "./RecallDrawer";

export function MemoryDetail({ detail }: { detail: MemoryDetailResponse }) {
  const m = detail.memory;
  const { select } = useAppStore();
  const forget = useForget();
  const [confirm, setConfirm] = useState(false);
  const [drawerQuery, setDrawerQuery] = useState<string | null>(null);

  const onForget = async () => {
    try {
      await forget.mutateAsync(m.id); // invalidates graph/clusters/memories
      toast.success("Memory forgotten", m.entity ?? m.content.slice(0, 48));
      select(null);
    } catch (e) {
      toast.error("Couldn’t forget", (e as Error).message);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="space-y-5 p-5">
        {/* header */}
        <div>
          <div className="flex items-center gap-2">
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: clusterHex(m.cluster_id) }}
            />
            <span className="font-mono text-[10px] uppercase tracking-wider text-text-2">
              {m.source}
            </span>
          </div>
          <h1 className="font-clash mt-2 text-xl font-semibold leading-tight tracking-tight text-text-0">
            {m.entity ?? m.content.slice(0, 60)}
          </h1>
          {(m.attribute || m.value) && (
            <p className="mt-1 font-mono text-sm text-text-1">
              {m.attribute && <span className="text-text-2">{m.attribute} → </span>}
              {m.value ?? "—"}
            </p>
          )}
        </div>

        {!m.entity && (
          <p className="text-sm leading-relaxed text-text-1">{m.content}</p>
        )}

        {/* metrics */}
        <div className="space-y-3">
          <Meter label="Confidence" value={m.confidence} />
          <Meter label="Importance" value={m.importance} />
        </div>

        {/* metadata rows */}
        <dl className="divide-y divide-border border-y border-border text-sm">
          <Row label="Created">
            <span className="tnum text-text-1" title={new Date(m.timestamp).toISOString()}>
              {relativeTime(m.timestamp)}
            </span>
          </Row>
          <Row label="Cluster">
            <span className="text-text-1">#{m.cluster_id ?? "—"}</span>
          </Row>
          <Row label="Pinned">
            <span className="text-text-1">{m.pinned ? "yes" : "no"}</span>
          </Row>
        </dl>

        {/* relationships */}
        <div>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-wider text-text-2">
            Relationships ({detail.relationships.length})
          </h2>
          <ul className="space-y-0.5">
            {detail.relationships.slice(0, 12).map((r) => (
              <li key={r.memory_id}>
                <button
                  onClick={() => select(r.memory_id)}
                  className="flex w-full items-center gap-2 rounded-sm px-1.5 py-1 text-left text-xs text-text-1 hover:bg-bg-3 hover:text-text-0"
                >
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ background: clusterHex(r.cluster_id) }}
                  />
                  <span className="truncate">{r.entity ?? r.value ?? r.memory_id}</span>
                  <span className="ml-auto font-mono text-[10px] text-text-2">{r.kind}</span>
                </button>
              </li>
            ))}
            {detail.relationships.length === 0 && (
              <li className="px-1.5 text-xs text-text-2">No linked memories.</li>
            )}
          </ul>
        </div>
      </div>

      {/* actions pinned to bottom */}
      <div className="mt-auto flex gap-2 border-t border-border p-4">
        <Button
          variant="ghost"
          size="sm"
          className="flex-1"
          onClick={() => setDrawerQuery(m.value ?? m.content)}
        >
          <Sparkles className="h-3.5 w-3.5" /> Recall similar
        </Button>
        {confirm ? (
          <Button variant="danger" size="sm" onClick={onForget} disabled={forget.isPending}>
            {forget.isPending ? "Forgetting…" : "Confirm forget"}
          </Button>
        ) : (
          <Button variant="danger" size="sm" onClick={() => setConfirm(true)}>
            <Trash2 className="h-3.5 w-3.5" /> Forget
          </Button>
        )}
      </div>

      {drawerQuery && (
        <RecallDrawer query={drawerQuery} onClose={() => setDrawerQuery(null)} />
      )}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-2">
      <dt className="text-text-2">{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}
