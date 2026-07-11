"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

/** Query keys — centralised so invalidation stays consistent. */
export const qk = {
  health: ["health"] as const,
  graph: (depth: number) => ["graph", depth] as const,
  clusters: ["clusters"] as const,
  memories: (cluster?: number) => ["memories", cluster ?? "all"] as const,
  memory: (id: string) => ["memory", id] as const,
  conversations: (q?: string) => ["conversations", q ?? ""] as const,
  conversation: (id: string) => ["conversation", id] as const,
};

export function useHealth() {
  return useQuery({
    queryKey: qk.health,
    queryFn: api.health,
    refetchInterval: 30_000, // sidebar status dot polls every 30s
    staleTime: 0,
  });
}

/** Current signed-in user (drives the admin-only sidebar icon). */
export function useMe() {
  return useQuery({
    queryKey: ["me"] as const,
    queryFn: api.auth.me,
    staleTime: 60_000,
    retry: false,
  });
}

export function useGraph(depth = 2) {
  return useQuery({ queryKey: qk.graph(depth), queryFn: () => api.graph(depth) });
}

export function useClusters() {
  return useQuery({ queryKey: qk.clusters, queryFn: api.clusters });
}

export function useMemories(cluster?: number) {
  return useQuery({
    queryKey: qk.memories(cluster),
    queryFn: () => api.memories({ cluster, limit: 500 }),
  });
}

export function useMemory(id: string | null) {
  return useQuery({
    queryKey: qk.memory(id ?? ""),
    queryFn: () => api.memory(id as string),
    enabled: !!id,
  });
}

export function useConversations(q?: string) {
  return useQuery({
    queryKey: qk.conversations(q),
    queryFn: () => api.conversations({ limit: 100, q }),
    refetchInterval: 30_000, // background enrichment patches title/summary/facts
  });
}

export function useConversation(id: string | null) {
  return useQuery({
    queryKey: qk.conversation(id ?? ""),
    queryFn: () => api.conversation(id as string),
    enabled: !!id,
  });
}

export function useDeleteConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteConversation(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });
}

export function useForget() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.forget(id),
    onSuccess: () => refreshGraph(qc),
  });
}

export function useIngest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      userQuery,
      assistantReply,
      fast,
    }: {
      userQuery: string;
      assistantReply: string;
      fast: boolean;
    }) => api.ingest(userQuery, assistantReply, fast),
    onSuccess: () => refreshGraph(qc),
  });
}

/** Memories changed → refetch everything derived from them. */
function refreshGraph(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: ["graph"] });
  qc.invalidateQueries({ queryKey: ["clusters"] });
  qc.invalidateQueries({ queryKey: ["memories"] });
  qc.invalidateQueries({ queryKey: ["health"] });
}
