"use client";

import { useState } from "react";
import { Search } from "lucide-react";
import { useConversations } from "@/hooks/use-gml";
import { ConversationCard } from "@/components/conversation/ConversationCard";
import { ConversationModal } from "@/components/conversation/ConversationModal";

export default function ConversationsPage() {
  const [q, setQ] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);
  const { data, isLoading } = useConversations(q.length > 2 ? q : undefined);
  const cards = data?.conversations ?? [];

  return (
    <div className="h-full overflow-y-auto px-8 py-8">
      <div className="mb-6 flex items-center gap-4">
        <h1 className="text-xl font-semibold text-text-0">Conversations</h1>
        <span className="text-xs text-text-2">{data?.total ?? 0} saved</span>
        <div className="ml-auto flex items-center gap-2 rounded-md border border-border bg-bg-1 px-3 py-1.5">
          <Search className="h-3.5 w-3.5 text-text-2" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search conversations…"
            className="w-56 bg-transparent text-sm text-text-0 outline-none placeholder:text-text-2"
          />
        </div>
      </div>

      {isLoading ? (
        <p className="text-sm text-text-2">Loading…</p>
      ) : cards.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 py-24 text-center">
          <div className="text-3xl opacity-30">🧠</div>
          <div className="text-sm font-medium text-text-1">No conversations yet</div>
          <div className="max-w-sm text-xs text-text-2">
            Chat on Claude, ChatGPT, or Gemini with the extension active — each
            turn is saved here as a card.
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">
          {cards.map((c) => (
            <ConversationCard key={c.id} card={c} onOpen={() => setOpenId(c.id)} />
          ))}
        </div>
      )}

      {openId && (
        <ConversationModal id={openId} onClose={() => setOpenId(null)} />
      )}
    </div>
  );
}
