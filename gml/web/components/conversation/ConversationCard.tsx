"use client";

import { Sparkles } from "lucide-react";
import type { ApiConversation } from "@/lib/types";
import { relativeTime } from "@/lib/utils";

const MODEL_GLYPH: Record<string, string> = {
  claude: "◆",
  chatgpt: "◎",
  gemini: "✦",
};

export function ConversationCard({
  card,
  onOpen,
}: {
  card: ApiConversation;
  onOpen: () => void;
}) {
  return (
    <button
      onClick={onOpen}
      className="flex flex-col items-start gap-2 rounded-md border border-border bg-bg-1 p-4 text-left transition-colors duration-[180ms] ease-out hover:border-border-strong hover:bg-bg-2"
    >
      <div className="flex w-full items-center gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wider text-text-2">
          {card.source_model || "chat"}
        </span>
        <span className="ml-auto text-sm text-text-2">
          {MODEL_GLYPH[card.source_model ?? ""] ?? "·"}
        </span>
      </div>

      <h3 className="line-clamp-2 text-sm font-semibold leading-snug text-text-0">
        {card.title || card.user_prompt?.slice(0, 70) || "Saved conversation"}
      </h3>

      <p className="line-clamp-3 text-xs leading-relaxed text-text-1">
        {card.summary || card.ai_response?.slice(0, 200) || "—"}
      </p>

      <div className="mt-1 flex w-full items-center gap-3 border-t border-border pt-2 text-[10px] text-text-2">
        <span>{card.created_at ? relativeTime(card.created_at) : ""}</span>
        <span className="ml-auto inline-flex items-center gap-1">
          <Sparkles className="h-3 w-3" />
          {card.fact_count} memories
        </span>
      </div>
    </button>
  );
}
