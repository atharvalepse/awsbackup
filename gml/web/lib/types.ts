/**
 * Mirrors the backend `/api` contract (orchestration/api_routes.py).
 * Keep in sync if the Pydantic models change.
 */

export interface ApiMemory {
  id: string;
  content: string;
  entity: string | null;
  attribute: string | null;
  value: string | null;
  confidence: number;
  importance: number;
  cluster_id: number | null;
  source: string;
  pinned: boolean;
  timestamp: string;
  summary_short: string | null;
}

export interface MemoryListResponse {
  total: number;
  limit: number;
  offset: number;
  memories: ApiMemory[];
}

export interface ConversationFact {
  id: string | null;
  content: string;
  entity: string | null;
  attribute: string | null;
  value: string | null;
  confidence: number | null;
}

export interface ApiConversation {
  id: string;
  title: string | null;
  summary: string | null;
  user_prompt: string | null;
  ai_response: string | null;
  source_url: string | null;
  source_model: string | null;
  facts: ConversationFact[];
  fact_count: number;
  created_at: string | null;
}

export interface ConversationListResponse {
  total: number;
  limit: number;
  offset: number;
  conversations: ApiConversation[];
}

export type RelationshipKind = "similarity" | "entity";

export interface Relationship {
  memory_id: string;
  entity: string | null;
  value: string | null;
  cluster_id: number | null;
  kind: RelationshipKind;
  weight: number | null;
}

export interface MemoryDetailResponse {
  memory: ApiMemory;
  relationships: Relationship[];
}

export interface GraphNode {
  id: string;
  label: string;
  entity: string | null;
  value: string | null;
  cluster_id: number;
  importance: number;
  val: number;
  x: number;
  y: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  depth: number;
}

export interface Cluster {
  id: number;
  label: string;
  centroid: { x: number; y: number };
  size: number;
  color_hint: string;
}

export interface ClusterListResponse {
  clusters: Cluster[];
}

export interface RecallResult {
  memory: ApiMemory;
  score: number;
  why: string | null;
}

export interface RecallResponse {
  query: string;
  results: RecallResult[];
}

export interface IngestResponse {
  mode: "llm" | "sdp";
  count: number;
  created: string[];
  detail: string | null;
}

export interface TraceStage {
  stage: string;
  name: string;
  duration_ms: number;
  output: Record<string, unknown>;
}

export interface TraceAnnotations {
  retrieved_ids: string[];
  ranked: { id: string; final_score: number }[];
  sam_kept_ids: string[];
  assembled_ids: string[];
  improved_query: string | null;
  sam_reasoning: string | null;
}

export interface TraceResponse {
  query: { text: string; x: number; y: number };
  stages: TraceStage[];
  annotations: TraceAnnotations;
  formatted_context: string;
}

export interface SynthesizeResponse {
  query: string;
  context: string;
  items_included: number;
}

/** One SSE `stage` event from the streaming recall/trace endpoints. */
export interface StageEvent {
  stage: string;
  duration_ms: number;
}

/** The `done` payload from POST /api/memory/recall/stream. */
export interface RecallStreamDone {
  results: RecallResult[];
  improved_query: string | null;
  sam_reasoning: string | null;
  formatted_context: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  memories: number;
  embedder: string;
}
