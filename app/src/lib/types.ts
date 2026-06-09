// lib/types.ts — Shared TypeScript types

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  model?: string;
  tokens?: number;
  latency?: number; // ms
  error?: boolean;
}

export interface ModelConfig {
  id: string;
  name: string;
  description: string;
  maxTokens: number;
  isCustom: boolean;
}

export interface GenerationSettings {
  temperature: number;      // 0.0 – 1.5
  maxNewTokens: number;     // 64 – 1024
  topP: number;             // 0.5 – 1.0
  repetitionPenalty: number; // 1.0 – 1.5
  systemPrompt: string;
}

export interface ChatRequest {
  messages: Message[];
  modelId: string;
  settings: GenerationSettings;
}

export interface ChatResponse {
  content: string;
  model: string;
  tokens?: number;
  latency: number;
  error?: string;
}

export interface HealthStatus {
  status: "ok" | "error";
  hfApiConfigured: boolean;
  defaultModel: string;
  timestamp: string;
}
