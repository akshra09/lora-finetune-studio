// lib/hf-client.ts — HuggingFace Inference API client

import { GenerationSettings, Message } from "./types";

const HF_ROUTER_BASE = "https://router.huggingface.co/v1";

export async function queryHuggingFace(
  modelId: string,
  messages: Message[],
  settings: GenerationSettings,
  apiKey: string
): Promise<{ text: string; tokens?: number }> {
  const formattedMessages = [];

  if (settings.systemPrompt && settings.systemPrompt.trim()) {
    formattedMessages.push({ role: "system", content: settings.systemPrompt.trim() });
  }

  for (const msg of messages) {
    formattedMessages.push({ role: msg.role, content: msg.content });
  }

  const response = await fetch(`${HF_ROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelId,
      messages: formattedMessages,
      temperature: settings.temperature,
      max_tokens: settings.maxNewTokens,
      top_p: settings.topP,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    const errorMessage = errData?.error?.message || `API error ${response.status}`;
    
    if (response.status === 401) {
      throw new Error("Invalid HuggingFace API key. Check your HUGGINGFACE_API_KEY.");
    }
    if (response.status === 404 || errorMessage.includes("not-supported") || errorMessage.includes("not supported")) {
      throw new Error(`Model not supported or not found: "${modelId}". Check the model ID or ensure it is supported by Hugging Face Inference Providers.`);
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();

  if (data?.choices && data.choices[0]?.message?.content !== undefined) {
    const text = data.choices[0].message.content as string;
    const tokens = data.usage?.completion_tokens;
    return { text: text.trim(), tokens };
  }

  throw new Error("Unexpected response format from HuggingFace API.");
}

export async function checkModelStatus(
  modelId: string,
  apiKey: string
): Promise<{ loaded: boolean; estimatedTime?: number }> {
  // For the router /v1 endpoints, status checking is done dynamically on request.
  // We can run a minimal completion request to verify model accessibility.
  try {
    const response = await fetch(`${HF_ROUTER_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: "ping" }],
        max_tokens: 1,
      }),
    });
    return { loaded: response.ok };
  } catch {
    return { loaded: false };
  }
}
