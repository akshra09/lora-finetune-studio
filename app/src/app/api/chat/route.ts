// app/api/chat/route.ts
import { NextRequest, NextResponse } from "next/server";
import { queryHuggingFace } from "@/lib/hf-client";
import { ChatRequest } from "@/lib/types";

export const runtime = "edge"; // Use Edge runtime for low latency

export async function POST(req: NextRequest) {
  const start = Date.now();

  try {
    const body = (await req.json()) as ChatRequest;
    const { messages, modelId, settings } = body;

    // Validate
    if (!modelId) {
      return NextResponse.json({ error: "modelId is required" }, { status: 400 });
    }
    if (!messages || messages.length === 0) {
      return NextResponse.json({ error: "messages array is required" }, { status: 400 });
    }

    const apiKey = process.env.HUGGINGFACE_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        {
          error:
            "HUGGINGFACE_API_KEY is not configured. Add it to your .env.local file " +
            "or Vercel environment variables.",
        },
        { status: 500 }
      );
    }

    // Call HuggingFace
    const result = await queryHuggingFace(modelId, messages, settings, apiKey);

    const latency = Date.now() - start;

    return NextResponse.json({
      content: result.text,
      model: modelId,
      tokens: result.tokens,
      latency,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    const latency = Date.now() - start;

    console.error("[chat/route] Error:", message);

    return NextResponse.json(
      { error: message, latency },
      { status: 500 }
    );
  }
}
