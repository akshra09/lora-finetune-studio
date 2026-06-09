// app/api/health/route.ts
import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    hfApiConfigured: !!process.env.HUGGINGFACE_API_KEY,
    defaultModel: process.env.DEFAULT_MODEL_ID ?? "not-set",
    timestamp: new Date().toISOString(),
  });
}
