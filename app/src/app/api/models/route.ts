// app/api/models/route.ts
import { NextResponse } from "next/server";
import { ModelConfig } from "@/lib/types";

const BUILT_IN_MODELS: ModelConfig[] = [
  {
    id: "Qwen/Qwen2.5-7B-Instruct",
    name: "Qwen 2.5 7B Instruct",
    description: "High quality, fast response, standard instruction model.",
    maxTokens: 2048,
    isCustom: false,
  },
  {
    id: "meta-llama/Llama-3.3-70B-Instruct",
    name: "Meta Llama 3.3 70B",
    description: "State-of-the-art 70B model from Meta.",
    maxTokens: 2048,
    isCustom: false,
  },
  {
    id: "Qwen/Qwen2.5-72B-Instruct",
    name: "Qwen 2.5 72B Instruct",
    description: "Extremely powerful 72B parameter model from Alibaba.",
    maxTokens: 2048,
    isCustom: false,
  },
];

export async function GET() {
  const models = [...BUILT_IN_MODELS];

  // Add user's custom model from env if set
  const customModel = process.env.DEFAULT_MODEL_ID;
  if (customModel && !models.find((m) => m.id === customModel) && !customModel.includes("your-lora-model")) {
    models.unshift({
      id: customModel,
      name: customModel.split("/").pop() ?? customModel,
      description: "Your fine-tuned LoRA model ✨",
      maxTokens: 2048,
      isCustom: true,
    });
  }

  return NextResponse.json({ models });
}
