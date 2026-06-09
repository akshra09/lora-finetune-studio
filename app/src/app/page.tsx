// app/page.tsx
import Header from "@/components/Header";
import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <main className="h-screen flex flex-col overflow-hidden grid-bg">
      <Header
        modelId={process.env.DEFAULT_MODEL_ID ?? "Qwen/Qwen2.5-7B-Instruct"}
        isConnected={!!process.env.HUGGINGFACE_API_KEY}
      />
      <div className="flex-1 overflow-hidden">
        <ChatInterface />
      </div>
    </main>
  );
}
