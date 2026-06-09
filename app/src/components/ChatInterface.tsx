"use client";
// components/ChatInterface.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { Send, Trash2, Settings2, X, ChevronRight } from "lucide-react";
import TextareaAutosize from "react-textarea-autosize";
import { Message, GenerationSettings } from "@/lib/types";
import MessageBubble from "./MessageBubble";
import ModelSelector from "./ModelSelector";
import SettingsPanel from "./SettingsPanel";
import clsx from "clsx";

const DEFAULT_SETTINGS: GenerationSettings = {
  temperature: 0.7,
  maxNewTokens: 256,
  topP: 0.9,
  repetitionPenalty: 1.1,
  systemPrompt: "You are a helpful, concise assistant.",
};

const DEFAULT_MODEL = process.env.NEXT_PUBLIC_DEFAULT_MODEL ?? "Qwen/Qwen2.5-7B-Instruct";

function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-lg flex items-center justify-center border bg-[var(--brand)] bg-opacity-15 border-[var(--brand)] border-opacity-30 flex-shrink-0">
        <span className="text-[var(--brand)] text-xs">AI</span>
      </div>
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl px-4 py-3 flex items-center gap-1.5">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  );
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Health check
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => {
        setIsConnected(d.hfApiConfigured);
        if (d.defaultModel && d.defaultModel !== "not-set") {
          setModelId(d.defaultModel);
        }
      })
      .catch(() => setIsConnected(false));
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...messages, userMsg],
          modelId,
          settings,
        }),
      });

      const data = await res.json();

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.error ?? data.content ?? "No response.",
        timestamp: Date.now(),
        model: modelId,
        latency: data.latency,
        error: !!data.error,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Network error. Check your connection and try again.",
          timestamp: Date.now(),
          error: true,
        },
      ]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [input, isLoading, messages, modelId, settings]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => setMessages([]);

  const STARTERS = [
    "Explain what LoRA fine-tuning is",
    "What are the key LoRA hyperparameters?",
    "How does QLoRA reduce memory usage?",
    "Write a Python function to reverse a string",
  ];

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Left Sidebar ── */}
      <aside
        className={clsx(
          "flex-shrink-0 border-r border-[var(--border)] bg-[var(--bg-elevated)] flex flex-col transition-all duration-300",
          showSettings ? "w-72" : "w-0 md:w-72 overflow-hidden"
        )}
      >
        <div className="p-4 border-b border-[var(--border)] flex items-center justify-between">
          <span className="text-sm font-semibold text-[var(--text-primary)]">Configuration</span>
          <button
            onClick={() => setShowSettings(false)}
            className="md:hidden p-1 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* Model selector */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
              Model
            </label>
            <ModelSelector selectedId={modelId} onSelect={setModelId} />
            {isConnected === false && (
              <p className="text-[11px] text-red-400 flex items-start gap-1.5">
                <span>⚠</span>
                <span>HUGGINGFACE_API_KEY not configured. Add it to .env.local</span>
              </p>
            )}
          </div>

          <div className="h-px bg-[var(--border)]" />

          {/* Generation settings */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
              Generation
            </label>
            <SettingsPanel settings={settings} onChange={setSettings} />
          </div>
        </div>

        {/* Bottom actions */}
        <div className="p-4 border-t border-[var(--border)]">
          <button
            onClick={clearChat}
            disabled={messages.length === 0}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-xs text-[var(--text-muted)] border border-[var(--border)] hover:border-red-500 hover:border-opacity-40 hover:text-red-400 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear conversation
          </button>
        </div>
      </aside>

      {/* ── Main Chat Area ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Mobile settings toggle */}
        <div className="md:hidden flex items-center gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
          <button
            onClick={() => setShowSettings(true)}
            className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
          >
            <Settings2 className="w-3.5 h-3.5" />
            Settings
          </button>
          <span className="text-[var(--text-muted)]">·</span>
          <span className="text-xs text-[var(--text-muted)]" style={{ fontFamily: "var(--font-mono)" }}>
            {modelId.split("/").pop()}
          </span>
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6">
          {messages.length === 0 ? (
            /* Empty state */
            <div className="h-full flex flex-col items-center justify-center text-center py-12 animate-fade-in">
              <div className="w-16 h-16 rounded-2xl bg-[var(--brand)] bg-opacity-10 border border-[var(--brand)] border-opacity-20 flex items-center justify-center mb-6">
                <span className="text-3xl">🧠</span>
              </div>
              <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-2">
                LoRA Fine-Tuning Studio
              </h2>
              <p className="text-sm text-[var(--text-secondary)] max-w-sm mb-8">
                Chat with your fine-tuned model. Select a model from the sidebar or type a message to begin.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
                {STARTERS.map((s) => (
                  <button
                    key={s}
                    onClick={() => { setInput(s); inputRef.current?.focus(); }}
                    className="flex items-center gap-2 px-4 py-3 rounded-xl border border-[var(--border)] hover:border-[var(--brand)] hover:border-opacity-50 hover:bg-[var(--bg-hover)] text-left text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all group"
                  >
                    <ChevronRight className="w-3 h-3 text-[var(--brand)] flex-shrink-0" />
                    <span>{s}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m) => <MessageBubble key={m.id} message={m} />)
          )}

          {isLoading && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input bar */}
        <div className="flex-shrink-0 px-4 md:px-8 py-4 border-t border-[var(--border)] bg-[var(--bg-elevated)]">
          <div className="flex items-end gap-3 max-w-4xl mx-auto">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="md:hidden flex-shrink-0 p-2.5 rounded-lg border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--brand)] hover:border-[var(--brand)] transition-colors mb-0.5"
            >
              <Settings2 className="w-4 h-4" />
            </button>

            <div className="flex-1 relative">
              <TextareaAutosize
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything… (Enter to send, Shift+Enter for newline)"
                minRows={1}
                maxRows={6}
                disabled={isLoading}
                className="w-full px-4 py-3 pr-12 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] text-[var(--text-primary)] placeholder-[var(--text-muted)] text-sm resize-none input-glow transition-all disabled:opacity-60"
              />
            </div>

            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className={clsx(
                "flex-shrink-0 p-3 rounded-xl transition-all mb-0.5",
                input.trim() && !isLoading
                  ? "bg-[var(--brand)] hover:bg-[var(--brand-dark)] text-[var(--bg-base)] shadow-lg shadow-[var(--brand-glow)] glow-brand"
                  : "bg-[var(--bg-card)] border border-[var(--border)] text-[var(--text-muted)] cursor-not-allowed"
              )}
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-center text-[10px] text-[var(--text-muted)] mt-2">
            Responses generated by HuggingFace Inference API · Model may produce errors
          </p>
        </div>
      </div>
    </div>
  );
}
