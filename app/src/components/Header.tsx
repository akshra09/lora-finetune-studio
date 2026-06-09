"use client";
// components/Header.tsx
import { Brain, Github, Zap } from "lucide-react";

interface HeaderProps {
  modelId: string;
  isConnected: boolean;
}

export default function Header({ modelId, isConnected }: HeaderProps) {
  const shortModel = modelId ? modelId.split("/").pop() ?? modelId : "No model";

  return (
    <header className="flex items-center justify-between px-5 py-3.5 border-b border-[var(--border)] bg-[var(--bg-elevated)] backdrop-blur z-10">
      {/* Left: Logo */}
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="w-8 h-8 rounded-lg bg-[var(--brand)] bg-opacity-15 border border-[var(--brand)] border-opacity-30 flex items-center justify-center">
            <Brain className="w-4 h-4 text-[var(--brand)]" />
          </div>
          <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-[var(--brand)] border-2 border-[var(--bg-elevated)]" />
        </div>
        <div>
          <h1
            className="text-sm font-semibold tracking-tight text-[var(--text-primary)]"
            style={{ fontFamily: "var(--font-mono)" }}
          >
            LoRA Studio
          </h1>
          <p className="text-[10px] text-[var(--text-muted)] -mt-0.5">Fine-Tuning Playground</p>
        </div>
      </div>

      {/* Center: Active model badge */}
      <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--border)] bg-[var(--bg-card)]">
        <Zap className="w-3 h-3 text-[var(--brand)]" />
        <span className="text-xs text-[var(--text-secondary)] font-medium" style={{ fontFamily: "var(--font-mono)" }}>
          {shortModel}
        </span>
        <span
          className={`w-1.5 h-1.5 rounded-full ${
            isConnected ? "bg-emerald-400" : "bg-red-400"
          }`}
        />
      </div>

      {/* Right: Links */}
      <div className="flex items-center gap-2">
        <a
          href="https://github.com/YOUR_USERNAME/lora-finetune-studio"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors"
        >
          <Github className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">GitHub</span>
        </a>
        <a
          href="https://huggingface.co/docs/peft"
          target="_blank"
          rel="noopener noreferrer"
          className="px-3 py-1.5 rounded-md text-xs font-medium text-[var(--brand)] border border-[var(--border-glow)] hover:bg-[var(--brand-glow)] transition-colors"
        >
          PEFT Docs
        </a>
      </div>
    </header>
  );
}
