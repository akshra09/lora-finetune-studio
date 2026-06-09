"use client";
// components/ModelSelector.tsx
import { useEffect, useState } from "react";
import { ChevronDown, Sparkles, Check } from "lucide-react";
import { ModelConfig } from "@/lib/types";
import clsx from "clsx";

interface Props {
  selectedId: string;
  onSelect: (id: string) => void;
}

export default function ModelSelector({ selectedId, onSelect }: Props) {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [customInput, setCustomInput] = useState("");

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models ?? []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const selected = models.find((m) => m.id === selectedId);

  const handleCustomAdd = () => {
    const id = customInput.trim();
    if (!id) return;
    const custom: ModelConfig = {
      id,
      name: id.split("/").pop() ?? id,
      description: "Custom model",
      maxTokens: 512,
      isCustom: true,
    };
    setModels((prev) => [custom, ...prev.filter((m) => m.id !== id)]);
    onSelect(id);
    setCustomInput("");
    setOpen(false);
  };

  return (
    <div className="relative w-full">
      {/* Trigger */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-2 px-3 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] hover:bg-[var(--bg-hover)] hover:border-[var(--brand)] transition-all text-left group"
      >
        <div className="flex items-center gap-2 min-w-0">
          <Sparkles className="w-3.5 h-3.5 text-[var(--brand)] flex-shrink-0" />
          <span className="text-sm text-[var(--text-primary)] truncate" style={{ fontFamily: "var(--font-mono)" }}>
            {loading ? "Loading…" : selected ? selected.name : "Select model"}
          </span>
          {selected?.isCustom && (
            <span className="flex-shrink-0 px-1.5 py-0.5 rounded text-[10px] bg-[var(--brand)] bg-opacity-20 text-[var(--brand)] font-medium">
              CUSTOM
            </span>
          )}
        </div>
        <ChevronDown className={clsx("w-4 h-4 text-[var(--text-muted)] transition-transform flex-shrink-0", open && "rotate-180")} />
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute top-full left-0 right-0 mt-1.5 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] shadow-2xl shadow-black/50 z-50 overflow-hidden animate-slide-up">
          <div className="p-2 space-y-0.5 max-h-72 overflow-y-auto">
            {models.map((m) => (
              <button
                key={m.id}
                onClick={() => { onSelect(m.id); setOpen(false); }}
                className={clsx(
                  "w-full text-left px-3 py-2.5 rounded-lg flex items-start gap-3 transition-colors group",
                  m.id === selectedId
                    ? "bg-[var(--brand)] bg-opacity-10 border border-[var(--border-glow)]"
                    : "hover:bg-[var(--bg-hover)]"
                )}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-[var(--text-primary)] truncate">
                      {m.name}
                    </span>
                    {m.isCustom && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-[var(--brand)] bg-opacity-20 text-[var(--brand)]">
                        YOUR MODEL
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-[var(--text-muted)] mt-0.5 truncate">{m.description}</p>
                </div>
                {m.id === selectedId && (
                  <Check className="w-4 h-4 text-[var(--brand)] flex-shrink-0 mt-0.5" />
                )}
              </button>
            ))}
          </div>

          {/* Custom model input */}
          <div className="p-2 border-t border-[var(--border)]">
            <p className="text-[10px] text-[var(--text-muted)] px-1 mb-1.5">
              Add any HuggingFace model ID:
            </p>
            <div className="flex gap-1.5">
              <input
                value={customInput}
                onChange={(e) => setCustomInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCustomAdd()}
                placeholder="username/model-name"
                className="flex-1 text-xs px-2.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-base)] text-[var(--text-primary)] placeholder-[var(--text-muted)] input-glow"
                style={{ fontFamily: "var(--font-mono)" }}
              />
              <button
                onClick={handleCustomAdd}
                disabled={!customInput.trim()}
                className="px-3 py-2 rounded-lg text-xs font-medium bg-[var(--brand)] text-[var(--bg-base)] hover:bg-[var(--brand-dark)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Add
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
