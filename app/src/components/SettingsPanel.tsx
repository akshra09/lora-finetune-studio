"use client";
// components/SettingsPanel.tsx
import { GenerationSettings } from "@/lib/types";

interface Props {
  settings: GenerationSettings;
  onChange: (s: GenerationSettings) => void;
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  hint?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-[var(--text-secondary)]">{label}</label>
        <span
          className="text-xs font-semibold text-[var(--brand)] tabular-nums"
          style={{ fontFamily: "var(--font-mono)" }}
        >
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      {hint && <p className="text-[10px] text-[var(--text-muted)]">{hint}</p>}
    </div>
  );
}

export default function SettingsPanel({ settings, onChange }: Props) {
  const set = (key: keyof GenerationSettings, value: unknown) =>
    onChange({ ...settings, [key]: value });

  return (
    <div className="space-y-5">
      {/* System Prompt */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-[var(--text-secondary)]">
          System Prompt
        </label>
        <textarea
          value={settings.systemPrompt}
          onChange={(e) => set("systemPrompt", e.target.value)}
          placeholder="You are a helpful assistant…"
          rows={3}
          className="w-full text-xs px-3 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg-base)] text-[var(--text-primary)] placeholder-[var(--text-muted)] resize-none input-glow transition-all"
        />
      </div>

      <div className="h-px bg-[var(--border)]" />

      <SliderRow
        label="Temperature"
        value={settings.temperature}
        min={0}
        max={1.5}
        step={0.05}
        hint="Higher = more creative / random"
        onChange={(v) => set("temperature", v)}
      />

      <SliderRow
        label="Max New Tokens"
        value={settings.maxNewTokens}
        min={32}
        max={1024}
        step={32}
        hint="Maximum tokens to generate"
        onChange={(v) => set("maxNewTokens", v)}
      />

      <SliderRow
        label="Top-P"
        value={settings.topP}
        min={0.1}
        max={1.0}
        step={0.05}
        hint="Nucleus sampling threshold"
        onChange={(v) => set("topP", v)}
      />

      <SliderRow
        label="Repetition Penalty"
        value={settings.repetitionPenalty}
        min={1.0}
        max={1.5}
        step={0.05}
        hint="Penalize repeated text"
        onChange={(v) => set("repetitionPenalty", v)}
      />

      {/* Reset */}
      <button
        onClick={() =>
          onChange({
            temperature: 0.7,
            maxNewTokens: 256,
            topP: 0.9,
            repetitionPenalty: 1.1,
            systemPrompt: "You are a helpful, concise assistant.",
          })
        }
        className="w-full py-2 rounded-lg text-xs text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
      >
        Reset to defaults
      </button>
    </div>
  );
}
