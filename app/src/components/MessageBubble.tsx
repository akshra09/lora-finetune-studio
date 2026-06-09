"use client";
// components/MessageBubble.tsx
import { Message } from "@/lib/types";
import { User, Bot, AlertCircle, Clock, Copy, Check } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import clsx from "clsx";

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={clsx(
        "group flex gap-3 animate-slide-up",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center border",
          isUser
            ? "bg-[var(--accent-purple)] bg-opacity-15 border-[var(--accent-purple)] border-opacity-30"
            : message.error
            ? "bg-red-500 bg-opacity-15 border-red-500 border-opacity-30"
            : "bg-[var(--brand)] bg-opacity-15 border-[var(--brand)] border-opacity-30"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-[var(--accent-purple)]" />
        ) : message.error ? (
          <AlertCircle className="w-4 h-4 text-red-400" />
        ) : (
          <Bot className="w-4 h-4 text-[var(--brand)]" />
        )}
      </div>

      {/* Bubble */}
      <div className={clsx("flex-1 min-w-0", isUser ? "flex flex-col items-end" : "")}>
        {/* Role label */}
        <div
          className={clsx(
            "flex items-center gap-2 mb-1.5",
            isUser ? "flex-row-reverse" : ""
          )}
        >
          <span className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider">
            {isUser ? "You" : "Assistant"}
          </span>
          {message.latency && (
            <span className="flex items-center gap-1 text-[10px] text-[var(--text-muted)]">
              <Clock className="w-2.5 h-2.5" />
              {(message.latency / 1000).toFixed(1)}s
            </span>
          )}
          {message.model && !isUser && (
            <span
              className="text-[10px] text-[var(--text-muted)]"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              {message.model.split("/").pop()}
            </span>
          )}
        </div>

        {/* Content */}
        <div
          className={clsx(
            "relative rounded-xl px-4 py-3 text-sm leading-relaxed max-w-[85%]",
            isUser
              ? "bg-[var(--accent-purple)] bg-opacity-10 border border-[var(--accent-purple)] border-opacity-20 text-[var(--text-primary)]"
              : message.error
              ? "bg-red-500 bg-opacity-10 border border-red-500 border-opacity-20 text-red-300"
              : "bg-[var(--bg-card)] border border-[var(--border)] text-[var(--text-primary)]"
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose-dark prose prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}

          {/* Copy button (assistant only) */}
          {!isUser && !message.error && (
            <button
              onClick={handleCopy}
              className="absolute top-2 right-2 p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-[var(--bg-hover)] transition-all text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-[var(--brand)]" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
