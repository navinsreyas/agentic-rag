"use client";

import { useState } from "react";

// Deployed backend (Cloud Run).
const BACKEND = "https://agentic-rag-1065091982503.us-east1.run.app";

// ---- Request / response types --------------------------------------------
type ChatRequest = { message: string };

type ToolCall = {
  tool_name: string;
  args?: Record<string, unknown>;
  tool_call_id?: string | null;
};

// Events emitted by the backend's /chat/stream SSE endpoint.
type StreamEvent =
  | { type: "session"; session_id: string }
  | { type: "text"; content: string }
  | { type: "info"; content: string }
  | { type: "tools"; tools: ToolCall[] }
  | { type: "error"; content: string }
  | { type: "end" };

type Status = "idle" | "streaming" | "done" | "error";

// Small colour hint per retrieval path — the tool labels are the differentiator.
const TOOL_COLORS: Record<string, string> = {
  vector_search: "#2563eb",
  hybrid_search: "#7c3aed",
  graph_search: "#059669",
  get_entity_relationships: "#059669",
  get_entity_timeline: "#059669",
  page_vector_search: "#d97706",
  pageindex_search: "#dc2626",
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [tools, setTools] = useState<ToolCall[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState("");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q || status === "streaming") return;

    setAnswer("");
    setTools([]);
    setErrorMsg("");
    setStatus("streaming");

    try {
      const body: ChatRequest = { message: q };
      const res = await fetch(`${BACKEND}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok || !res.body) {
        setStatus("error");
        setErrorMsg(
          `HTTP ${res.status}` +
            (res.status === 429 ? " — rate limit reached (20/hour per IP). Try again later." : "")
        );
        return;
      }

      // The endpoint is POST, so EventSource can't be used — read the SSE stream
      // manually from the fetch body.
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let sawError = false;

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? ""; // keep the (possibly partial) last chunk

        for (const chunk of chunks) {
          const dataLine = chunk.split("\n").find((l) => l.startsWith("data:"));
          if (!dataLine) continue;
          const jsonStr = dataLine.slice(5).trim();
          if (!jsonStr) continue;

          let ev: StreamEvent;
          try {
            ev = JSON.parse(jsonStr) as StreamEvent;
          } catch {
            continue;
          }

          if (ev.type === "text") setAnswer((a) => a + ev.content);
          else if (ev.type === "info") setAnswer((a) => a + `\n[${ev.content}]\n`);
          else if (ev.type === "tools") setTools(ev.tools);
          else if (ev.type === "error") {
            sawError = true;
            setErrorMsg(ev.content);
          }
          // "session" and "end" need no UI handling here.
        }
      }

      setStatus(sawError ? "error" : "done");
    } catch (err) {
      setStatus("error");
      setErrorMsg(
        err instanceof Error
          ? `${err.message} (if this is a CORS error, the backend must allow this origin)`
          : "Network error."
      );
    }
  }

  return (
    <main
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: "2rem 1rem",
        fontFamily: "system-ui, sans-serif",
        lineHeight: 1.5,
      }}
    >
      <h1 style={{ fontSize: "1.4rem", marginBottom: "0.25rem" }}>Agentic RAG</h1>
      <p style={{ color: "#666", marginTop: 0, fontSize: "0.9rem" }}>
        Ask a question — the agent routes it to vector, hybrid, graph, or PageIndex retrieval.
      </p>

      <form onSubmit={onSubmit} style={{ display: "flex", gap: "0.5rem", margin: "1rem 0" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. How does OpenAI relate to Microsoft?"
          style={{
            flex: 1,
            padding: "0.6rem 0.75rem",
            fontSize: "1rem",
            border: "1px solid #ccc",
            borderRadius: 6,
          }}
        />
        <button
          type="submit"
          disabled={status === "streaming" || query.trim() === ""}
          style={{
            padding: "0.6rem 1.1rem",
            fontSize: "1rem",
            border: "none",
            borderRadius: 6,
            background: status === "streaming" ? "#94a3b8" : "#2563eb",
            color: "white",
            cursor: status === "streaming" ? "default" : "pointer",
          }}
        >
          {status === "streaming" ? "…" : "Ask"}
        </button>
      </form>

      {/* Answer */}
      <section
        style={{
          minHeight: 80,
          padding: "1rem",
          border: "1px solid #333",
          borderRadius: 8,
          background: "#1a1a1a",
          color: "#eee",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {answer ||
          (status === "streaming" ? (
            "Thinking…"
          ) : (
            <span style={{ color: "#999" }}>The answer will stream here.</span>
          ))}
      </section>

      {/* Tools used — the key differentiator */}
      {tools.length > 0 && (
        <div style={{ marginTop: "0.75rem" }}>
          <span style={{ fontSize: "0.8rem", color: "#666", marginRight: "0.5rem" }}>Tools used:</span>
          {tools.map((t, i) => (
            <span
              key={`${t.tool_name}-${i}`}
              style={{
                display: "inline-block",
                margin: "0 0.35rem 0.35rem 0",
                padding: "0.15rem 0.55rem",
                fontSize: "0.78rem",
                fontWeight: 600,
                color: "white",
                background: TOOL_COLORS[t.tool_name] ?? "#475569",
                borderRadius: 999,
              }}
            >
              {t.tool_name}
            </span>
          ))}
        </div>
      )}

      {status === "error" && (
        <p style={{ color: "#dc2626", marginTop: "0.75rem", fontSize: "0.9rem" }}>Error: {errorMsg}</p>
      )}
    </main>
  );
}
