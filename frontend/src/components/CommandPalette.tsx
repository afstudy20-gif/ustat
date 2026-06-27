/**
 * CommandPalette — the ⌘K / Ctrl+K command box.
 *
 * A modal overlay with a single text input. As the user types, the rule-based
 * parser (commandParser.ts) resolves the query into an intent + column slots
 * and shows a live preview ("ROC curve · outcome=group, score=age"). On Enter
 * the resolved fields are written into the target panel's panelCache and the
 * app switches to that tab — the form comes up pre-filled.
 *
 * When the parser can't recognise an intent, the palette falls back to the
 * existing TEST_CATALOG search (plain tab navigation) so it never feels dead.
 *
 * Privacy: nothing leaves the browser. Parsing is a pure client-side function.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Search, CornerDownLeft, ArrowRight } from "lucide-react";
import { useStore, analysisCols } from "../store";
import { parseCommand, applyResult, INTENT_SCHEMAS, type ParseResult } from "../lib/commandParser";

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

/** Flat list of example commands shown when the input is empty, drawn from the
 *  supported intents. Helps the user discover the available vocabulary. */
const EXAMPLES: { intent: string; example: string }[] = [
  { intent: "roc", example: "roc group age" },
  { intent: "ttest_2sample", example: "ttest age group" },
  { intent: "ttest_1sample", example: "onesample bmi" },
  { intent: "anova", example: "anova sbp group" },
  { intent: "mannwhitney", example: "mannwhitney bmi sex" },
  { intent: "kruskal", example: "kruskal age group" },
  { intent: "correlation", example: "correlation bmi sbp" },
];

export default function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  const session = useStore((s) => s.session);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const setPanelCache = useStore((s) => s.setPanelCache);
  // Subscribe to panelCache so applyResult's read-modify-write sees the latest
  // values (the store helper reads panelCache internally).
  const panelCache = useStore((s) => s.panelCache);

  // Columns available for matching — analysis-eligible only, so excluded / id
  // columns aren't suggested.
  const columns = useMemo(
    () => (session ? analysisCols(session.columns) : []),
    [session],
  );

  const result: ParseResult = useMemo(
    () => parseCommand(query, columns),
    [query, columns],
  );

  // Focus the input whenever the palette opens, and clear on close so a
  // previous half-typed command doesn't linger.
  useEffect(() => {
    if (open) {
      setQuery("");
      // Defer to the next frame so the input is mounted before we focus it.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Escape closes; click on the backdrop closes. The ⌘K toggle is handled by
  // the parent (App.tsx) so it works whether the palette is open or not.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const submit = () => {
    // Prefer the parsed command when an intent was recognised. Otherwise, if
    // the user picked nothing, just close.
    if (result.intent) {
      applyResult(result, { setActiveTab, setPanelCache });
      onClose();
    }
    void panelCache; // keep the subscription live so applyResult sees fresh cache
  };

  // Build a friendly preview of resolved / missing slots.
  const slotPreview = result.fields.length > 0 && (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {result.fields.map((f) => {
        const filled = f.value != null;
        return (
          <span
            key={f.key}
            className={`text-[11px] px-2 py-0.5 rounded-md border ${
              filled
                ? "bg-indigo-50 border-indigo-200 text-indigo-700"
                : "bg-gray-50 border-gray-200 text-gray-400"
            }`}
          >
            {f.label}: <span className={filled ? "font-semibold" : "italic"}>{f.value ?? "—"}</span>
          </span>
        );
      })}
    </div>
  );

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[12vh] px-4 bg-black/30 backdrop-blur-[1px]"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Input row */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-100">
          <Search size={16} className="text-gray-400 flex-shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Type a command, e.g. roc group age  ·  ttest bmi sex  ·  correlation age sbp"
            className="flex-1 text-sm bg-transparent focus:outline-none placeholder-gray-400"
            autoComplete="off"
            spellCheck={false}
          />
          <kbd className="text-[10px] text-gray-400 bg-gray-100 border border-gray-200 rounded px-1.5 py-0.5">
            esc
          </kbd>
        </div>

        {/* Body: preview or examples */}
        <div className="max-h-80 overflow-y-auto">
          {query.trim() === "" ? (
            <div className="p-3">
              <p className="text-[10px] uppercase tracking-wide text-gray-400 font-semibold mb-2 px-1">
                Supported commands
              </p>
              <div className="space-y-0.5">
                {EXAMPLES.map((ex) => {
                  const schema = INTENT_SCHEMAS.find((s) => s.intent === ex.intent);
                  return (
                    <button
                      key={ex.intent}
                      onClick={() => setQuery(ex.example)}
                      className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-lg hover:bg-indigo-50 transition-colors text-left group"
                    >
                      <span className="flex flex-col">
                        <span className="text-sm text-gray-700 font-mono">{ex.example}</span>
                        <span className="text-[10px] text-gray-400">{schema?.title}</span>
                      </span>
                      <ArrowRight size={14} className="text-gray-300 group-hover:text-indigo-500 flex-shrink-0" />
                    </button>
                  );
                })}
              </div>
              <p className="text-[10px] text-gray-400 mt-3 px-1 leading-relaxed">
                Column names are matched loosely (typos, Turkish characters, partial names).
                No data leaves your browser.
              </p>
            </div>
          ) : result.intent ? (
            <div className="p-4">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-indigo-600 bg-indigo-50 border border-indigo-200 rounded-md px-2 py-0.5">
                  {result.title}
                </span>
                <span className="text-[10px] text-gray-400">→ {result.tab} tab</span>
              </div>
              {slotPreview}
              {result.fields.length > 0 && !result.complete && (
                <p className="text-[11px] text-amber-600 mt-2">
                  Missing:{" "}
                  {result.fields
                    .filter((f) => f.value == null)
                    .map((f) => f.label)
                    .join(", ")}
                </p>
              )}
              <p className="text-[11px] text-gray-400 mt-3 flex items-center gap-1">
                <CornerDownLeft size={12} /> Press Enter to open the form pre-filled. You'll still
                press Run yourself.
              </p>
            </div>
          ) : (
            <div className="p-4">
              <p className="text-xs text-gray-500">
                No command recognised. Try a supported verb like{" "}
                <code className="text-indigo-600">roc</code>,{" "}
                <code className="text-indigo-600">ttest</code>,{" "}
                <code className="text-indigo-600">anova</code>, or{" "}
                <code className="text-indigo-600">correlation</code> followed by your column names.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
