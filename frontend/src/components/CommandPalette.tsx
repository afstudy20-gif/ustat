/**
 * CommandPalette — the ⌘K / Ctrl+K command box.
 *
 * Structured "click-to-build" interface (mirrors the Compute panel's pattern):
 * a left column of tappable variables, a row of test-keyword buttons, and a
 * central command area that accumulates the chosen tokens. The user doesn't
 * have to type a long sentence — they tap "roc", then tap "group", then "age".
 *
 * The single source of truth is the command string (`query`). Every token
 * insertion, manual keystroke, and ⌫/Clear operates on that string, so the
 * existing rule-based parser (commandParser.ts) keeps working unchanged — the
 * builder is just a friendlier way to produce the same input.
 *
 * On Enter the resolved fields are written into the target panel's panelCache
 * and the app switches to that tab — the form comes up pre-filled.
 *
 * Privacy: nothing leaves the browser. Parsing is a pure client-side function.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Search, CornerDownLeft, X } from "lucide-react";
import { useStore, analysisCols, type ColMeta } from "../store";
import { parseCommand, applyResult, INTENT_SCHEMAS, type ParseResult } from "../lib/commandParser";

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

/** Test-keyword buttons shown above the command area. Each inserts its token
 *  into the command string at the cursor. Labels are short so they read like
 *  natural prefixes: "roc", "ttest", "anova"… */
const TEST_TOKENS: { token: string; label: string; title: string }[] = [
  { token: "roc", label: "ROC", title: "ROC curve — binary outcome + numeric score" },
  { token: "ttest", label: "t-test", title: "Independent t-test — numeric + 2-level group" },
  { token: "onesample", label: "1-sample t", title: "One-sample t-test — numeric variable" },
  { token: "anova", label: "ANOVA", title: "One-way ANOVA — numeric + ≥3-level group" },
  { token: "mannwhitney", label: "Mann-Whitney", title: "Mann-Whitney U — numeric + 2-level group" },
  { token: "kruskal", label: "Kruskal-Wallis", title: "Kruskal-Wallis — numeric + ≥3-level group" },
  { token: "correlation", label: "Correlation", title: "Correlation — two numeric variables" },
];

/** Separator / operator tokens (Compute-style button row). */
const SEP_TOKENS: string[] = ["vs", "by", "and", ","];

/** Short kind badge colour per column measurement level. */
function kindBadge(kind: ColMeta["kind"]): { cls: string; ch: string } {
  switch (kind) {
    case "numeric": return { cls: "bg-blue-50 text-blue-600", ch: "n" };
    case "categorical": return { cls: "bg-amber-50 text-amber-600", ch: "c" };
    case "ordinal": return { cls: "bg-teal-50 text-teal-600", ch: "o" };
    case "date": return { cls: "bg-purple-50 text-purple-600", ch: "d" };
    default: return { cls: "bg-gray-100 text-gray-500", ch: "t" };
  }
}

export default function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  // Cursor position tracked from the textarea so token insertions land where
  // the user expects (mirrors the Compute FormulaTab's insert() helper).
  const selRef = useRef<{ start: number; end: number }>({ start: 0, end: 0 });
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const [varFilter, setVarFilter] = useState("");

  const session = useStore((s) => s.session);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const setPanelCache = useStore((s) => s.setPanelCache);
  // Subscribe to panelCache so applyResult's read-modify-write sees the latest
  // values (the store helper reads panelCache internally).
  const panelCache = useStore((s) => s.panelCache);

  const columns = useMemo(
    () => (session ? analysisCols(session.columns) : []),
    [session],
  );

  const filteredColumns = useMemo(() => {
    const f = varFilter.trim().toLowerCase();
    if (!f) return columns;
    return columns.filter((c) => c.name.toLowerCase().includes(f));
  }, [columns, varFilter]);

  const result: ParseResult = useMemo(
    () => parseCommand(query, columns),
    [query, columns],
  );

  // Focus + clear whenever the palette opens.
  useEffect(() => {
    if (open) {
      setQuery("");
      setVarFilter("");
      selRef.current = { start: 0, end: 0 };
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Escape closes.
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

  /** Insert a token at the current cursor position, padding with a leading
   *  space when needed so tokens stay word-separated (Compute pattern). */
  const insertAtCursor = (token: string) => {
    const el = inputRef.current;
    const { start, end } = selRef.current;
    const before = query.slice(0, start);
    const after = query.slice(end);
    // Prepend a space if we're not at the very start and the preceding char
    // isn't already whitespace — keeps "roc"+"group" from becoming "rocgroup".
    const needSpace = before.length > 0 && !/\s$/.test(before);
    const insert = (needSpace ? " " : "") + token;
    const next = before + insert + after;
    setQuery(next);
    const pos = start + insert.length;
    selRef.current = { start: pos, end: pos };
    // Restore focus + caret after React re-renders.
    requestAnimationFrame(() => {
      el?.focus();
      el?.setSelectionRange(pos, pos);
    });
  };

  /** Delete the token (word) immediately left of the cursor, or the selection. */
  const backspaceToken = () => {
    const el = inputRef.current;
    const { start, end } = selRef.current;
    if (start !== end) {
      const next = query.slice(0, start) + query.slice(end);
      setQuery(next);
      selRef.current = { start, end: start };
      requestAnimationFrame(() => { el?.focus(); el?.setSelectionRange(start, start); });
      return;
    }
    if (start === 0) return;
    // Walk back over trailing spaces, then over the preceding word.
    let i = start;
    while (i > 0 && /\s/.test(query[i - 1])) i--;
    while (i > 0 && !/\s/.test(query[i - 1])) i--;
    const next = query.slice(0, i) + query.slice(start);
    setQuery(next);
    selRef.current = { start: i, end: i };
    requestAnimationFrame(() => { el?.focus(); el?.setSelectionRange(i, i); });
  };

  const submit = () => {
    if (result.intent) {
      applyResult(result, { setActiveTab, setPanelCache });
      onClose();
    }
    void panelCache; // keep the subscription live so applyResult sees fresh cache
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[8vh] px-4 bg-black/30 backdrop-blur-[1px]"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden flex flex-col max-h-[78vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header: search icon + esc hint */}
        <div className="flex items-center justify-between gap-2 px-4 py-2.5 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <Search size={16} className="text-gray-400" />
            <span className="text-sm font-semibold text-gray-700">Command</span>
            <span className="text-[11px] text-gray-400">— build a test command by tapping tokens</span>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 p-1 rounded hover:bg-gray-100">
            <X size={15} />
          </button>
        </div>

        <div className="flex flex-1 min-h-0">
          {/* Left: variable list (Compute-style) */}
          <div className="w-48 flex-shrink-0 border-r border-gray-100 flex flex-col">
            <div className="p-2 border-b border-gray-100">
              <input
                value={varFilter}
                onChange={(e) => setVarFilter(e.target.value)}
                placeholder="Filter variables…"
                className="w-full text-xs px-2 py-1 border border-gray-200 rounded-md focus:outline-none focus:border-indigo-400"
              />
            </div>
            <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
              {filteredColumns.length === 0 ? (
                <p className="text-[11px] text-gray-400 px-1 py-2 text-center">No variables</p>
              ) : (
                filteredColumns.map((c) => {
                  const badge = kindBadge(c.kind);
                  return (
                    <button
                      key={c.name}
                      onClick={() => insertAtCursor(c.name)}
                      className="w-full flex items-center gap-1.5 text-left text-xs font-mono px-1.5 py-1 rounded hover:bg-indigo-50 hover:text-indigo-700 text-gray-700 group"
                      title={`${c.name} (${c.kind})`}
                    >
                      <span className="truncate flex-1">{c.name}</span>
                      <span className={`text-[9px] font-sans font-semibold rounded px-1 ${badge.cls}`}>{badge.ch}</span>
                    </button>
                  );
                })
              )}
            </div>
            <p className="text-[10px] text-gray-400 px-2 py-1.5 border-t border-gray-100">
              Tap to insert at cursor
            </p>
          </div>

          {/* Right: builder + preview */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Test-keyword buttons */}
            <div className="px-3 py-2 border-b border-gray-100">
              <p className="text-[10px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">Tests</p>
              <div className="flex flex-wrap gap-1">
                {TEST_TOKENS.map((t) => (
                  <button
                    key={t.token}
                    onClick={() => insertAtCursor(t.token)}
                    title={t.title}
                    className="px-2 py-1 text-xs rounded-md border border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 transition-colors"
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Command textarea */}
            <div className="px-3 py-2">
              <textarea
                ref={inputRef}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  selRef.current = { start: e.target.selectionStart ?? e.target.value.length, end: e.target.selectionEnd ?? e.target.value.length };
                }}
                onKeyUp={(e) => {
                  const t = e.target as HTMLTextAreaElement;
                  selRef.current = { start: t.selectionStart ?? query.length, end: t.selectionEnd ?? query.length };
                }}
                onClick={(e) => {
                  const t = e.target as HTMLTextAreaElement;
                  selRef.current = { start: t.selectionStart ?? query.length, end: t.selectionEnd ?? query.length };
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submit();
                  }
                }}
                placeholder="e.g.  roc group age   ·   ttest bmi sex   ·   correlation age sbp"
                rows={2}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono focus:border-indigo-500 focus:outline-none resize-none"
                autoComplete="off"
                spellCheck={false}
              />

              {/* Separator / operator buttons */}
              <div className="flex flex-wrap gap-1 mt-2">
                {SEP_TOKENS.map((op) => (
                  <button
                    key={op}
                    onClick={() => insertAtCursor(op)}
                    className="px-2 py-0.5 text-xs rounded border border-gray-300 hover:bg-gray-100 bg-white text-gray-600"
                  >
                    {op}
                  </button>
                ))}
                <button
                  onClick={backspaceToken}
                  className="px-2 py-0.5 text-xs rounded border border-gray-300 hover:bg-gray-100 bg-white text-gray-500"
                  title="Delete last token"
                >⌫</button>
                <button
                  onClick={() => { setQuery(""); selRef.current = { start: 0, end: 0 }; requestAnimationFrame(() => inputRef.current?.focus()); }}
                  className="px-2 py-0.5 text-xs rounded border border-gray-300 hover:bg-gray-100 bg-white text-gray-400"
                >Clear</button>
              </div>
            </div>

            {/* Live preview / examples */}
            <div className="flex-1 overflow-y-auto px-3 pb-3">
              {query.trim() === "" ? (
                <div className="text-[11px] text-gray-400 leading-relaxed">
                  <p className="mb-1.5">Tap a test, then tap your variables. Examples:</p>
                  <div className="flex flex-col gap-1">
                    {INTENT_SCHEMAS.map((s) => (
                      <span key={s.intent} className="font-mono text-gray-500">· {s.intent}</span>
                    ))}
                  </div>
                </div>
              ) : result.intent ? (
                <PreviewBlock result={result} />
              ) : (
                <p className="text-[11px] text-gray-500">
                  Start with a test token: <code className="text-indigo-600">roc</code>,{" "}
                  <code className="text-indigo-600">ttest</code>,{" "}
                  <code className="text-indigo-600">anova</code>,{" "}
                  <code className="text-indigo-600">correlation</code>…
                </p>
              )}
            </div>

            {/* Footer: submit hint */}
            <div className="px-3 py-2 border-t border-gray-100 flex items-center justify-between">
              <span className="text-[11px] text-gray-400">
                {result.intent
                  ? result.complete
                    ? "Ready — opens the form pre-filled"
                    : "Fill the missing slots or press Enter anyway"
                  : "Pick a test to build a command"}
              </span>
              <button
                onClick={submit}
                disabled={!result.intent}
                className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Open
                <CornerDownLeft size={12} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Live preview of the parsed command: title chip + resolved/missing slot chips. */
function PreviewBlock({ result }: { result: ParseResult }) {
  return (
    <div>
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-semibold text-indigo-600 bg-indigo-50 border border-indigo-200 rounded-md px-2 py-0.5">
          {result.title}
        </span>
        <span className="text-[10px] text-gray-400">→ {result.tab} tab</span>
      </div>
      {result.fields.length > 0 && (
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
      )}
    </div>
  );
}
