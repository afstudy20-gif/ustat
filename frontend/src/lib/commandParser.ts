/**
 * commandParser — rule-based natural-language command parsing for the ⌘K
 * command palette.
 *
 * No LLM, no network. Runs in milliseconds on the client, so the palette can
 * show a live preview as the user types. The vocabulary is dynamic: the
 * column list comes from the active session, so "age" only resolves when a
 * column named "age" actually exists.
 *
 * Pipeline (parseCommand):
 *   1. Intent detection  — match test keywords/aliases against the query,
 *      strip the matched tokens so they don't get mistaken for columns.
 *   2. Column extraction — split the remainder on connectors and quote pairs.
 *   3. Fuzzy column match — Turkish-character-aware, Levenshtein ≤ 2 or
 *      substring; also matches against label / display_name.
 *   4. Slot assignment   — for each recognised intent, fill its declared
 *      field slots from the matched columns, using column kind as a tiebreak.
 *
 * When no intent is recognised, the result is a plain "navigate to tab" action
 * (reusing the existing TEST_CATALOG search behaviour) with no field fill.
 */

import type { ColMeta } from "../store";
import { INTENT_SCHEMAS, SCHEMA_BY_INTENT, slotAccepts, type IntentSchema } from "./commandSchema";

export interface ParsedField {
  key: string;
  label: string;
  /** Resolved column name, or null if a required slot is still unfilled. */
  value: string | null;
}

export interface ParseResult {
  /** Null when no intent was recognised (plain navigation fallback). */
  intent: string | null;
  /** Human-readable title, e.g. "ROC curve". */
  title: string;
  /** Target tab id. */
  tab: string;
  /** Fields to write into panelCache, with resolved values. */
  fields: ParsedField[];
  /** True once every *required* field has a value. */
  complete: boolean;
  /** Short preview string for the palette UI, e.g.
   *  "ROC curve · outcome=group, score=age". */
  preview: string;
}

// ── Intent keyword table ──────────────────────────────────────────────
// Each entry maps an intent id to the tokens/phrases that trigger it. Tokens
// are matched case-insensitively as whole words or word-prefixes; multi-word
// phrases are matched as substrings. Turkish aliases are included so
// "lojistik", "ki kare", "korelasyon" work without an accent-perfect type.

interface IntentKeyword {
  intent: string;
  /** Single-word tokens (matched as \b token). */
  words: string[];
  /** Multi-word phrases (matched as substring). */
  phrases: string[];
}

const INTENT_KEYWORDS: IntentKeyword[] = [
  {
    intent: "roc",
    words: ["roc", "auc", "roccurve"],
    phrases: ["roc curve", "roc analysis", "roc eğrisi", "receiver operating"],
  },
  {
    intent: "ttest_2sample",
    words: ["ttest", "t-test", "ttest2", "student"],
    phrases: ["independent t", "two sample t", "two-sample t", "bağımsız t", "bagimsiz t", "iki örneklem t"],
  },
  {
    intent: "ttest_1sample",
    words: ["onesample"],
    phrases: ["one sample t", "one-sample t", "tek örneklem t", "tek örneklem t", "single sample t"],
  },
  {
    intent: "anova",
    words: ["anova"],
    phrases: ["one-way anova", "one way anova", "tek yönlü varyans", "tek yonlu varyans"],
  },
  {
    intent: "mannwhitney",
    words: ["mannwhitney", "mann-whitney", "mwu", "wilcoxonranksum"],
    phrases: ["mann whitney", "mann-whitney u", "rank sum", "rank-sum", "m-w"],
  },
  {
    intent: "kruskal",
    words: ["kruskal", "kw"],
    phrases: ["kruskal wallis", "kruskal-wallis"],
  },
  {
    intent: "correlation",
    words: ["correlation", "corr", "korelasyon", "pearson", "spearman"],
    phrases: ["correlation analysis", "korelasyon analizi"],
  },
];

// Connectors that separate column references: "a vs b", "a by b",
// "a göre b", "a ile b", comma, "and", "ve".
const CONNECTOR_RE = /\s+(?:vs\.?|versus|by|x|against|over|across|on|ile|göre|karşı|karsi|ve|and|then|then)\s+|,\s*|\s*;\s*/i;
const QUOTE_SENTINEL = "\u0001";

// ── Helpers ───────────────────────────────────────────────────────────

/** Collapse Turkish characters to ASCII so "yaş" ≈ "yas" ≈ "yas". Also
 *  lowercases and trims. Lets users type without perfect accents. */
function normalize(s: string): string {
  return s
    .toLocaleLowerCase("tr")
    .replace(/ı/g, "i")
    .replace(/İ/g, "i")
    .replace(/ş/g, "s")
    .replace(/ğ/g, "g")
    .replace(/ü/g, "u")
    .replace(/ö/g, "o")
    .replace(/ç/g, "c")
    .trim();
}

/** Iterative Levenshtein with a length-difference early exit. */
function levenshtein(a: string, b: string, max: number): number {
  if (a === b) return 0;
  const la = a.length, lb = b.length;
  if (Math.abs(la - lb) > max) return max + 1;
  if (la === 0) return lb;
  if (lb === 0) return la;
  let prev = new Array<number>(lb + 1);
  let cur = new Array<number>(lb + 1);
  for (let j = 0; j <= lb; j++) prev[j] = j;
  for (let i = 1; i <= la; i++) {
    cur[0] = i;
    let best = cur[0];
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= lb; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (cur[j] < best) best = cur[j];
    }
    if (best > max) return max + 1;
    [prev, cur] = [cur, prev];
  }
  return prev[lb];
}

export interface ColumnCandidate {
  /** Resolved canonical column name (exact, as in session.columns). */
  name: string;
  /** 0 = exact, higher = fuzzier. */
  score: number;
}

/** Find the best column match for a raw token against the session's columns.
 *  Matches name, label, and display_name. Returns null when nothing is close
 *  enough (Levenshtein > 2 and no substring overlap). */
function matchColumn(token: string, columns: ColMeta[]): ColumnCandidate | null {
  const t = normalize(token);
  if (!t) return null;
  let best: ColumnCandidate | null = null;
  for (const col of columns) {
    const candidates = [col.name, col.label, col.display_name].filter(Boolean) as string[];
    for (const raw of candidates) {
      const c = normalize(raw);
      let score = Infinity;
      if (c === t) score = 0;
      else if (c.includes(t) || t.includes(c)) {
        // Substring match — score by length difference so "age" matching
        // "age_group" ranks below an exact "age" column.
        score = Math.abs(c.length - t.length);
      } else {
        score = levenshtein(t, c, 2);
      }
      if (score <= 2 && (!best || score < best.score)) {
        best = { name: col.name, score };
      }
    }
  }
  return best;
}

// ── Intent detection ──────────────────────────────────────────────────

/** Try to detect an intent from the query. Returns the matched intent id and
 *  the query with intent tokens stripped (so they aren't read as columns). */
function detectIntent(query: string): { intent: string | null; remainder: string } {
  const norm = normalize(query);
  // Phrases first (more specific). First match wins — the table is ordered so
  // more specific intents (e.g. "two sample t") precede generic ones.
  for (const { intent, phrases } of INTENT_KEYWORDS) {
    for (const p of phrases) {
      if (norm.includes(normalize(p))) {
        return { intent, remainder: stripToken(query, p) };
      }
    }
  }
  // Then single-word tokens, matched as word boundaries on the normalised form.
  // We work on the normalised string for detection but strip from the original
  // query to preserve column-name casing.
  for (const { intent, words } of INTENT_KEYWORDS) {
    for (const w of words) {
      const wn = normalize(w);
      const re = new RegExp(`\\b${wn.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
      if (re.test(norm)) {
        return { intent, remainder: stripToken(query, w) };
      }
    }
  }
  return { intent: null, remainder: query };
}

/** Remove all occurrences of a token from the query (case-insensitive,
 *  Turkish-aware). */
function stripToken(query: string, token: string): string {
  const tn = normalize(token);
  const re = new RegExp(tn.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  return query.replace(re, " ").replace(/\s{2,}/g, " ").trim();
}

// ── Column extraction & slot assignment ───────────────────────────────

/** Pull column-reference chunks out of the (intent-stripped) remainder.
 *  Splits on connectors and quotes; filters out short noise tokens. */
function extractColumnTokens(remainder: string): string[] {
  const cleaned = remainder
    // Treat quoted strings as single units before splitting on connectors.
    .replace(/["'`]([^"'`]+)["'`]/g, ` ${QUOTE_SENTINEL}$1${QUOTE_SENTINEL} `)
    .split(CONNECTOR_RE)
    .map((s) => s.split(QUOTE_SENTINEL).join("").trim())
    .filter((s) => s.length >= 2);
  // Deduplicate while preserving order.
  const seen = new Set<string>();
  const out: string[] = [];
  for (const c of cleaned) {
    const k = normalize(c);
    if (k && !seen.has(k)) {
      seen.add(k);
      out.push(c);
    }
  }
  return out;
}

/** Assign matched columns to an intent's declared slots.
 *
 *  Strategy: prefer an exact kind fit (a slot wanting "numeric" gets a numeric
 *  column first). If kinds don't disambiguate, fill slots in declaration order
 *  with the remaining columns (position-based, as the user typed them). */
function assignSlots(
  schema: IntentSchema,
  matched: ColumnCandidate[],
  columns: ColMeta[],
): ParsedField[] {
  const kindByName = new Map(columns.map((c) => [c.name, c.kind]));
  const used = new Set<string>();
  const fields: ParsedField[] = [];

  // Pass 1: greedy kind-fit. For each slot, take the first unused column whose
  // kind the slot accepts.
  for (const slot of schema.fields) {
    let chosen: ColumnCandidate | null = null;
    for (const cand of matched) {
      if (used.has(cand.name)) continue;
      const kind = kindByName.get(cand.name);
      if (kind && slotAccepts(slot.kind, kind)) {
        chosen = cand;
        break;
      }
    }
    if (chosen) {
      used.add(chosen.name);
      fields.push({ key: slot.key, label: slot.label, value: chosen.name });
    } else {
      fields.push({ key: slot.key, label: slot.label, value: null });
    }
  }

  // Pass 2: fill any still-empty slot with leftover columns, in order.
  const leftovers = matched.filter((c) => !used.has(c.name));
  let li = 0;
  for (const field of fields) {
    if (field.value !== null) continue;
    if (li < leftovers.length) {
      field.value = leftovers[li].name;
      used.add(leftovers[li].name);
      li++;
    }
  }
  return fields;
}

// ── Public API ────────────────────────────────────────────────────────

/** Parse a natural-language command string into an actionable result.
 *
 *  `columns` is the active session's analysis-eligible columns (the caller
 *  should pass `analysisCols(session.columns)`). */
export function parseCommand(query: string, columns: ColMeta[]): ParseResult {
  const trimmed = query.trim();
  if (!trimmed) return emptyResult();

  const { intent, remainder } = detectIntent(trimmed);
  if (!intent) {
    // No intent → plain navigation fallback. The caller (palette) will show the
    // TEST_CATALOG search matches instead of a field preview.
    return {
      intent: null,
      title: "Search analyses…",
      tab: "",
      fields: [],
      complete: false,
      preview: "",
    };
  }

  const schema = SCHEMA_BY_INTENT[intent];
  if (!schema) return emptyResult();

  const tokens = extractColumnTokens(remainder);
  const matched: ColumnCandidate[] = [];
  for (const tok of tokens) {
    const m = matchColumn(tok, columns);
    if (m && !matched.some((c) => c.name === m.name)) matched.push(m);
  }

  const fields = assignSlots(schema, matched, columns);
  const complete = schema.fields
    .filter((f) => f.required)
    .every((f) => fields.find((pf) => pf.key === f.key)?.value != null);

  const parts = fields
    .filter((f) => f.value)
    .map((f) => `${f.label.replace(/\s*\(.*\)$/, "")}=${f.value}`);
  const preview = `${schema.title}${parts.length ? " · " + parts.join(", ") : ""}`;

  return {
    intent,
    title: schema.title,
    tab: schema.tab,
    fields,
    complete,
    preview,
  };
}

function emptyResult(): ParseResult {
  return {
    intent: null,
    title: "",
    tab: "",
    fields: [],
    complete: false,
    preview: "",
  };
}

/** Apply a ParseResult to the store: write the resolved fields into the
 *  target panel's cache (and select the combo sub-tab + test radio if the
 *  schema declares them), then switch to the target tab.
 *
 *  Exported so the palette's "run" handler is a one-liner and the store
 *  coupling lives next to the parser, not in the component. */
export function applyResult(
  result: ParseResult,
  store: {
    setActiveTab: (t: string) => void;
    setPanelCache: (panel: string, data: unknown) => void;
  },
): void {
  if (!result.intent) {
    if (result.tab) store.setActiveTab(result.tab);
    return;
  }
  const schema = SCHEMA_BY_INTENT[result.intent];
  if (!schema) return;

  // Combo sub-tab (e.g. Tests → hypothesis) must be written before the panel
  // mounts so usePersistedPanelState picks it up on first render.
  if (schema.comboId && schema.comboSub) {
    const cur = (store as unknown as {
      panelCache?: Record<string, Record<string, unknown>>;
    }).panelCache?.[schema.comboId] ?? {};
    store.setPanelCache(schema.comboId, { ...cur, sub: schema.comboSub });
  }

  // Build the field map for the target panel, merging over its existing cache
  // so we don't wipe unrelated fields the user may have set.
  const panelCache = (store as unknown as {
    panelCache?: Record<string, Record<string, unknown>>;
  }).panelCache;
  const existing = panelCache?.[schema.panelId] ?? {};
  const merged: Record<string, unknown> = { ...existing };
  if (schema.fixed) Object.assign(merged, schema.fixed);
  if (schema.testValue) merged.test = schema.testValue;
  for (const f of result.fields) {
    if (f.value != null) {
      // The Correlation pairwise panel stores its variables as a string[].
      if (schema.panelId === "correlation_pairwise" && f.key === "vars") {
        const prev = Array.isArray(merged.vars) ? (merged.vars as string[]) : [];
        merged.vars = Array.from(new Set([...prev, f.value]));
      } else {
        merged[f.key] = f.value;
      }
    }
  }
  store.setPanelCache(schema.panelId, merged);
  store.setActiveTab(schema.tab);
}

/** Re-export the schema list so the palette can render the supported commands
 *  as suggestions / placeholders. */
export { INTENT_SCHEMAS };
