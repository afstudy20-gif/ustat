/**
 * commandSchema — declarative mapping from a recognised command *intent* to the
 * UI surface it targets (tab, panelId, optional combo sub-tab) and the column
 * slots it can pre-fill.
 *
 * The parser (commandParser.ts) resolves a natural-language string to one of
 * these intents, then assigns the user's column references into the declared
 * slots. Because the pre-fill is done by writing into the Zustand `panelCache`
 * *before* the target tab mounts, the slot `key` names must exactly match the
 * `usePersistedPanelState(panelId, key, ...)` declarations in each panel.
 *
 * MVP scope: ROC + the most common Tests and Correlation entries. Anything not
 * listed here falls back to plain tab navigation (the existing header search
 * behaviour).
 */

import type { ColKind } from "../store";

/** What kind of column a slot accepts. Mirrors the isNumericKind /
 *  isCategoricalKind helpers in store.ts — ordinal is accepted by both the
 *  numeric and the categorical filters (it is numeric-coded ordered data). */
export type SlotKind = "numeric" | "categorical" | "any";

/** A single fillable field in the target panel. */
export interface FieldSlot {
  /** Exact `key` the panel reads via usePersistedPanelState. */
  key: string;
  /** Human-readable label for the preview shown in the palette. */
  label: string;
  /** Column kind this slot accepts. Used to disambiguate when the user gives
   *  bare column names without saying which role each plays. */
  kind: SlotKind;
  /** When true the slot must be filled for the command to be considered
   *  complete (drives the preview's "missing: …" hint). */
  required?: boolean;
}

export interface IntentSchema {
  /** Identifier, e.g. "roc", "ttest_2sample". */
  intent: string;
  /** One-line description shown in the palette preview. */
  title: string;
  /** Target top-level tab id (matches TABS in App.tsx). */
  tab: string;
  /** panelCache namespace the target panel reads from. */
  panelId: string;
  /** Combo sub-tab to select (for tabs like "tests"/"models" that wrap
   *  several sub-panels). Written to panelCache[comboId].sub. */
  comboId?: string;
  comboSub?: string;
  /** For the Hypothesis panel: the `test` radio value to select. */
  testValue?: string;
  /** Fixed field values the command always sets (e.g. ROC mode="single"). */
  fixed?: Record<string, unknown>;
  /** Column slots to fill from the parsed column references. */
  fields: FieldSlot[];
}

/** Resolve a SlotKind against a column kind using the same ordinal-accepts-both
 *  rule as store.ts's isNumericKind / isCategoricalKind. */
export function slotAccepts(slot: SlotKind, kind: ColKind): boolean {
  if (slot === "any") return true;
  if (slot === "numeric") return kind === "numeric" || kind === "ordinal";
  // categorical
  return kind === "categorical" || kind === "ordinal";
}

/**
 * The supported intents, in display order. Each row is self-contained — the
 * parser does not hardcode any test-specific logic, it just reads this table.
 */
export const INTENT_SCHEMAS: IntentSchema[] = [
  {
    intent: "roc",
    title: "ROC curve",
    tab: "roc",
    panelId: "roc",
    fixed: { mode: "single" },
    fields: [
      { key: "outcomeCol", label: "Outcome (binary)", kind: "categorical", required: true },
      { key: "scoreCol", label: "Score (numeric)", kind: "numeric", required: true },
    ],
  },
  {
    intent: "ttest_2sample",
    title: "Independent t-test",
    tab: "tests",
    panelId: "hypothesis",
    comboId: "combo_tests",
    comboSub: "hypothesis",
    testValue: "ttest_2sample",
    fields: [
      { key: "col", label: "Variable (numeric)", kind: "numeric", required: true },
      { key: "groupCol", label: "Group (categorical)", kind: "categorical", required: true },
    ],
  },
  {
    intent: "ttest_1sample",
    title: "One-sample t-test",
    tab: "tests",
    panelId: "hypothesis",
    comboId: "combo_tests",
    comboSub: "hypothesis",
    testValue: "ttest_1sample",
    fields: [
      { key: "col", label: "Variable (numeric)", kind: "numeric", required: true },
    ],
  },
  {
    intent: "anova",
    title: "One-way ANOVA",
    tab: "tests",
    panelId: "hypothesis",
    comboId: "combo_tests",
    comboSub: "hypothesis",
    testValue: "anova",
    fields: [
      { key: "col", label: "Variable (numeric)", kind: "numeric", required: true },
      { key: "groupCol", label: "Group (≥3 levels)", kind: "categorical", required: true },
    ],
  },
  {
    intent: "mannwhitney",
    title: "Mann-Whitney U",
    tab: "tests",
    panelId: "hypothesis",
    comboId: "combo_tests",
    comboSub: "hypothesis",
    testValue: "mannwhitney",
    fields: [
      { key: "col", label: "Variable (numeric)", kind: "numeric", required: true },
      { key: "groupCol", label: "Group (2 levels)", kind: "categorical", required: true },
    ],
  },
  {
    intent: "kruskal",
    title: "Kruskal-Wallis",
    tab: "tests",
    panelId: "hypothesis",
    comboId: "combo_tests",
    comboSub: "hypothesis",
    testValue: "kruskal",
    fields: [
      { key: "col", label: "Variable (numeric)", kind: "numeric", required: true },
      { key: "groupCol", label: "Group (≥3 levels)", kind: "categorical", required: true },
    ],
  },
  {
    intent: "correlation",
    title: "Correlation (Pearson / Spearman)",
    tab: "correlation",
    panelId: "correlation_pairwise",
    fields: [
      { key: "vars", label: "Variables (≥2 numeric)", kind: "numeric", required: true },
    ],
  },
];

/** Quick lookup by intent id. */
export const SCHEMA_BY_INTENT: Record<string, IntentSchema> = Object.fromEntries(
  INTENT_SCHEMAS.map((s) => [s.intent, s]),
);
