import { useState, useEffect, useMemo } from "react";
import { getSparklines } from "../../api";
import { isNumericKind, type Session } from "../../store";

export interface ModelData {
  numCols: string[];
  allCols: string[];
  /** Columns with <=2 unique non-null values, both in {0,1} — valid binary outcomes/events. */
  binaryCols: string[];
  /** Missing-value count per column name. */
  missingCounts: Record<string, number>;
  /** Per-column mini distribution data, loaded async from the API. */
  sparklines: Record<string, { type: string; data: number[] }>;
}

/** Derived column metadata + async sparklines for the Models panel.
 * Extracted from ModelsPanel so the panel owns flow/UI, not data shaping. */
export function useModelData(session: Session | null): ModelData {
  const numCols = session ? session.columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name) : [];
  const allCols = session ? session.columns.map((c) => c.name) : [];

  const binaryCols = useMemo(() => {
    if (!session) return [];
    const out: string[] = [];
    for (const col of session.columns) {
      const vals = new Set<unknown>();
      for (const row of session.preview) {
        const v = row[col.name];
        if (v == null || v === "") continue;
        vals.add(typeof v === "number" ? v : Number(v));
        if (vals.size > 2) break;
      }
      const arr = [...vals];
      if (arr.length === 0 || arr.length > 2) continue;
      if (arr.every((v) => v === 0 || v === 1)) out.push(col.name);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.session_id]);

  const missingCounts = useMemo(() => {
    if (!session) return {};
    const counts: Record<string, number> = {};
    for (const col of session.columns) {
      counts[col.name] = session.preview.filter(
        (row) => row[col.name] === null || row[col.name] === undefined || row[col.name] === ""
      ).length;
    }
    return counts;
    // `session` is read inside the body but we depend on its primitive
    // sub-fields so we don't re-count on every unrelated session reidentify.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.preview, session?.columns]);

  const [sparklines, setSparklines] = useState<Record<string, { type: string; data: number[] }>>({});
  useEffect(() => {
    if (!session) return;
    getSparklines(session.session_id)
      .then((r) => setSparklines(r.data))
      .catch(() => {});
    // Re-fetch only when the dataset changes — depending on `session` itself
    // would refetch on every cell edit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.session_id]);

  return { numCols, allCols, binaryCols, missingCounts, sparklines };
}
