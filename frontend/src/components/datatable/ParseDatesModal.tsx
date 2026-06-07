import { useState, useEffect, useCallback } from "react";
import { parseColumnDates } from "../../api";
import { useStore } from "../../store";
import type { ColMeta, Session } from "../../store";

interface DateStats {
  n_total: number;
  n_ok: number;
  n_bad: number;
  n_empty: number;
  order_used: string;
}
interface DateSample { raw: string | null; parsed: string | null }

/** Parse a mixed-format text column into real dates (datetime64, in place).
 *
 * Handles numeric separators, TR/EN month names, Excel serial numbers and
 * 2-digit years, resolving DMY/MDY ambiguity across the whole column. Stored
 * as datetime64 (ISO) so survival / time-series read it directly. Mirrors the
 * FindReplaceModal pattern with a live preview. */
export function ParseDatesModal({
  colName, columns, session, onClose, onApplied,
}: {
  colName: string;
  columns: ColMeta[];
  session: Session;
  onClose: () => void;
  onApplied: () => void;
}) {
  const col = columns.find((c) => c.name === colName);
  const [order, setOrder] = useState<"auto" | "dmy" | "mdy">("auto");
  const [century, setCentury] = useState(50);
  const [sample, setSample] = useState<DateSample[]>([]);
  const [stats, setStats] = useState<DateStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const errMsg = (e: unknown) =>
    (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;

  const loadPreview = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await parseColumnDates(session.session_id, {
        column: colName, order, century_threshold: century, preview_only: true,
      });
      setSample(res.data.sample ?? []);
      setStats(res.data.stats ?? null);
    } catch (e: unknown) {
      setError(errMsg(e) ?? "Preview failed");
    } finally {
      setLoading(false);
    }
  }, [session.session_id, colName, order, century]);

  useEffect(() => { loadPreview(); }, [loadPreview]);

  const handleApply = async () => {
    setApplying(true); setError(null);
    try {
      const res = await parseColumnDates(session.session_id, {
        column: colName, order, century_threshold: century,
      });
      const d = res.data;
      const newCol: ColMeta = { ...(col ?? { name: colName }), name: d.name, dtype: d.dtype, kind: d.kind };
      useStore.getState().addSessionColumn(newCol, d.preview_values);
      onApplied();
      onClose();
    } catch (e: unknown) {
      setError(errMsg(e) ?? "Parse failed");
    } finally {
      setApplying(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl w-[30rem] max-h-[85vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">Parse as date</h3>
            <p className="text-[11px] text-gray-400 mt-0.5">
              {colName}
              {col?.kind && <span className="ml-1 text-indigo-500">({col.kind})</span>}
            </p>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-lg">✕</button>
        </div>

        {/* Options */}
        <div className="px-5 pt-3 grid grid-cols-2 gap-3">
          <div>
            <label className="text-[11px] text-gray-500 block mb-1">Day/Month ambiguity</label>
            <select value={order} onChange={(e) => setOrder(e.target.value as typeof order)}
              className="select text-xs w-full py-1">
              <option value="auto">Auto-detect</option>
              <option value="dmy">Day first (DD/MM)</option>
              <option value="mdy">Month first (MM/DD)</option>
            </select>
          </div>
          <div>
            <label className="text-[11px] text-gray-500 block mb-1">2-digit year cutoff</label>
            <input type="number" min={0} max={99} value={century}
              onChange={(e) => setCentury(Math.max(0, Math.min(99, parseInt(e.target.value) || 0)))}
              className="select text-xs w-full py-1" />
          </div>
        </div>

        {/* Stats */}
        {stats && (
          <div className="px-5 pt-2 flex flex-wrap gap-1.5 text-[11px]">
            <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">{stats.n_ok} parsed</span>
            {stats.n_bad > 0 && <span className="px-2 py-0.5 rounded bg-red-50 text-red-600 border border-red-200">{stats.n_bad} unrecognised</span>}
            {stats.n_empty > 0 && <span className="px-2 py-0.5 rounded bg-gray-50 text-gray-500 border border-gray-200">{stats.n_empty} empty</span>}
            <span className="px-2 py-0.5 rounded bg-indigo-50 text-indigo-600 border border-indigo-200">order: {stats.order_used.toUpperCase()}</span>
          </div>
        )}

        {/* Preview */}
        <div className="flex-1 overflow-y-auto px-5 py-3">
          <p className="text-[10px] text-gray-400 uppercase tracking-wide mb-1">Preview (first rows)</p>
          {loading ? (
            <p className="text-xs text-gray-400 py-4 text-center">Loading…</p>
          ) : sample.length === 0 ? (
            <p className="text-xs text-gray-400 py-4 text-center">No values</p>
          ) : (
            <div className="space-y-1">
              {sample.map((s, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  <span className="max-w-[12rem] truncate font-mono text-gray-500 bg-gray-100 px-2 py-0.5 rounded flex-shrink-0" title={s.raw ?? ""}>
                    {s.raw === null || s.raw === "" ? "—" : s.raw}
                  </span>
                  <span className="text-gray-400">→</span>
                  <span className={`font-mono ${s.parsed ? "text-emerald-700" : "text-red-400"}`}>
                    {s.parsed ?? (s.raw ? "⚠ unrecognised" : "")}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {error && <p className="px-5 text-xs text-red-500">{error}</p>}

        {/* Footer */}
        <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-between">
          <p className="text-[10px] text-gray-400 max-w-[14rem] leading-snug">
            Stored as a real date (ISO). Overwrites this column in place.
          </p>
          <div className="flex gap-2">
            <button onClick={onClose}
              className="px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50">
              Cancel
            </button>
            <button onClick={handleApply} disabled={applying || loading || !stats || stats.n_ok === 0}
              className="px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-40">
              {applying ? "Converting…" : "Convert to date"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
