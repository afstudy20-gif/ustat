import { useState } from "react";
import api, { runDropMissing, runCleanOutliers, runFindReplace } from "../api";
import { useStore } from "../store";
import type { ColMeta } from "../store";

// Data cleaning / row-filtering tools: listwise deletion of missing rows,
// IQR / Z-score outlier removal, and find-&-replace. Lives in the Missing-data
// tab (moved out of Compute). Mutates the session in place via the store.

export function CleaningTab({
  sessionId,
  columns,
  numCols,
}: {
  sessionId: string;
  columns: ColMeta[];
  numCols: ColMeta[];
}) {
  const [cleanMode, setCleanMode] = useState<"missing" | "outliers" | "find_replace">("missing");
  const [selectedCols, setSelectedCols] = useState<string[]>([]);
  
  // Outliers variables
  const [outlierMethod, setOutlierMethod] = useState("iqr");
  const [outlierThreshold, setOutlierThreshold] = useState(1.5);
  
  // Find & Replace variables
  const [findValue, setFindValue] = useState("");
  const [replaceValue, setReplaceValue] = useState("");
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const refreshSession = async () => {
    const refresh = await api.get(`/api/stats/${sessionId}/refresh`);
    const session = useStore.getState().session;
    if (session) {
      useStore.getState().setSession({ ...session, ...refresh.data });
    }
  };

  const handleClean = async () => {
    if (selectedCols.length === 0) {
      setError("Please select at least one column.");
      return;
    }
    setLoading(true); setError(null); setSuccess(null);
    try {
      if (cleanMode === "missing") {
        const res = await runDropMissing(sessionId, { columns: selectedCols });
        await refreshSession();
        setSuccess(`Success! Dropped ${res.data.deleted} rows with missing values. ${res.data.remaining_rows} rows remaining.`);
      } else if (cleanMode === "outliers") {
        const res = await runCleanOutliers(sessionId, {
          columns: selectedCols,
          method: outlierMethod,
          threshold: outlierThreshold
        });
        await refreshSession();
        setSuccess(`Success! Removed ${res.data.deleted} outlier rows. ${res.data.remaining_rows} rows remaining.`);
      } else if (cleanMode === "find_replace") {
        const res = await runFindReplace(sessionId, {
          columns: selectedCols,
          find_value: findValue,
          replace_value: replaceValue
        });
        await refreshSession();
        setSuccess(`Success! Replaced ${res.data.replaced_count} cell values across selected columns.`);
      }
    } catch (e: unknown) {
      setError((e as { response?: { data?: { detail?: string } } }).response?.data?.detail ?? "Data cleaning failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-800">Data Cleaning</h3>
        <p className="text-[11px] text-gray-400 mt-0.5">Listwise deletion · outlier removal · find &amp; replace</p>
      </div>
      <div className="px-5 py-4 space-y-4">
        <div className="space-y-1">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Cleaning Operation</label>
          <div className="flex gap-2">
            {[
              ["missing", "Drop Missing (Listwise)"],
              ["outliers", "Clean Outliers"],
              ["find_replace", "Find & Replace"]
            ].map(([id, label]) => (
              <label key={id} className="flex items-center gap-1 text-xs cursor-pointer text-gray-700 bg-gray-50 hover:bg-gray-100 border rounded-lg px-2.5 py-1">
                <input
                  type="radio"
                  name="clean_mode"
                  checked={cleanMode === id}
                  onChange={() => {
                    setCleanMode(id as "missing" | "outliers" | "find_replace");
                    setSelectedCols([]);
                    setError(null);
                    setSuccess(null);
                  }}
                  className="accent-indigo-600"
                />
                {label}
              </label>
            ))}
          </div>
        </div>

        {/* Variables Selection */}
        <div className="space-y-1">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Select Variables
          </label>
          <div className="max-h-40 overflow-y-auto rounded-lg border border-gray-200 divide-y divide-gray-50">
            {(cleanMode === "outliers" ? numCols : columns).map((c) => {
              const checked = selectedCols.includes(c.name);
              return (
                <label key={c.name} className={`flex items-center gap-2 px-2.5 py-1 text-xs cursor-pointer ${checked ? "bg-indigo-50/50" : "hover:bg-gray-50"}`}>
                  <input type="checkbox" checked={checked}
                    onChange={() => setSelectedCols((p) => p.includes(c.name) ? p.filter((x) => x !== c.name) : [...p, c.name])}
                    className="accent-indigo-500" />
                  <span className="text-gray-700 truncate flex-1">{c.name}</span>
                  <span className="text-[9px] text-gray-400">{c.kind}</span>
                </label>
              );
            })}
          </div>
          <p className="text-[10px] text-gray-400">{selectedCols.length} selected</p>
        </div>

        {/* Dynamic controls based on operation */}
        {cleanMode === "outliers" && (
          <div className="grid grid-cols-2 gap-3 bg-gray-50 border rounded-xl p-3">
            <div className="space-y-1">
              <label className="text-xs text-gray-500">Method</label>
              <select
                className="select text-xs w-full"
                value={outlierMethod}
                onChange={(e) => setOutlierMethod(e.target.value)}
              >
                <option value="iqr">IQR (Interquartile Range)</option>
                <option value="zscore">Z-score</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-gray-500">
                {outlierMethod === "iqr" ? "Threshold (IQR multiplier)" : "Threshold (Z score)"}
              </label>
              <input
                type="number"
                step="0.1"
                className="select text-xs w-full"
                value={outlierThreshold}
                onChange={(e) => setOutlierThreshold(parseFloat(e.target.value) || 1.5)}
              />
            </div>
          </div>
        )}

        {cleanMode === "find_replace" && (
          <div className="grid grid-cols-2 gap-3 bg-gray-50 border rounded-xl p-3">
            <div className="space-y-1">
              <label className="text-xs text-gray-500">Find value</label>
              <input
                type="text"
                placeholder="e.g. 999 or 'N/A'"
                className="select text-xs w-full font-mono"
                value={findValue}
                onChange={(e) => setFindValue(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-gray-500">Replace value</label>
              <input
                type="text"
                placeholder="e.g. 1 or leave blank for missing"
                className="select text-xs w-full font-mono"
                value={replaceValue}
                onChange={(e) => setReplaceValue(e.target.value)}
              />
            </div>
          </div>
        )}

        <button
          className="btn-primary w-full py-1.5"
          onClick={handleClean}
          disabled={loading || selectedCols.length === 0}
        >
          {loading ? "Processing..." : "Apply Cleaning"}
        </button>

        {error && <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-700">{error}</div>}
        {success && <div className="bg-green-50 border border-green-200 rounded-lg px-3 py-2 text-xs text-green-800">{success}</div>}
      </div>
    </div>
  );
}
