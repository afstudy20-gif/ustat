import { useState, useEffect } from "react";
import { useStore, type ColMeta } from "../store";
import { saveMetadata } from "../api";

const ROLES = ["", "outcome", "predictor", "covariate", "id", "time", "event"] as const;
const ROLE_COLORS: Record<string, string> = {
  outcome: "bg-red-50 text-red-700",
  predictor: "bg-blue-50 text-blue-700",
  covariate: "bg-purple-50 text-purple-700",
  id: "bg-gray-100 text-gray-500",
  time: "bg-amber-50 text-amber-700",
  event: "bg-green-50 text-green-700",
};

// Auto-detect roles from column names
function autoDetectRole(name: string): string {
  const n = name.toLowerCase();
  if (/^(id|patient|subject|case)/.test(n)) return "id";
  if (/^(time|duration|fupt|follow|days|months|years)/.test(n)) return "time";
  if (/^(event|death|exitus|status|censor|outcome|endpoint)/.test(n)) return "event";
  return "";
}

export default function DataDictionaryPanel() {
  const session = useStore((s) => s.session);
  const setSession = useStore((s) => s.setSession);
  if (!session) return null;

  const [meta, setMeta] = useState<Record<string, Partial<ColMeta>>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Initialize meta from session columns
  useEffect(() => {
    const m: Record<string, Partial<ColMeta>> = {};
    for (const col of session.columns) {
      m[col.name] = {
        label: col.label ?? "",
        description: col.description ?? "",
        units: col.units ?? "",
        role: col.role ?? (autoDetectRole(col.name) as ColMeta["role"]),
        value_labels: col.value_labels ?? {},
      };
    }
    setMeta(m);
  }, [session.session_id]);

  const update = (colName: string, field: string, value: any) => {
    setMeta((prev) => ({ ...prev, [colName]: { ...prev[colName], [field]: value } }));
    setSaved(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveMetadata(session.session_id, meta);
      // Update local session columns with the metadata
      const updatedCols = session.columns.map((c) => ({
        ...c,
        ...(meta[c.name] ?? {}),
      }));
      setSession({ ...session, columns: updatedCols });
      setSaved(true);
    } catch {
      /* ignore */
    } finally {
      setSaving(false);
    }
  };

  const handleAutoDetect = () => {
    const updated = { ...meta };
    for (const col of session.columns) {
      const detected = autoDetectRole(col.name);
      if (detected && !updated[col.name]?.role) {
        updated[col.name] = { ...updated[col.name], role: detected as ColMeta["role"] };
      }
    }
    setMeta(updated);
    setSaved(false);
  };

  const exportCSV = () => {
    const rows = [["Name", "Label", "Type", "Units", "Role", "Description"]];
    for (const col of session.columns) {
      const m = meta[col.name] ?? {};
      rows.push([col.name, m.label ?? "", col.kind, m.units ?? "", m.role ?? "", m.description ?? ""]);
    }
    const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\r\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "data_dictionary.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-gray-900">Data Dictionary</h2>
          <p className="text-xs text-gray-400">{session.columns.length} variables \u00B7 {session.rows} observations</p>
        </div>
        <div className="flex gap-2">
          <button onClick={handleAutoDetect} className="text-xs px-3 py-1.5 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50">
            Auto-detect roles
          </button>
          <button onClick={exportCSV} className="text-xs px-3 py-1.5 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50">
            Export CSV
          </button>
          <button onClick={handleSave} disabled={saving} className="btn-primary text-xs px-4 py-1.5">
            {saving ? "Saving\u2026" : saved ? "\u2713 Saved" : "Save Metadata"}
          </button>
        </div>
      </div>

      {/* Dictionary table */}
      <div className="overflow-auto rounded-xl border border-gray-200">
        <table className="w-full text-xs">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-40">Variable</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-12">Type</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-48">Label</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-20">Units</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-28">Role</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium">Description</th>
            </tr>
          </thead>
          <tbody>
            {session.columns.map((col) => {
              const m = meta[col.name] ?? {};
              return (
                <tr key={col.name} className="border-t border-gray-100 hover:bg-gray-50">
                  <td className="px-3 py-1.5 font-mono font-medium text-gray-800">{col.name}</td>
                  <td className="px-3 py-1.5">
                    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                      col.kind === "numeric" ? "bg-blue-100 text-blue-700" :
                      col.kind === "categorical" ? "bg-orange-100 text-orange-700" :
                      col.kind === "date" ? "bg-purple-100 text-purple-700" :
                      "bg-gray-100 text-gray-500"
                    }`}>{col.kind === "numeric" ? "num" : col.kind === "categorical" ? "cat" : col.kind === "date" ? "date" : "txt"}</span>
                  </td>
                  <td className="px-1 py-1">
                    <input className="w-full bg-transparent border border-transparent hover:border-gray-200 focus:border-indigo-400 rounded px-2 py-0.5 text-xs focus:outline-none"
                      value={m.label ?? ""} placeholder="Variable label\u2026"
                      onChange={(e) => update(col.name, "label", e.target.value)} />
                  </td>
                  <td className="px-1 py-1">
                    <input className="w-full bg-transparent border border-transparent hover:border-gray-200 focus:border-indigo-400 rounded px-2 py-0.5 text-xs focus:outline-none"
                      value={m.units ?? ""} placeholder="e.g. mmHg"
                      onChange={(e) => update(col.name, "units", e.target.value)} />
                  </td>
                  <td className="px-1 py-1">
                    <select className={`w-full text-xs rounded px-1 py-0.5 border border-transparent hover:border-gray-200 focus:border-indigo-400 focus:outline-none ${ROLE_COLORS[m.role ?? ""] ?? ""}`}
                      value={m.role ?? ""} onChange={(e) => update(col.name, "role", e.target.value)}>
                      {ROLES.map((r) => <option key={r} value={r}>{r || "\u2014"}</option>)}
                    </select>
                  </td>
                  <td className="px-1 py-1">
                    <input className="w-full bg-transparent border border-transparent hover:border-gray-200 focus:border-indigo-400 rounded px-2 py-0.5 text-xs focus:outline-none"
                      value={m.description ?? ""} placeholder="Description\u2026"
                      onChange={(e) => update(col.name, "description", e.target.value)} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Role legend */}
      <div className="flex gap-3 text-[10px] text-gray-400">
        {Object.entries(ROLE_COLORS).map(([role, cls]) => (
          <span key={role} className={`px-1.5 py-0.5 rounded ${cls}`}>{role}</span>
        ))}
      </div>
    </div>
  );
}
