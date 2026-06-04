/**
 * RecentSessionsPanel — lists previously-saved sessions on the upload
 * landing page so the user can pick up where they left off without
 * re-uploading their dataset.
 *
 * Each card carries: dataset name, n × m, last-visited tab, save
 * timestamp, an indicator of auto vs. manual save, and two actions —
 * "Devam et" (re-upload to the backend → setSession + restore tab) and
 * "Sil" (purge the local snapshot).
 *
 * Everything renders from IndexedDB; the backend is contacted only on
 * Devam et and on the explicit Save button elsewhere in the app.
 */

import { useCallback, useEffect, useState } from "react";
import { Clock, Database, RotateCcw, Trash2, Sparkles, FileText, HardDrive } from "lucide-react";
import api from "../api";
import { useStore } from "../store";
import {
  listRecentSessions,
  deleteRecentSession,
  getRecentSession,
  subscribeSessions,
  getStorageEstimate,
  clearAllRecentSessions,
  type RecentSessionMeta,
} from "../lib/sessionDb";

function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function fmtAgo(epochMs: number): string {
  const diff = Date.now() - epochMs;
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return "az önce";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} dk önce`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} sa önce`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day} gün önce`;
  const d = new Date(epochMs);
  return d.toLocaleDateString();
}

const TAB_LABELS: Record<string, string> = {
  data: "Data",
  summary: "Summary",
  table: "Table 1",
  tests: "Tests",
  correlation: "Correlation",
  roc: "ROC",
  models: "Models",
  psm: "PSM",
  iptw: "IPTW",
  dca: "DCA",
  meta: "Meta",
  missing: "Missing",
  visual: "Visual",
  compute: "Compute",
  causal: "Causal",
  code: "Code",
};

export default function RecentSessionsPanel() {
  const setSession = useStore((s) => s.setSession);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const [items, setItems] = useState<RecentSessionMeta[]>([]);
  const [estimate, setEstimate] = useState<{ count: number; bytes: number; capCount: number; capBytes: number } | null>(null);
  const [restoring, setRestoring] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const reload = useCallback(async () => {
    try {
      const [list, est] = await Promise.all([listRecentSessions(), getStorageEstimate()]);
      setItems(list);
      setEstimate(est);
    } catch {
      // IndexedDB unavailable (Safari private mode etc.) — silently
      // degrade; the upload zone still works.
      setItems([]);
      setEstimate(null);
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void reload();
    const unsub = subscribeSessions(() => { void reload(); });
    return unsub;
  }, [reload]);

  if (loaded && items.length === 0) return null;

  const onRestore = async (id: string) => {
    setRestoring(id);
    setError(null);
    try {
      const rec = await getRecentSession(id);
      if (!rec) throw new Error("Snapshot not found");
      // POST /api/sessions/load_session expects multipart File; wrap
      // the stored payload as a Blob so the existing endpoint accepts it.
      const blob = new Blob([rec.payload], { type: "application/json" });
      const form = new FormData();
      form.append("file", blob, `${rec.name || "session"}.json`);
      const res = await api.post("/api/sessions/load_session", form);
      setSession(res.data);
      // Restore the user's last tab, falling back to Data.
      if (rec.activeTab) setActiveTab(rec.activeTab);
      // Re-hydrate column-decimal overrides the same way UploadZone
      // does on a fresh load — keeps the data table formatting stable.
      try {
        const dres = await api.get(`/api/sessions/${res.data.session_id}/decimals`);
        if (dres.data && Object.keys(dres.data).length > 0) {
          const { useStore: store } = await import("../store");
          store.setState({ columnDecimals: dres.data });
        }
      } catch { /* non-fatal */ }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Snapshot yüklenemedi");
    } finally {
      setRestoring(null);
    }
  };

  const onDelete = async (id: string) => {
    if (!window.confirm("Bu kayıt silinsin mi? Bu işlem geri alınamaz.")) return;
    await deleteRecentSession(id);
    void reload();
  };

  const onClearAll = async () => {
    if (!window.confirm("Tüm yerel oturum kayıtları silinsin mi? Bu işlem geri alınamaz.")) return;
    await clearAllRecentSessions();
    void reload();
  };

  if (!loaded) return null;

  return (
    <div className="w-full max-w-3xl mt-2">
      <div className="flex items-center justify-between mb-2 px-1">
        <div className="flex items-center gap-1.5">
          <Clock size={14} className="text-indigo-500" />
          <h3 className="text-xs font-semibold text-gray-700">Son Çalışmalar</h3>
          <span className="text-[10px] text-gray-400 font-normal">
            (otomatik olarak tarayıcınızda saklanır — sunucuya gönderilmez)
          </span>
        </div>
        {estimate && (
          <div className="flex items-center gap-2 text-[10px] text-gray-400">
            <HardDrive size={11} />
            <span>{estimate.count}/{estimate.capCount} · {fmtBytes(estimate.bytes)}</span>
            <button
              onClick={onClearAll}
              className="text-gray-400 hover:text-red-500 underline-offset-2 hover:underline"
            >
              Tümünü temizle
            </button>
          </div>
        )}
      </div>

      {error && (
        <p className="text-[11px] text-red-500 mb-2 px-1">{error}</p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {items.map((it) => (
          <div
            key={it.id}
            className="bg-white border border-gray-200 rounded-xl p-3 shadow-sm hover:shadow-md hover:border-indigo-300 transition-all group"
          >
            <div className="flex items-start justify-between mb-2 gap-2">
              <div className="flex items-center gap-1.5 min-w-0 flex-1">
                <FileText size={13} className="text-indigo-500 flex-shrink-0" />
                <span
                  className="text-xs font-semibold text-gray-800 truncate"
                  title={it.name}
                >
                  {it.name}
                </span>
              </div>
              {it.source === "auto" && (
                <span
                  className="text-[8px] uppercase tracking-wide bg-indigo-50 text-indigo-600 px-1.5 py-0.5 rounded font-bold flex-shrink-0"
                  title="Otomatik kayıt"
                >
                  <Sparkles size={9} className="inline mr-0.5" />Auto
                </span>
              )}
            </div>

            <div className="flex items-center gap-2 text-[10px] text-gray-500 mb-2">
              {(it.nRows != null && it.nCols != null) && (
                <span className="flex items-center gap-0.5">
                  <Database size={10} />
                  {it.nRows.toLocaleString()} × {it.nCols}
                </span>
              )}
              <span className="text-gray-300">·</span>
              <span>{fmtBytes(it.sizeBytes)}</span>
              <span className="text-gray-300">·</span>
              <span title={new Date(it.savedAt).toLocaleString()}>{fmtAgo(it.savedAt)}</span>
            </div>

            {it.activeTab && (
              <p className="text-[10px] text-gray-400 mb-2.5">
                Kaldığı yer:{" "}
                <span className="font-semibold text-gray-600">
                  {TAB_LABELS[it.activeTab] ?? it.activeTab}
                </span>
              </p>
            )}

            <div className="flex items-center gap-1.5">
              <button
                onClick={() => onRestore(it.id)}
                disabled={restoring === it.id}
                className="flex-1 flex items-center justify-center gap-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-[11px] font-semibold px-2.5 py-1.5 rounded-lg transition-colors"
              >
                <RotateCcw size={11} />
                {restoring === it.id ? "Yükleniyor…" : "Devam et"}
              </button>
              <button
                onClick={() => onDelete(it.id)}
                className="text-gray-400 hover:text-red-600 hover:bg-red-50 px-2 py-1.5 rounded-lg transition-colors"
                title="Bu kaydı sil"
              >
                <Trash2 size={12} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
