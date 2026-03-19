import { useCallback, useState } from "react";
import { Upload } from "lucide-react";
import { uploadFile } from "../api";
import { useStore } from "../store";

export default function UploadZone() {
  const setSession = useStore((s) => s.setSession);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handle = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const res = await uploadFile(file);
      setSession(res.data);
    } catch (e: any) {
      const detail = e.response?.data?.detail;
      const status = e.response?.status;
      const msg = detail
        ? `${detail}`
        : e.message?.includes("Network")
        ? "Cannot connect to backend (localhost:8000). Is it running?"
        : `Upload failed (${status ?? e.message ?? "unknown error"})`;
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [setSession]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-8 p-8 bg-gray-50">
      <div className="flex items-center gap-3">
        <img src="/logo.png" alt="uStat logo" className="w-10 h-10 rounded-xl object-contain" />
        <div>
          <h1 className="text-2xl font-bold text-gray-900 leading-tight">uStat</h1>
          <p className="text-xs text-gray-400 leading-none">Statistical Analysis Platform</p>
        </div>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`w-full max-w-lg border-2 border-dashed rounded-2xl p-16 flex flex-col items-center gap-4 transition-colors cursor-pointer
          ${dragging ? "border-indigo-500 bg-indigo-50" : "border-gray-300 hover:border-gray-400 bg-white"}`}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        <Upload size={40} className="text-gray-400" />
        <div className="text-center">
          <p className="text-gray-700 font-medium">Drop your data file here</p>
          <p className="text-gray-400 text-sm mt-1">or click to browse</p>
          <p className="text-gray-300 text-xs mt-3">CSV · Excel · SAS · SPSS · Stata</p>
        </div>
        <input
          id="file-input"
          type="file"
          className="hidden"
          accept=".csv,.xlsx,.xls,.sas7bdat,.sav,.dta"
          onChange={(e) => e.target.files?.[0] && handle(e.target.files[0])}
        />
      </div>

      {loading && <p className="text-indigo-600 animate-pulse">Uploading and parsing…</p>}
      {error && <p className="text-red-500 text-sm">{error}</p>}

      <p className="text-gray-400 text-xs text-center max-w-sm">
        Data stays on your local machine — everything runs through your own backend.
      </p>
    </div>
  );
}
