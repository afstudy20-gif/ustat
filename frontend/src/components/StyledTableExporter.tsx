import { useState } from "react";
import { exportStyledTable } from "../api";
import { downloadBlob } from "../lib/tiffEncoder";
import {
  copyStyledTable,
  downloadStyledHtml,
  type StyledTableData,
} from "../lib/styledTable";

/** Compact export bar for a publication table: Copy (rich HTML → Word/Docs),
 *  Word .docx (server-rendered, styled), and standalone .html. The data is
 *  supplied lazily so callers can build columns/rows only when an action runs. */
export function StyledTableExporter({ data }: { data: () => StyledTableData }) {
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);

  const handleCopy = async () => {
    const ok = await copyStyledTable(data());
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    }
  };

  const handleDocx = async () => {
    setBusy(true);
    try {
      const d = data();
      const res = await exportStyledTable({
        title: d.title, caption: d.caption, columns: d.columns, rows: d.rows,
        filename: d.filename ?? "table",
      });
      downloadBlob(res.data as Blob, `${d.filename ?? "table"}.docx`);
    } catch {
      /* surfaced by the disabled state resetting; server error is rare */
    } finally {
      setBusy(false);
    }
  };

  const btn = "text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors disabled:opacity-40";

  return (
    <div className="flex items-center gap-1.5">
      <button onClick={handleCopy} className={btn} title="Copy as a styled table (paste into Word / Google Docs)">
        {copied ? "Copied ✓" : "Copy table"}
      </button>
      <button onClick={handleDocx} disabled={busy} className={btn} title="Download as a styled Word document">
        {busy ? "…" : "Word"}
      </button>
      <button onClick={() => downloadStyledHtml(data())} className={btn} title="Download as a styled HTML file">
        HTML
      </button>
    </div>
  );
}
