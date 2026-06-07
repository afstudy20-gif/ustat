/** Shared model for a publication-styled table export. */
export interface StyledTableData {
  title?: string;
  caption?: string;
  columns: string[];
  rows: string[][];
  /** Base filename (no extension) for downloads. */
  filename?: string;
}

const esc = (s: unknown): string =>
  String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Build a self-contained HTML table with inline styles — pastes into Word /
 *  Google Docs keeping the header rule + borders, and works as a standalone
 *  .html file. Kept deliberately simple so Word's HTML importer is happy. */
export function buildStyledTableHtml(d: StyledTableData, opts: { fullDoc?: boolean } = {}): string {
  const th = d.columns
    .map((c, i) =>
      `<th style="text-align:${i === 0 ? "left" : "left"};padding:6px 10px;border-bottom:2px solid #333;font-weight:bold;">${esc(c)}</th>`,
    )
    .join("");
  const trs = d.rows
    .map(
      (row) =>
        `<tr>${d.columns
          .map(
            (_, i) =>
              `<td style="padding:6px 10px;border-bottom:1px solid #ccc;">${esc(row[i] ?? "")}</td>`,
          )
          .join("")}</tr>`,
    )
    .join("");

  const titleHtml = d.title
    ? `<p style="font-weight:bold;font-size:13px;margin:0 0 8px;">${esc(d.title)}</p>`
    : "";
  const captionHtml = d.caption
    ? `<p style="font-size:11px;color:#555;font-style:italic;margin:8px 0 0;">${esc(d.caption)}</p>`
    : "";
  const table =
    `${titleHtml}<table style="border-collapse:collapse;font-family:'Times New Roman',Georgia,serif;font-size:12px;">` +
    `<thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>${captionHtml}`;

  if (!opts.fullDoc) return table;
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${esc(d.title ?? "Table")}</title></head><body>${table}</body></html>`;
}

/** Tab-separated text fallback for plain-text clipboard targets. */
export function buildTsv(d: StyledTableData): string {
  const header = d.columns.join("\t");
  const body = d.rows.map((r) => d.columns.map((_, i) => r[i] ?? "").join("\t")).join("\n");
  return `${header}\n${body}`;
}

/** Copy the table to the clipboard as rich HTML (+ plain-text fallback) so it
 *  pastes styled into Word / Docs. Returns true on success. */
export async function copyStyledTable(d: StyledTableData): Promise<boolean> {
  const html = buildStyledTableHtml(d);
  const tsv = buildTsv(d);
  try {
    if (navigator.clipboard && "write" in navigator.clipboard && typeof ClipboardItem !== "undefined") {
      await navigator.clipboard.write([
        new ClipboardItem({
          "text/html": new Blob([html], { type: "text/html" }),
          "text/plain": new Blob([tsv], { type: "text/plain" }),
        }),
      ]);
      return true;
    }
    await navigator.clipboard.writeText(tsv);
    return true;
  } catch {
    return false;
  }
}

/** Download the table as a standalone styled .html file. */
export function downloadStyledHtml(d: StyledTableData): void {
  const html = buildStyledTableHtml(d, { fullDoc: true });
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${d.filename ?? "table"}.html`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}
