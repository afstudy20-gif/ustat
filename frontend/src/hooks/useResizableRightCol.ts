import { useState, useRef, useEffect, useCallback } from "react";

/**
 * Drag-to-resize right column hook for the 3-column analysis panels.
 *
 * - Returns the current right-column width (px) and an `onDragStart`
 *   pointer-down handler for the divider element.
 * - Persists the width per `key` in localStorage so the user's
 *   preferred layout survives reloads / tab switches.
 * - Clamps to [min, max] (defaults: 240 – 960 px).
 */
export function useResizableRightCol(
  key: string,
  initial = 480,
  min = 240,
  max = 960,
  /**
   * Which side the column being sized lives on.
   * - "right" (default): divider sits to the LEFT of the column.
   *   Dragging LEFT grows the column.
   * - "left": divider sits to the RIGHT of the column.
   *   Dragging RIGHT grows the column.
   */
  side: "left" | "right" = "right",
) {
  const storageKey = `uStat.colW.${key}`;

  const [w, setW] = useState<number>(() => {
    if (typeof window === "undefined") return initial;
    try {
      const raw = window.localStorage.getItem(storageKey);
      const n = raw ? parseInt(raw, 10) : NaN;
      if (Number.isFinite(n) && n >= min && n <= max) return n;
    } catch {
      /* localStorage may throw under privacy mode — fall back to default */
    }
    return initial;
  });

  const dragRef = useRef<{ startX: number; startW: number } | null>(null);

  // Persist width changes (debounced indirectly by React batching).
  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, String(w));
    } catch {
      /* ignore */
    }
  }, [storageKey, w]);

  const onMove = useCallback((e: PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    // For a column on the RIGHT side of the layout: dragging the divider
    // LEFT grows the column → newW = startW − dx.
    // For a column on the LEFT side of the layout: dragging the divider
    // RIGHT grows the column → newW = startW + dx.
    const dx = e.clientX - d.startX;
    const next = side === "right"
      ? Math.max(min, Math.min(max, d.startW - dx))
      : Math.max(min, Math.min(max, d.startW + dx));
    setW(next);
  }, [min, max, side]);

  const onUp = useCallback(() => {
    dragRef.current = null;
    document.removeEventListener("pointermove", onMove);
    document.removeEventListener("pointerup", onUp);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, [onMove]);

  const onDragStart = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startW: w };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  }, [w, onMove, onUp]);

  // Double-click resets to default width.
  const onReset = useCallback(() => setW(initial), [initial]);

  return { w, onDragStart, onReset };
}
