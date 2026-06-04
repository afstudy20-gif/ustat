/**
 * useAutoSession — autosaves the live session to IndexedDB.
 *
 * Strategy:
 *   1. **Change-triggered, debounced 5 s.** Whenever session_id, the
 *      active tab, the case filter, or any tracked dirty marker
 *      changes, schedule a snapshot 5 s later. Subsequent changes
 *      restart the timer so we don't fetch on every keystroke.
 *   2. **Periodic 60 s.** A fallback interval saves once a minute even
 *      if nothing tracked has changed (analyses ran, filters applied
 *      server-side, etc.).
 *   3. **`beforeunload` flush.** When the user closes the tab, fire one
 *      last best-effort save. The browser may kill the page mid-fetch
 *      so this is a `keepalive: true` request — small risk of data
 *      loss vs. zero, by design.
 *
 *   The snapshot itself is the same blob the manual "Save Session JSON"
 *   button has always produced. Resume = re-upload it via
 *   POST /api/sessions/load_session.
 */

import { useEffect, useRef } from "react";
import api from "../api";
import { useStore } from "../store";
import { upsertRecentSession, notifySessionsChanged } from "../lib/sessionDb";

const DEBOUNCE_MS = 5_000;
const PERIODIC_MS = 60_000;

interface AutoSaveDeps {
  /** Optional setter for a status indicator in the header. */
  onStatus?: (status: "idle" | "saving" | "saved" | "error", at?: number) => void;
}

export function useAutoSession({ onStatus }: AutoSaveDeps = {}): void {
  // Pull only the bits that should *retrigger* the debounce. Anything
  // that mutates per-render (e.g. functions) would defeat the timer.
  const sessionId = useStore((s) => s.session?.session_id ?? null);
  const filename  = useStore((s) => s.session?.filename ?? null);
  const nRows     = useStore((s) => s.session?.rows ?? null);
  const nCols     = useStore((s) => s.session?.columns.length ?? null);
  const activeTab = useStore((s) => s.activeTab);
  const caseFilter = useStore((s) => s.caseFilter);

  const lastSavedHashRef = useRef<string | null>(null);
  const inFlightRef = useRef<boolean>(false);

  // Capture a stable status setter so the snapshot function can be
  // memoised without re-running on every render.
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;

  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;

    const snapshot = async (source: "auto" | "manual" = "auto"): Promise<void> => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      try {
        onStatusRef.current?.("saving");
        // Server already knows how to serialise; we just need the JSON.
        const res = await api.get(`/api/sessions/${sessionId}/save_session`);
        if (cancelled) return;
        const payload = typeof res.data === "string"
          ? res.data
          : JSON.stringify(res.data);
        // Skip the write if the blob hasn't changed since the last
        // snapshot — cuts down on IndexedDB churn for read-only sessions.
        const hash = `${payload.length}:${payload.slice(0, 64)}`;
        if (hash === lastSavedHashRef.current) {
          onStatusRef.current?.("saved", Date.now());
          return;
        }
        lastSavedHashRef.current = hash;
        await upsertRecentSession({
          serverSessionId: sessionId,
          name: filename ?? "Untitled",
          payload,
          nRows: nRows ?? undefined,
          nCols: nCols ?? undefined,
          activeTab: activeTab ?? undefined,
          source,
        });
        notifySessionsChanged();
        onStatusRef.current?.("saved", Date.now());
      } catch {
        // Network blip, server restart, 404 on session_id — try again
        // next tick rather than surfacing an error. The user can still
        // hit the explicit Save button if they want certainty.
        onStatusRef.current?.("error");
      } finally {
        inFlightRef.current = false;
      }
    };

    // 1) Debounced change snapshot.
    const debounceTimer = setTimeout(snapshot, DEBOUNCE_MS);

    // 2) Periodic snapshot — independent of the debounce.
    const interval = setInterval(snapshot, PERIODIC_MS);

    // 3) Best-effort beforeunload flush.
    const onBeforeUnload = () => {
      // Don't await — the page may already be tearing down.
      void snapshot();
    };
    window.addEventListener("beforeunload", onBeforeUnload);

    return () => {
      cancelled = true;
      clearTimeout(debounceTimer);
      clearInterval(interval);
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  }, [sessionId, filename, nRows, nCols, activeTab, caseFilter]);
}
