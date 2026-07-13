import { useEffect, useState, type Dispatch, type SetStateAction } from "react";
import { useStore } from "../store";

/**
 * usePersistedPanelState — a drop-in `useState` replacement that mirrors its
 * value into the Zustand `panelCache` so it survives a panel unmount/remount
 * (e.g. the user switches to the Data tab and back). The cached value is read
 * once on mount; if present it overrides `initialValue`.
 *
 * The returned setter has the exact `useState` signature, including the
 * functional-updater form `setX(prev => next)`, so existing call sites work
 * unchanged.
 *
 * panelCache is in-memory only and is cleared when the session changes, so
 * selections never leak across datasets.
 *
 * @param panelId  unique panel id, e.g. "cox", "models", "descriptive_numeric"
 * @param key      field name within that panel's cache object, e.g. "duration"
 * @param initialValue default when nothing is cached yet
 */
export function usePersistedPanelState<T>(
  panelId: string,
  key: string,
  initialValue: T,
): [T, Dispatch<SetStateAction<T>>] {
  // Read the cached value exactly once (on mount). Reading inside the lazy
  // initialiser avoids subscribing the component to panelCache mutations,
  // which would defeat the "set once, mirror out" contract and risk loops.
  const [value, setValue] = useState<T>(() => {
    const rawCached = useStore.getState().panelCache[panelId];
    const cached = rawCached && typeof rawCached === "object" ? rawCached as Record<string, unknown> : undefined;
    return cached && key in cached ? (cached[key] as T) : initialValue;
  });

  // Mirror to panelCache whenever the value changes (idempotent when equal).
  useEffect(() => {
    const rawCur = useStore.getState().panelCache[panelId];
    const cur = rawCur && typeof rawCur === "object" ? rawCur as Record<string, unknown> : {};
    if (cur[key] !== value) {
      useStore.getState().setPanelCache(panelId, { ...cur, [key]: value });
    }
  }, [panelId, key, value]);

  return [value, setValue];
}
