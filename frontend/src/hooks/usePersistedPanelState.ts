import { useState } from "react";
import { useStore } from "../store";

/**
 * usePersistedPanelState — a useState replacement that mirrors its value into
 * the Zustand `panelCache` so it survives a panel unmount/remount (e.g. the
 * user switches to the Data tab and back). The cached value is read once on
 * mount; if present it overrides `initialValue`.
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
): [T, (v: T) => void] {
  const setPanelCache = useStore((s) => s.setPanelCache);

  // Read the cached value exactly once (on mount). Reading inside the lazy
  // initialiser avoids re-subscribing the component to panelCache mutations,
  // which would defeat the "set once, mirror out" contract and risk loops.
  const [value, setValue] = useState<T>(() => {
    const cached = useStore.getState().panelCache[panelId];
    return cached && key in cached ? (cached[key] as T) : initialValue;
  });

  const update = (next: T) => {
    setValue(next);
    const cur = useStore.getState().panelCache[panelId] ?? {};
    setPanelCache(panelId, { ...cur, [key]: next });
  };

  return [value, update];
}
