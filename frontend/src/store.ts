import { create } from "zustand";

export interface ColMeta {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical" | "boolean" | "text";
}

export interface Session {
  session_id: string;
  filename: string;
  rows: number;
  columns: ColMeta[];
  preview: Record<string, unknown>[];
}

interface AppState {
  session: Session | null;
  activeTab: string;
  setSession: (s: Session) => void;
  setActiveTab: (t: string) => void;
  clearSession: () => void;
  // Column kind override (data tab kind badge)
  updateColumnKind: (name: string, kind: ColMeta["kind"]) => void;
  // Inline cell editing
  updatePreviewCell: (rowIdx: number, col: string, value: unknown) => void;
  // Computed columns (Compute tab)
  addSessionColumn: (col: ColMeta, previewValues: (number | string | null)[]) => void;
  removeSessionColumn: (name: string) => void;
  // Table 1 persistence across tab switches
  table1Result: any;
  setTable1Result: (r: any) => void;
  clearTable1: () => void;
}

export const useStore = create<AppState>((set) => ({
  session: null,
  activeTab: "data",
  table1Result: null,
  setSession: (s) => set({ session: s, activeTab: "data", table1Result: null }),
  setActiveTab: (t) => set({ activeTab: t }),
  clearSession: () => set({ session: null, activeTab: "data", table1Result: null }),
  updateColumnKind: (name, kind) =>
    set((state) => {
      if (!state.session) return state;
      return {
        session: {
          ...state.session,
          columns: state.session.columns.map((c) =>
            c.name === name ? { ...c, kind } : c
          ),
        },
      };
    }),
  updatePreviewCell: (rowIdx, col, value) =>
    set((state) => {
      if (!state.session) return state;
      const preview = [...state.session.preview];
      preview[rowIdx] = { ...preview[rowIdx], [col]: value };
      return { session: { ...state.session, preview } };
    }),
  addSessionColumn: (col, previewValues) =>
    set((state) => {
      if (!state.session) return state;
      // Replace existing column with same name, or append
      const columns = [
        ...state.session.columns.filter((c) => c.name !== col.name),
        col,
      ];
      const preview = state.session.preview.map((row, i) => ({
        ...row,
        [col.name]: previewValues[i] ?? null,
      }));
      return { session: { ...state.session, columns, preview } };
    }),
  removeSessionColumn: (name) =>
    set((state) => {
      if (!state.session) return state;
      const columns = state.session.columns.filter((c) => c.name !== name);
      const preview = state.session.preview.map((row) => {
        const r = { ...row };
        delete r[name];
        return r;
      });
      return { session: { ...state.session, columns, preview } };
    }),
  setTable1Result: (r) => set({ table1Result: r }),
  clearTable1: () => set({ table1Result: null }),
}));
