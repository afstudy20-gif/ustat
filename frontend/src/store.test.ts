import { describe, it, expect, beforeEach } from "vitest";
import { useStore } from "./store";

describe("renameInPanelCache", () => {
  beforeEach(() => {
    useStore.setState({ panelCache: {} });
  });

  it("remaps a plain string selection to the new column name", () => {
    useStore.getState().setPanelCache("charts", { x: "LDL", color: "SEX" });
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache.charts).toEqual({ x: "LDL_mgdl", color: "SEX" });
  });

  it("remaps the old name inside array selections without touching other entries", () => {
    useStore.getState().setPanelCache("iptw", { covariates: ["AGE", "LDL", "SEX"] });
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache.iptw).toEqual({ covariates: ["AGE", "LDL_mgdl", "SEX"] });
  });

  it("remaps the same old name across multiple panels in one call", () => {
    useStore.getState().setPanelCache("charts", { x: "LDL" });
    useStore.getState().setPanelCache("table1", { variables: ["LDL", "AGE"] });
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache.charts).toEqual({ x: "LDL_mgdl" });
    expect(useStore.getState().panelCache.table1).toEqual({ variables: ["LDL_mgdl", "AGE"] });
  });

  it("leaves values that don't match the old name untouched", () => {
    useStore.getState().setPanelCache("charts", { x: "AGE", bins: 20 });
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache.charts).toEqual({ x: "AGE", bins: 20 });
  });

  it("is a no-op on an empty panelCache", () => {
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache).toEqual({});
  });
});
