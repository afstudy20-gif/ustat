import React from "react";
import { useResizableRightCol } from "../hooks/useResizableRightCol";

/**
 * 3-column layout helper for analysis panels.
 *
 *  ┌─────────┬───────────────┬─────────┐
 *  │  left   │    middle     │  right  │
 *  │ controls│     chart     │ results │
 *  └─────────┴───────────────┴─────────┘
 *
 * - `left`   resizable, default 340 px (controls + Run button)
 * - `middle` flexible (chart, plot, figure)
 * - `right`  resizable, default 380 px (result text, tables, interpretation)
 *
 * Both column widths can be dragged via the dividers between them. Widths
 * persist per `storageKey` in localStorage so the user's preferred layout
 * survives reloads.
 *
 * Falls back to a single column on narrow screens (< xl: 1280 px).
 */
export default function ThreeCol({
  left,
  middle,
  right,
  storageKey = "ThreeCol",
}: {
  left: React.ReactNode;
  middle: React.ReactNode;
  right: React.ReactNode;
  storageKey?: string;
}) {
  const leftCol = useResizableRightCol(`${storageKey}.left`, 340, 200, 560, "left");
  const rightCol = useResizableRightCol(`${storageKey}.right`, 380, 240, 720, "right");

  // Width applied via CSS variables; Tailwind `xl:w-[var(--…)]` activates only
  // above the xl breakpoint so mobile / tablet keeps the stacked layout.
  const styleVars = {
    ["--left-col" as any]: `${leftCol.w}px`,
    ["--right-col" as any]: `${rightCol.w}px`,
  } as React.CSSProperties;

  return (
    <div className="flex flex-col xl:flex-row gap-3 items-stretch" style={styleVars}>
      <div className="space-y-2 xl:flex-shrink-0 xl:w-[var(--left-col)]">{left}</div>
      <div
        role="separator"
        aria-orientation="vertical"
        title="Sürükle: sol sütun genişliği · Çift tık: sıfırla"
        onPointerDown={leftCol.onDragStart}
        onDoubleClick={leftCol.onReset}
        className="hidden xl:block w-1.5 rounded-full bg-gray-300/60 hover:bg-indigo-400/80 cursor-col-resize self-stretch transition-colors"
      />
      <div className="flex-1 min-w-0">{middle}</div>
      <div
        role="separator"
        aria-orientation="vertical"
        title="Sürükle: sağ sütun genişliği · Çift tık: sıfırla"
        onPointerDown={rightCol.onDragStart}
        onDoubleClick={rightCol.onReset}
        className="hidden xl:block w-1.5 rounded-full bg-gray-300/60 hover:bg-indigo-400/80 cursor-col-resize self-stretch transition-colors"
      />
      <div className="space-y-2 xl:flex-shrink-0 xl:w-[var(--right-col)]">{right}</div>
    </div>
  );
}
