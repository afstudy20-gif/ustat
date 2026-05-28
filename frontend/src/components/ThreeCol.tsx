import React from "react";

/**
 * 3-column layout helper for analysis panels.
 *
 *  ┌─────────┬───────────────┬─────────┐
 *  │  left   │    middle     │  right  │
 *  │ controls│     chart     │ results │
 *  └─────────┴───────────────┴─────────┘
 *
 * - `left`   fixed 340px (controls + Run button)
 * - `middle` flexible (chart, plot, figure)
 * - `right`  fixed 380px (result text, tables, interpretation)
 *
 * Falls back to single column on narrow screens (< 1100px) so collapsible
 * cards still readable on tablets.
 */
export default function ThreeCol({
  left,
  middle,
  right,
}: {
  left: React.ReactNode;
  middle: React.ReactNode;
  right: React.ReactNode;
}) {
  return (
    <div className="flex flex-col xl:flex-row gap-3">
      <div className="xl:w-[340px] xl:flex-shrink-0 space-y-2">{left}</div>
      <div className="flex-1 min-w-0">{middle}</div>
      <div className="xl:w-[380px] xl:flex-shrink-0 space-y-2">{right}</div>
    </div>
  );
}
