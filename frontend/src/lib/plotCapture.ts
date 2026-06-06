import type { RefObject } from "react";

export interface PlotCaptureHooks {
  beforeCapture?: () => void | Promise<void>;
  afterCapture?: () => void | Promise<void>;
}

const CAPTURE_HOOKS = "__wizPlotCaptureHooks" as const;

interface CaptureTarget {
  el?: unknown;
  elRef?: { current?: unknown };
  [CAPTURE_HOOKS]?: PlotCaptureHooks;
}

function asCaptureTarget(value: unknown): CaptureTarget | null {
  if ((typeof value !== "object" || value === null) && typeof value !== "function") {
    return null;
  }
  return value as CaptureTarget;
}

function captureTargets(plotRef: RefObject<unknown>): CaptureTarget[] {
  const ref = asCaptureTarget(plotRef.current);
  if (!ref) return [];
  return [ref, asCaptureTarget(ref.el), asCaptureTarget(ref.elRef?.current)]
    .filter((target): target is CaptureTarget => target !== null);
}

export function registerPlotCaptureHooks(
  plotRef: RefObject<unknown>,
  hooks: PlotCaptureHooks,
): () => void {
  const targets = captureTargets(plotRef);
  for (const target of targets) target[CAPTURE_HOOKS] = hooks;

  return () => {
    for (const target of targets) {
      if (target[CAPTURE_HOOKS] === hooks) delete target[CAPTURE_HOOKS];
    }
  };
}

export async function withRegisteredPlotCapture<T>(
  plotRef: RefObject<unknown>,
  capture: () => Promise<T>,
): Promise<T> {
  const targets = captureTargets(plotRef);
  const hooks = targets
    .map((target) => target[CAPTURE_HOOKS] as PlotCaptureHooks | undefined)
    .find(Boolean);

  await hooks?.beforeCapture?.();
  try {
    return await capture();
  } finally {
    await hooks?.afterCapture?.();
  }
}
