import { useState } from "react";
import { RefreshCw } from "lucide-react";

/** Clear this origin's caches + service-worker registrations, then
 *  hard-reload with a cache-bust query string. Origin-scoped — never
 *  touches other sites' data. Session state is held server-side, not in
 *  localStorage, so a refresh on the splash screen has no data side
 *  effects; after a dataset is open, the user is asked first.
 *
 *  Mirrors the pattern used by the sister Notepad app
 *  (see `not.drtr.uk` source) so behaviour is consistent across the
 *  drtr.uk suite.
 */
export async function refreshAppCaches(): Promise<void> {
  try {
    if ("serviceWorker" in navigator) {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map((r) => r.unregister().catch(() => null)));
    }
    if (typeof window !== "undefined" && window.caches) {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => caches.delete(k).catch(() => null)));
    }
  } catch (err) {
    // Best-effort — even if cache eviction fails we still hard-reload.
    console.warn("[refresh] cache eviction failed:", err);
  }
  const url = new URL(location.href);
  url.searchParams.set("_r", Date.now().toString(36));
  location.replace(url.toString());
}

interface RefreshAppButtonProps {
  /** When true, ask before reloading (used in-app where a dataset may be open). */
  confirmBeforeReload?: boolean;
  /** Visual variant: 'icon' for header (compact), 'inline' for splash (text + icon). */
  variant?: "icon" | "inline";
  /** Idle label for the 'inline' variant (e.g. "Refresh app", "Update app"). */
  label?: string;
  className?: string;
}

export default function RefreshAppButton({
  confirmBeforeReload = false,
  variant = "icon",
  label = "Refresh app",
  className = "",
}: RefreshAppButtonProps) {
  const [spinning, setSpinning] = useState(false);

  const handleClick = async () => {
    if (spinning) return;
    if (confirmBeforeReload) {
      const ok = window.confirm(
        "Refresh this app and clear its cached files?\n\n" +
        "Any work that has not been saved (Save Session) will be lost. " +
        "Server-side session data is unaffected and clears on its own 30 minutes after inactivity. " +
        "Other websites are not touched.",
      );
      if (!ok) return;
    }
    setSpinning(true);
    await refreshAppCaches();
  };

  if (variant === "inline") {
    return (
      <button
        onClick={handleClick}
        disabled={spinning}
        title="Clear this app's cache and reload"
        className={`flex items-center gap-1.5 text-gray-400 hover:text-indigo-600 text-xs transition-colors disabled:opacity-50 ${className}`}
      >
        <RefreshCw size={14} className={spinning ? "animate-spin" : ""} />
        {spinning ? "Refreshing…" : label}
      </button>
    );
  }

  return (
    <button
      onClick={handleClick}
      disabled={spinning}
      title="Clear this app's cache and reload"
      className={`p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors disabled:opacity-50 ${className}`}
    >
      <RefreshCw size={16} className={spinning ? "animate-spin" : ""} />
    </button>
  );
}
