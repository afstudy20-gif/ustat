/**
 * UpdatePrompt — auto-detects a newer build of uSTAT and offers a one-click reload.
 *
 * How it works:
 *   1. vite-plugin-pwa registers a service worker that pre-caches every asset
 *      hashed in the dist/ folder. When you deploy a new build, the SW spec
 *      tells the browser "there is a new SW waiting" → `needRefresh` fires.
 *   2. We also poll the SW registration with `registration.update()` every
 *      60 s so users who keep a long-lived tab open get the prompt without
 *      having to navigate away. Network-failure-tolerant.
 *   3. If the SW lifecycle is unavailable for any reason (Safari private
 *      mode, corporate proxy stripping SW, etc.) we fall back to polling a
 *      static `/version.json` artifact produced at build time and showing
 *      a "soft" reload banner when the hash changes.
 *
 *   The toast is dismissible. Clicking "Reload to update" calls
 *   `updateServiceWorker(true)` which activates the waiting SW and reloads
 *   the page, picking up the new bundle.
 */

import { useEffect, useState } from "react";
import { useRegisterSW } from "virtual:pwa-register/react";
import { RefreshCw, X, AlertCircle, CheckCircle2 } from "lucide-react";

// Build-time version stamp. Injected by vite.config.ts via `define`.
declare const __APP_VERSION__: string;
declare const __BUILD_TIME__: string;

const POLL_INTERVAL_MS = 60_000; // 1 minute — cheap, picks up fresh deploys fast
const FALLBACK_POLL_MS = 5 * 60_000; // 5 minutes — non-SW fallback poll

export default function UpdatePrompt() {
  // ── Primary path: service-worker update detection ───────────────────
  const {
    needRefresh: [needRefresh, setNeedRefresh],
    offlineReady: [offlineReady, setOfflineReady],
    updateServiceWorker,
  } = useRegisterSW({
    onRegisteredSW(_swUrl, registration) {
      if (!registration) return;
      // Periodically ask the browser to check for a newer SW. Without this,
      // a tab left open for hours after a deploy may never refresh.
      setInterval(() => {
        registration.update().catch(() => null);
      }, POLL_INTERVAL_MS);
    },
    onRegisterError(err) {
      console.warn("[UpdatePrompt] SW register failed:", err);
    },
  });

  // ── Fallback path: poll /version.json when SW is unavailable ────────
  const [fallbackStale, setFallbackStale] = useState(false);
  useEffect(() => {
    if ("serviceWorker" in navigator) return; // primary path handles it
    const check = async () => {
      try {
        const res = await fetch("/version.json", { cache: "no-store" });
        if (!res.ok) return;
        const { version, build } = (await res.json()) as { version?: string; build?: string };
        // Build timestamp is the canonical identifier — package.json version
        // may not be bumped between deploys, but the build stamp always is.
        const remoteKey = build ?? version ?? "";
        const localKey = (typeof __BUILD_TIME__ === "string" ? __BUILD_TIME__ : __APP_VERSION__) ?? "";
        if (remoteKey && localKey && remoteKey !== localKey) {
          setFallbackStale(true);
        }
      } catch {
        /* network blip — try again next tick */
      }
    };
    void check();
    const t = setInterval(check, FALLBACK_POLL_MS);
    return () => clearInterval(t);
  }, []);

  const showRefresh = needRefresh || fallbackStale;
  const showOffline = offlineReady && !needRefresh;

  if (!showRefresh && !showOffline) return null;

  // ── UI ──────────────────────────────────────────────────────────────
  if (showRefresh) {
    return (
      <div className="fixed bottom-4 right-4 z-[60] max-w-sm animate-in slide-in-from-bottom-2 fade-in duration-300">
        <div className="bg-white border-2 border-indigo-500 rounded-2xl shadow-2xl shadow-indigo-200/50 overflow-hidden">
          <div className="bg-indigo-600 text-white px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle size={16} />
              <span className="text-xs font-bold tracking-tight">Yeni sürüm hazır</span>
            </div>
            <button
              onClick={() => { setNeedRefresh(false); setFallbackStale(false); }}
              className="text-indigo-200 hover:text-white hover:bg-indigo-700 rounded p-0.5 transition-colors"
              title="Dismiss (you can reload later from the header)"
            >
              <X size={14} />
            </button>
          </div>
          <div className="p-4">
            <p className="text-xs text-gray-700 leading-relaxed">
              uSTAT güncellendi. Mevcut sekmeniz eski sürümü kullanıyor olabilir.
              Yeni özellikleri ve hata düzeltmelerini almak için sayfayı yenileyin.
            </p>
            <p className="text-[10px] text-gray-400 font-mono mt-1">
              Eski: v{__APP_VERSION__} · Build {__BUILD_TIME__}
            </p>
            <div className="mt-3 flex items-center gap-2">
              <button
                onClick={() => {
                  if (fallbackStale) {
                    // No SW to swap — just hard reload with a cache buster.
                    const url = new URL(location.href);
                    url.searchParams.set("_v", Date.now().toString(36));
                    location.replace(url.toString());
                    return;
                  }
                  void updateServiceWorker(true);
                }}
                className="flex-1 flex items-center justify-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold px-3 py-2 rounded-lg shadow-sm transition-colors"
              >
                <RefreshCw size={14} />
                Reload to update
              </button>
              <button
                onClick={() => { setNeedRefresh(false); setFallbackStale(false); }}
                className="text-xs text-gray-500 hover:text-gray-700 px-2 py-2"
              >
                Sonra
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Optional: subtle "ready to work offline" confirmation, auto-dismissing.
  return (
    <div className="fixed bottom-4 right-4 z-[60] max-w-sm animate-in fade-in duration-300">
      <div className="bg-emerald-50 border border-emerald-200 rounded-xl shadow-md px-3 py-2 flex items-center gap-2">
        <CheckCircle2 size={14} className="text-emerald-600 flex-shrink-0" />
        <span className="text-[11px] text-emerald-800">Çevrimdışı kullanım için hazır.</span>
        <button
          onClick={() => setOfflineReady(false)}
          className="ml-1 text-emerald-600 hover:text-emerald-900"
        >
          <X size={12} />
        </button>
      </div>
    </div>
  );
}
