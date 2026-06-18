/**
 * Google OAuth client configuration for uSTAT cloud sync.
 *
 * Mirrors notepad's `js/cloud-config.js`, adapted for uSTAT's session model.
 *
 * SETUP (one-time, by app owner):
 *   1. https://console.cloud.google.com → New Project.
 *   2. APIs & Services → Enable "Google Drive API".
 *   3. OAuth consent screen (External):
 *        - App name: uSTAT
 *        - Add scope: https://www.googleapis.com/auth/drive.appdata
 *   4. Credentials → Create Credentials → OAuth client ID (Web application):
 *        - Authorized JavaScript origins:
 *            https://ustat.drtr.uk          (web deployment)
 *            http://localhost:5173           (Vite dev)
 *            http://127.0.0.1:5173           (Tauri webview, dev)
 *            http://localhost                (loopback, any port)
 *            http://127.0.0.1                (loopback, any port)
 *        - Authorized redirect URIs (iOS standalone / blocked-popup fallback):
 *            https://ustat.drtr.uk/
 *            http://localhost:5173/
 *            http://127.0.0.1:5173/
 *            (NOTE the trailing slash — must EXACTLY match location.origin + '/')
 *   5. Paste the Client ID below. Client IDs are PUBLIC — safe to commit.
 *
 * Cross-platform notes:
 *   - uSTAT's Tauri desktop webview runs at http://127.0.0.1:<port> (NOT a
 *     custom protocol), so it behaves like a regular web origin. The GIS
 *     popup flow works there once 127.0.0.1 is an authorized origin.
 *   - Popup flow everywhere by default; the module auto-switches to a
 *     full-page redirect flow when popups are unreliable (iOS standalone,
 *     embedded WebViews). Redirect flow REQUIRES the redirect URIs above.
 *   - Same Google account on any platform ⇒ same hidden appDataFolder
 *     ⇒ same uSTAT sessions.
 */

/** All config constants for the cloud-sync module. */
export const CLOUD_CONFIG = {
  // PASTE OAUTH CLIENT ID HERE (format: 1234567890-abcdef.apps.googleusercontent.com)
  GOOGLE_CLIENT_ID:
    "866965837196-5rvslbk301vi2j0rg3l4697ocptnb21e.apps.googleusercontent.com",

  // Scope: hidden app-private folder in user's Drive + profile info.
  SCOPE:
    "https://www.googleapis.com/auth/drive.appdata " +
    "https://www.googleapis.com/auth/userinfo.email " +
    "https://www.googleapis.com/auth/userinfo.profile",

  // Drive API base endpoints.
  DRIVE_API: "https://www.googleapis.com/drive/v3",
  DRIVE_UPLOAD: "https://www.googleapis.com/upload/drive/v3",

  // Sync tuning (mirrors notepad's defaults).
  PUSH_DEBOUNCE_MS: 30_000, // wait N ms after last snapshot before pushing
  PULL_INTERVAL_MS: 120_000, // background pull cadence
  MAX_SESSION_BYTES: 5 * 1024 * 1024, // 5 MB per session JSON (Drive accepts larger; cap for sanity)
} as const;

// Drive appDataFolder layout:
//   ustat-index.json        → { version, lastSync, sessions: [{ id, name, updated, bytes, rev }] }
//   ustat-session-<id>.json → { meta: {...}, payload: "<save_session JSON string>" }
export const INDEX_FILE_NAME = "ustat-index.json";
export const sessionFileName = (id: string): string => `ustat-session-${id}.json`;
