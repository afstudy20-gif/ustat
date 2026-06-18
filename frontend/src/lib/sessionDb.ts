/**
 * Local-only autosave for uSTAT sessions.
 *
 * Mirrors the endnotere editor pattern (Dexie / IndexedDB store + 600 ms
 * debounce + cross-tab BroadcastChannel) but adapted for uSTAT's
 * server-side session model:
 *
 *   - The FastAPI backend keeps the live DataFrame in memory with a 30 min
 *     TTL. Snapshotting it for "resume where I left off" requires fetching
 *     the existing JSON exporter (`GET /api/sessions/{sid}/save_session`)
 *     and storing the blob in IndexedDB.
 *   - Resuming reuploads the stored JSON via `POST
 *     /api/sessions/load_session` (multipart File), receives a fresh
 *     session_id, and hands the resulting Session object back to the
 *     Zustand store.
 *
 * Nothing leaves the user's browser — the snapshot lives in IndexedDB,
 * scoped to this origin only.
 */

import Dexie, { type EntityTable } from "dexie";

// ── Types ─────────────────────────────────────────────────────────────

/** Lightweight metadata kept alongside each saved blob. Mirrored to the
 *  card grid on the upload zone without having to deserialise the full
 *  session every render. */
export interface RecentSessionMeta {
  id: string;             // local UUID — not the server session_id
  serverSessionId?: string; // last-known server id (helps dedupe)
  name: string;             // dataset filename or user-chosen label
  savedAt: number;          // epoch ms — used for LRU ordering
  sizeBytes: number;        // JSON blob length, for the storage cap
  nRows?: number;
  nCols?: number;
  activeTab?: string;       // header tab the user was on
  source: "auto" | "manual";
}

/** Full record stored in IndexedDB — extends the metadata with the
 *  serialised session JSON. */
export interface RecentSessionRecord extends RecentSessionMeta {
  // Stringified JSON returned by the backend's save_session endpoint.
  // Keep as a string (not parsed) so re-uploading is a one-liner and
  // there is no schema-version coupling here.
  payload: string;
}

// ── DB schema ─────────────────────────────────────────────────────────

export interface SessionDB extends Dexie {
  sessions: EntityTable<RecentSessionRecord, "id">;
}

let _db: SessionDB | null = null;

function getDb(): SessionDB {
  if (typeof window === "undefined") {
    throw new Error("sessionDb is browser-only");
  }
  if (_db) return _db;
  const db = new Dexie("wiz3-sessions-v1") as SessionDB;
  db.version(1).stores({
    // Indexes: id (primary) + savedAt (LRU ordering) + serverSessionId
    // (dedup lookups on autosave).
    sessions: "id, savedAt, serverSessionId",
  });
  // v2 adds a `name` index for filename-based dedup — the server
  // session_id isn't stable across reloads, so the filename is the
  // identity that collapses duplicate rows.
  db.version(2).stores({
    sessions: "id, savedAt, serverSessionId, name",
  });
  _db = db;
  return db;
}

// ── Capacity policy ──────────────────────────────────────────────────

const MAX_SESSIONS = 20;
const MAX_TOTAL_BYTES = 200 * 1024 * 1024; // 200 MB hard cap

/** Drop the oldest records until the store fits within the cap. */
async function pruneToCap(): Promise<void> {
  const db = getDb();
  const all = await db.sessions.orderBy("savedAt").toArray();
  if (all.length <= MAX_SESSIONS) {
    const total = all.reduce((s, r) => s + r.sizeBytes, 0);
    if (total <= MAX_TOTAL_BYTES) return;
  }
  let total = all.reduce((s, r) => s + r.sizeBytes, 0);
  let i = 0;
  // Oldest-first deletion until both caps satisfied.
  while (
    (all.length - i > MAX_SESSIONS || total > MAX_TOTAL_BYTES) &&
    i < all.length - 1 // never delete the very newest
  ) {
    total -= all[i].sizeBytes;
    await db.sessions.delete(all[i].id);
    i++;
  }
}

// ── Public API ────────────────────────────────────────────────────────

/** Generate a local id. Avoids importing a uuid library — Dexie just
 *  needs uniqueness within this store. */
function newLocalId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

/** Collapse records that share a display name down to the newest one.
 *  The server session_id changes on every upload / reload / restore, so
 *  the same logical file accumulates a row per browser session. Identity
 *  that survives reloads is the *name*, so we keep the freshest snapshot
 *  per name and drop the stale ones. */
async function dedupeByName(): Promise<void> {
  const db = getDb();
  const rows = await db.sessions.orderBy("savedAt").reverse().toArray();
  const seen = new Set<string>();
  const stale: string[] = [];
  for (const r of rows) {
    const key = r.name || r.id;
    if (seen.has(key)) {
      stale.push(r.id);   // older (rows already sorted newest-first)
    } else {
      seen.add(key);
    }
  }
  if (stale.length) {
    await db.sessions.bulkDelete(stale);
  }
}

export async function listRecentSessions(): Promise<RecentSessionMeta[]> {
  const db = getDb();
  await dedupeByName();
  const rows = await db.sessions.orderBy("savedAt").reverse().toArray();
  return rows.map((row) => {
    const { payload: _ignored, ...meta } = row;
    void _ignored;
    return meta;
  });
}

export async function getRecentSession(id: string): Promise<RecentSessionRecord | undefined> {
  return getDb().sessions.get(id);
}

export async function deleteRecentSession(id: string): Promise<void> {
  await getDb().sessions.delete(id);
}

export async function clearAllRecentSessions(): Promise<void> {
  await getDb().sessions.clear();
}

/** Upsert a session blob, deduping so the same logical file occupies a
 *  single row.
 *
 *  The server session_id is NOT stable — every upload, reload, and
 *  restore mints a fresh one — so keying only on it spawns a new row per
 *  browser session (the "why are there 3 copies of my file" bug). We
 *  match in two passes:
 *    1. by serverSessionId  → same in-progress session (covers renames,
 *       where the id is stable but the name just changed)
 *    2. by name             → same file across reloads / re-uploads /
 *       restores (the id differs but the filename is the user's stable
 *       identity)
 */
export async function upsertRecentSession(input: {
  serverSessionId: string;
  name: string;
  payload: string;
  nRows?: number;
  nCols?: number;
  activeTab?: string;
  source: "auto" | "manual";
}): Promise<RecentSessionMeta> {
  const db = getDb();
  let existing =
    input.serverSessionId
      ? await db.sessions.where("serverSessionId").equals(input.serverSessionId).first()
      : undefined;
  if (!existing && input.name) {
    existing = await db.sessions.where("name").equals(input.name).first();
  }
  const id = existing?.id ?? newLocalId();
  const rec: RecentSessionRecord = {
    id,
    serverSessionId: input.serverSessionId,
    name: input.name,
    payload: input.payload,
    sizeBytes: input.payload.length,
    nRows: input.nRows,
    nCols: input.nCols,
    activeTab: input.activeTab,
    savedAt: Date.now(),
    source: input.source,
  };
  await db.sessions.put(rec);
  await pruneToCap();
  const { payload: _ignored, ...meta } = rec;
  void _ignored;
  return meta;
}

/**
 * Raw upsert used by the Google Drive cloud-sync pull path. Identical to
 * {@link upsertRecentSession} EXCEPT it preserves the original `savedAt`
 * timestamp instead of stamping `Date.now()`. This is essential for the
 * last-write-wins clock: when a remote snapshot is pulled back locally, the
 * record's `savedAt` must reflect when the snapshot was *taken* (so a
 * subsequent push does not mark it newer than the remote and bounce it
 * back), not when it landed in IndexedDB.
 */
export async function upsertRecentSessionRaw(input: {
  serverSessionId?: string;
  name: string;
  payload: string;
  savedAt: number;
  nRows?: number;
  nCols?: number;
  activeTab?: string;
  source: "auto" | "manual";
}): Promise<RecentSessionMeta> {
  const db = getDb();
  let existing: RecentSessionRecord | undefined =
    input.serverSessionId
      ? await db.sessions.where("serverSessionId").equals(input.serverSessionId).first()
      : undefined;
  if (!existing && input.name) {
    existing = await db.sessions.where("name").equals(input.name).first();
  }
  const id = existing?.id ?? newLocalId();
  const rec: RecentSessionRecord = {
    id,
    serverSessionId: input.serverSessionId,
    name: input.name,
    payload: input.payload,
    sizeBytes: input.payload.length,
    nRows: input.nRows,
    nCols: input.nCols,
    activeTab: input.activeTab,
    savedAt: input.savedAt,
    source: input.source,
  };
  await db.sessions.put(rec);
  await pruneToCap();
  const { payload: _ignored, ...meta } = rec;
  void _ignored;
  return meta;
}

// ── Cross-tab notifications ───────────────────────────────────────────

const CHANNEL = "wiz3-sessions";

let _bc: BroadcastChannel | null = null;

function getChannel(): BroadcastChannel | null {
  if (typeof window === "undefined") return null;
  if (typeof BroadcastChannel === "undefined") return null;
  if (_bc) return _bc;
  _bc = new BroadcastChannel(CHANNEL);
  return _bc;
}

export function notifySessionsChanged(): void {
  const bc = getChannel();
  if (bc) bc.postMessage({ type: "changed", at: Date.now() });
}

export function subscribeSessions(onChange: () => void): () => void {
  const bc = getChannel();
  if (!bc) return () => {};
  const handler = () => onChange();
  bc.addEventListener("message", handler);
  return () => bc.removeEventListener("message", handler);
}

// ── Storage estimate (for diagnostics / UI) ───────────────────────────

export async function getStorageEstimate(): Promise<{
  count: number;
  bytes: number;
  capCount: number;
  capBytes: number;
}> {
  const db = getDb();
  const all = await db.sessions.toArray();
  const bytes = all.reduce((s, r) => s + r.sizeBytes, 0);
  return {
    count: all.length,
    bytes,
    capCount: MAX_SESSIONS,
    capBytes: MAX_TOTAL_BYTES,
  };
}
