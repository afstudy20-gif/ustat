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

export async function listRecentSessions(): Promise<RecentSessionMeta[]> {
  const db = getDb();
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

/** Upsert a session blob. If a record already exists for the same
 *  `serverSessionId`, overwrite it (keeps the auto-save stream from
 *  multiplying duplicates as the user works). */
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
  const existing = input.serverSessionId
    ? await db.sessions
        .where("serverSessionId")
        .equals(input.serverSessionId)
        .first()
    : undefined;
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
