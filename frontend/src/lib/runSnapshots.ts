import type { RunState, RuntimeEvent } from "./types";

export type PersistedRunSnapshot = {
  graphId: string;
  activeRunId: string | null;
  events: RuntimeEvent[];
  runState: RunState | null;
  savedAt: string;
};

const STORAGE_KEY = "agentic-nodes-run-snapshots-v1";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readSnapshotMap(): Record<string, PersistedRunSnapshot> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).filter(([, value]) => {
        if (!isRecord(value)) {
          return false;
        }
        return typeof value.graphId === "string";
      }),
    ) as Record<string, PersistedRunSnapshot>;
  } catch {
    return {};
  }
}

function writeSnapshotMap(snapshots: Record<string, PersistedRunSnapshot>): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshots));
}

export function loadPersistedRunSnapshot(graphId: string): PersistedRunSnapshot | null {
  const snapshots = readSnapshotMap();
  return snapshots[graphId] ?? null;
}

export function savePersistedRunSnapshot(snapshot: PersistedRunSnapshot): PersistedRunSnapshot {
  const snapshots = readSnapshotMap();
  snapshots[snapshot.graphId] = snapshot;
  writeSnapshotMap(snapshots);
  return snapshot;
}

export function clearPersistedRunSnapshot(graphId: string): void {
  const snapshots = readSnapshotMap();
  if (!Object.prototype.hasOwnProperty.call(snapshots, graphId)) {
    return;
  }
  delete snapshots[graphId];
  writeSnapshotMap(snapshots);
}

export function clearAllPersistedRunSnapshots(): void {
  writeSnapshotMap({});
}
