from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from graph_agent.api.run_state_reducer import build_run_state, replay_events
from graph_agent.runtime.core import utc_now_iso


def _merge_snapshot_metadata(recovered: dict[str, Any], snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        return recovered
    for key in ("status_reason", "runtime_instance_id", "last_heartbeat_at"):
        if key in snapshot:
            recovered[key] = snapshot.get(key)
    snapshot_agent_runs = snapshot.get("agent_runs")
    recovered_agent_runs = recovered.get("agent_runs")
    if isinstance(snapshot_agent_runs, dict) and isinstance(recovered_agent_runs, dict):
        for agent_id, snapshot_agent_state in snapshot_agent_runs.items():
            recovered_agent_state = recovered_agent_runs.get(agent_id)
            if not isinstance(snapshot_agent_state, dict) or not isinstance(recovered_agent_state, dict):
                continue
            for key in ("status_reason", "runtime_instance_id", "last_heartbeat_at"):
                if key in snapshot_agent_state:
                    recovered_agent_state[key] = snapshot_agent_state.get(key)
    return recovered


class FilesystemRunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[3] / ".logs" / "runs"

    def initialize_run(self, state: Mapping[str, Any]) -> None:
        run_id = str(state["run_id"])
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            self._write_json(
                manifest_path,
                {
                    "run_id": run_id,
                    "graph_id": state.get("graph_id"),
                    "agent_id": state.get("agent_id"),
                    "agent_name": state.get("agent_name"),
                    "parent_run_id": state.get("parent_run_id"),
                    "input_payload": state.get("input_payload"),
                    "created_at": utc_now_iso(),
                },
            )
        self.write_state(run_id, state)

    def append_event(self, run_id: str, event: Mapping[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    def write_state(self, run_id: str, state: Mapping[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "state.json", state)

    def load_manifest(self, run_id: str) -> dict[str, Any] | None:
        manifest_path = self._run_dir(run_id) / "manifest.json"
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        events_path = self._run_dir(run_id) / "events.jsonl"
        if not events_path.exists():
            return []
        return [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def load_state(self, run_id: str) -> dict[str, Any] | None:
        state_path = self._run_dir(run_id) / "state.json"
        if not state_path.exists():
            return None
        return json.loads(state_path.read_text(encoding="utf-8"))

    def recover_run_state(self, run_id: str) -> dict[str, Any] | None:
        manifest = self.load_manifest(run_id) or {}
        snapshot = self.load_state(run_id)
        events = self.load_events(run_id)
        if not manifest and snapshot is None and not events:
            return None
        if not events:
            return snapshot
        graph_id = str(
            manifest.get("graph_id")
            or (snapshot or {}).get("graph_id")
            or run_id
        )
        input_payload = manifest.get("input_payload", (snapshot or {}).get("input_payload"))
        initial_state = build_run_state(
            run_id,
            graph_id,
            input_payload,
            agent_id=manifest.get("agent_id", (snapshot or {}).get("agent_id")),
            parent_run_id=manifest.get("parent_run_id", (snapshot or {}).get("parent_run_id")),
            agent_name=manifest.get("agent_name", (snapshot or {}).get("agent_name")),
        )
        return _merge_snapshot_metadata(replay_events(initial_state, events), snapshot)

    def list_runs(self, *, graph_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for run_dir in sorted(self.root.iterdir(), key=lambda candidate: candidate.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            state = self.load_state(run_dir.name)
            recovered = self.recover_run_state(run_dir.name)
            manifest = self.load_manifest(run_dir.name)
            if state is None and manifest is None:
                continue
            graph_value = (state or {}).get("graph_id", (manifest or {}).get("graph_id"))
            if graph_id is not None and graph_value != graph_id:
                continue
            rows.append(
                {
                    "run_id": str((recovered or state or {}).get("run_id") or (manifest or {}).get("run_id") or run_dir.name),
                    "graph_id": graph_value,
                    "status": (recovered or state or {}).get("status"),
                    "status_reason": (recovered or state or {}).get("status_reason"),
                    "started_at": (recovered or state or {}).get("started_at"),
                    "ended_at": (recovered or state or {}).get("ended_at"),
                    "created_at": (manifest or {}).get("created_at"),
                    "agent_id": (recovered or state or {}).get("agent_id", (manifest or {}).get("agent_id")),
                    "agent_name": (recovered or state or {}).get("agent_name", (manifest or {}).get("agent_name")),
                    "parent_run_id": (recovered or state or {}).get("parent_run_id", (manifest or {}).get("parent_run_id")),
                    "runtime_instance_id": (recovered or state or {}).get("runtime_instance_id"),
                    "last_heartbeat_at": (recovered or state or {}).get("last_heartbeat_at"),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(path)


RunLogStore = FilesystemRunStore
