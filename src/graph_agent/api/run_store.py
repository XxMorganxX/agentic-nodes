from __future__ import annotations

import os
from typing import Any, Protocol


class RunStore(Protocol):
    def initialize_run(self, state: dict[str, Any]) -> None: ...

    def append_event(self, run_id: str, event: dict[str, Any]) -> None: ...

    def write_state(self, run_id: str, state: dict[str, Any]) -> None: ...

    def load_manifest(self, run_id: str) -> dict[str, Any] | None: ...

    def load_events(self, run_id: str) -> list[dict[str, Any]]: ...

    def load_state(self, run_id: str) -> dict[str, Any] | None: ...

    def recover_run_state(self, run_id: str) -> dict[str, Any] | None: ...

    def list_runs(self, *, graph_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]: ...


def build_default_run_store() -> RunStore:
    backend = os.environ.get("GRAPH_AGENT_RUN_STORE", "filesystem").strip().lower() or "filesystem"
    if backend == "supabase":
        from graph_agent.api.supabase_run_store import SupabaseRunStore

        return SupabaseRunStore.from_env()
    from graph_agent.api.run_log_store import FilesystemRunStore

    return FilesystemRunStore()
