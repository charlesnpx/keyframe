from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from keyframe.pipeline.snapshotters import SnapshotterRegistry


class TraceSink:
    def enter(self, stage: str, value: object | None = None) -> None:
        pass

    def exit(self, stage: str, value: object | None = None) -> None:
        pass

    def decision(self, stage: str, name: str, payload: Mapping[str, Any]) -> None:
        pass

    def write(self, path: str | Path) -> None:
        pass


class NoOpTraceSink(TraceSink):
    pass


class SnapshotTraceSink(TraceSink):
    def __init__(self, snapshotter: SnapshotterRegistry | None = None):
        self.records: list[dict[str, Any]] = []
        self.snapshotter = snapshotter or SnapshotterRegistry()

    def enter(self, stage: str, value: object | None = None) -> None:
        self.records.append(self._materialize("enter", stage, value))

    def exit(self, stage: str, value: object | None = None) -> None:
        self.records.append(self._materialize("exit", stage, value))

    def decision(self, stage: str, name: str, payload: Mapping[str, Any]) -> None:
        self.records.append(self._materialize("decision", stage, {"name": name, "payload": dict(payload)}))

    def _materialize(self, event: str, stage: str, value: object | None) -> dict[str, Any]:
        payload = None if value is None else self.snapshotter.snapshot(stage, value)
        record = {
            "event": event,
            "stage": stage,
            "payload": payload,
        }
        return json.loads(json.dumps(record, sort_keys=True))

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema": 1, "records": self.records}, indent=2),
            encoding="utf-8",
        )
