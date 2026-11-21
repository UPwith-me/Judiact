"""Append-only evaluation output, with recovery of an interrupted final write."""

import json
from pathlib import Path


def read_predictions(path):
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    with path.open("rb+") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if handle.read(1) or line.endswith(b"\n"):
                    raise ValueError(f"Invalid complete prediction at {path}:{start}")
                # A killed worker can leave one incomplete JSON object, never a whole batch.
                handle.seek(start)
                handle.truncate()
                break
            identifier = row["sample_id"]
            if identifier in rows and rows[identifier] != row:
                raise ValueError(f"Conflicting predictions for {identifier} in {path}")
            rows[identifier] = row
        handle.seek(0, 2)
        if handle.tell():
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    return rows


def append_prediction(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def record_settings(directory, settings):
    path = Path(directory) / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != settings:
            raise ValueError(f"Resume would mix different evaluation settings: {path}")
    else:
        path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
