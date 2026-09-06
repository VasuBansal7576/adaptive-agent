"""Private AppWorld worker entry point.

Kept outside the adaptive_agent package so AppWorld's pinned Pydantic 1.x
runtime cannot import the main package's Pydantic 2.x models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    for method in ("model_dump", "to_dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except TypeError:
                continue
    return str(value)


def _public_supervisor(value: Any) -> dict[str, str]:
    """Keep only the public supervisor identity fields used for discovery."""
    fields = {
        "first_name": "firstName",
        "last_name": "lastName",
        "email": "email",
        "phone_number": "phoneNumber",
    }
    result: dict[str, str] = {}
    for source, target in fields.items():
        item = value.get(source) if isinstance(value, Mapping) else getattr(value, source, None)
        if isinstance(item, str) and item:
            result[target] = item
    return result


def _public_app_descriptions(value: Any) -> dict[str, str]:
    """Return public app descriptions without carrying runtime objects across the boundary."""
    if not isinstance(value, Mapping):
        return {}
    return {str(name): description for name, description in value.items() if isinstance(description, str)}


def run(root: Path) -> int:
    from appworld.environment import AppWorld  # type: ignore[import-not-found]

    world: Any | None = None
    for raw in sys.stdin:
        frame: dict[str, Any] = {}
        try:
            frame = json.loads(raw)
            if not isinstance(frame, dict) or not isinstance(frame.get("id"), int):
                raise ValueError("request frame requires integer id")
            operation = frame.get("operation")
            payload = frame.get("payload") or {}
            if not isinstance(payload, dict):
                raise ValueError("request payload must be an object")
            if operation == "reset":
                if world is not None:
                    world.close()
                task_id = str(payload["taskId"])
                world = AppWorld(
                    task_id,
                    experiment_name=str(payload.get("experimentName", "adaptive-agent")),
                    random_seed=int(payload.get("seed", 0)),
                    load_ground_truth=True,
                    ground_truth_mode="minimal",
                    raise_on_failure=False,
                    show_api_response_schemas=False,
                )
                result: Any = {
                    "taskId": task_id,
                    "instruction": str(world.task.instruction),
                    "allowedApps": list(getattr(world.task, "allowed_apps", ())),
                    "supervisor": _public_supervisor(getattr(world.task, "supervisor", {})),
                    "appDescriptions": _public_app_descriptions(getattr(world.task, "app_descriptions", {})),
                }
            elif operation == "call":
                if world is None:
                    raise ValueError("worker has not been reset")
                tool = str(payload["tool"])
                app, separator, api = tool.partition("__")
                if not separator or not app or not api or app.startswith("_") or api.startswith("_"):
                    raise ValueError("invalid AppWorld API name")
                app_obj = getattr(world.apis, app, None)
                fn = getattr(app_obj, api, None) if app_obj is not None else None
                if not callable(fn):
                    raise ValueError(f"unknown AppWorld API: {tool}")
                arguments = payload.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise ValueError("API arguments must be an object")
                result = _jsonable(fn(**arguments))
            elif operation == "evaluate":
                if world is None:
                    raise ValueError("worker has not been reset")
                tracker = world.evaluate()
                result = {"success": bool(getattr(tracker, "success", False)), "numTests": int(getattr(tracker, "num_tests", 0) or 0), "passCount": int(getattr(tracker, "pass_count", 0) or 0), "failCount": int(getattr(tracker, "fail_count", 0) or 0), "taskCompleted": bool(world.task_completed())}
            elif operation == "close":
                if world is not None:
                    world.close()
                print(json.dumps({"id": frame["id"], "ok": True, "result": {"closed": True}}, separators=(",", ":")), flush=True)
                return 0
            else:
                raise ValueError(f"unknown worker operation: {operation}")
            response = {"id": frame["id"], "ok": True, "result": result}
        except Exception as exc:
            response = {"id": frame.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, separators=(",", ":")), flush=True)
    if world is not None:
        world.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    raise SystemExit(run(Path(args.root)))
