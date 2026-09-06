"""Environment registration, JSON Schema validation, and capability discovery.

No finance, support, or IT workflow is encoded here. Tool behavior and outcomes are
supplied by the environment package through its reset fixture, policy, and evaluator.

The validator implements the draft-2020-12 subset needed by environment contracts:
type unions, const/enum, numeric bounds (minimum, maximum, exclusiveMinimum,
exclusiveMaximum, multipleOf), string constraints (minLength, maxLength, pattern),
array constraints (minItems, maxItems, uniqueItems, items), object constraints
(properties, required, additionalProperties, propertyNames), composition
(allOf/anyOf/oneOf/not), and dependentRequired. Local `$defs`/`$ref` aliases are
resolved within the same schema document. External refs are not supported.
"""

from __future__ import annotations

import re
from typing import Any

from adaptive_agent.models import ArtifactRef, EnvironmentManifest, TaskInput, ToolSchema
from adaptive_agent.store import Store


class SchemaValidationError(ValueError):
    pass


_TYPE_MAP = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _check_type(value: Any, stype: str, path: str) -> None:
    check = _TYPE_MAP.get(stype)
    if check is None:
        raise SchemaValidationError(f"{path}: unsupported schema type {stype!r}")
    if not check(value):
        raise SchemaValidationError(f"{path}: expected {stype}, got {type(value).__name__}")


def _resolve_ref(ref: str, root: dict[str, Any], path: str) -> dict[str, Any]:
    """Resolve local '#/$defs/name' or '#/definitions/name' aliases only."""
    if not ref.startswith("#/"):
        raise SchemaValidationError(f"{path}: external $ref {ref!r} not supported")
    parts = ref[2:].split("/")
    node: Any = root
    for part in parts:
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SchemaValidationError(f"{path}: unresolvable $ref {ref!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaValidationError(f"{path}: $ref {ref!r} does not resolve to a schema")
    return node


def _validate(value: Any, schema: Any, path: str, root: dict[str, Any], depth: int) -> None:
    if depth > 64:
        raise SchemaValidationError(f"{path}: schema recursion depth exceeded")
    if isinstance(schema, bool):
        if schema is False:
            raise SchemaValidationError(f"{path}: boolean schema false rejects all values")
        return
    if not isinstance(schema, dict):
        raise SchemaValidationError(f"{path}: schema must be an object or boolean")

    if "$ref" in schema:
        _validate(value, _resolve_ref(schema["$ref"], root, path), path, root, depth + 1)
        # Sibling keywords are allowed alongside $ref in draft 2020-12; continue validating.

    # Composition
    if "allOf" in schema:
        for i, sub in enumerate(schema["allOf"]):
            _validate(value, sub, f"{path}.allOf[{i}]", root, depth + 1)
    if "anyOf" in schema:
        for i, sub in enumerate(schema["anyOf"]):
            try:
                _validate(value, sub, f"{path}.anyOf[{i}]", root, depth + 1)
                break
            except SchemaValidationError:
                continue
        else:
            raise SchemaValidationError(f"{path}: value matches no anyOf branch")
    if "oneOf" in schema:
        matches = 0
        for i, sub in enumerate(schema["oneOf"]):
            try:
                _validate(value, sub, f"{path}.oneOf[{i}]", root, depth + 1)
                matches += 1
            except SchemaValidationError:
                continue
        if matches != 1:
            raise SchemaValidationError(f"{path}: value matches {matches} oneOf branches")
    if "not" in schema:
        try:
            _validate(value, schema["not"], f"{path}.not", root, depth + 1)
        except SchemaValidationError:
            pass
        else:
            raise SchemaValidationError(f"{path}: value must NOT match the 'not' schema")

    # Type (single or union alias)
    stype = schema.get("type")
    if isinstance(stype, list):
        if not any(_TYPE_MAP.get(t, lambda v: False)(value) for t in stype):
            raise SchemaValidationError(f"{path}: expected one of {stype}, got {type(value).__name__}")
    elif isinstance(stype, str):
        _check_type(value, stype, path)

    # Enumerated / constant
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path}: value {value!r} != const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: value {value!r} not in enum {schema['enum']}")

    # Numeric constraints
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path}: {value} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise SchemaValidationError(f"{path}: {value} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise SchemaValidationError(f"{path}: {value} >= exclusiveMaximum {schema['exclusiveMaximum']}")
        if "multipleOf" in schema:
            m = schema["multipleOf"]
            if not isinstance(m, (int, float)) or m == 0:
                raise SchemaValidationError(f"{path}: invalid multipleOf {m!r}")
            if abs(value / m - round(value / m)) > 1e-9:
                raise SchemaValidationError(f"{path}: {value} is not a multiple of {m}")

    # String constraints
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaValidationError(f"{path}: string shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaValidationError(f"{path}: string longer than maxLength {schema['maxLength']}")
        if "pattern" in schema:
            if not re.search(schema["pattern"], value):
                raise SchemaValidationError(f"{path}: string does not match pattern {schema['pattern']!r}")

    # Array constraints
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaValidationError(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(f"{path}: more than maxItems {schema['maxItems']}")
        if schema.get("uniqueItems"):
            seen = {json_key(i) for i in value}
            if len(seen) != len(value):
                raise SchemaValidationError(f"{path}: items are not unique")
        items = schema.get("items")
        if items is not None:
            for i, item in enumerate(value):
                _validate(item, items, f"{path}[{i}]", root, depth + 1)
        contains = schema.get("contains")
        if contains is not None:
            hit = False
            for i, item in enumerate(value):
                try:
                    _validate(item, contains, f"{path}[{i}]", root, depth + 1)
                    hit = True
                    break
                except SchemaValidationError:
                    continue
            if not hit:
                raise SchemaValidationError(f"{path}: no item matches 'contains' schema")

    # Object constraints
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise SchemaValidationError(f"{path}: missing required property '{key}'")
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in value:
                _validate(value[key], prop_schema, f"{path}.{key}", root, depth + 1)
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in value:
                if key not in properties:
                    raise SchemaValidationError(f"{path}: additional property '{key}' not allowed")
        elif isinstance(additional, dict):
            for key in value:
                if key not in properties:
                    _validate(value[key], additional, f"{path}.{key}", root, depth + 1)
        pnames = schema.get("propertyNames")
        if pnames is not None:
            for key in value:
                _validate(key, pnames, f"{path}.propertyNames[{key!r}]", root, depth + 1)
        deps = schema.get("dependentRequired", {})
        for key, needed in deps.items():
            if key in value:
                for dep in needed:
                    if dep not in value:
                        raise SchemaValidationError(
                            f"{path}: property '{key}' requires '{dep}'"
                        )


def json_key(value: Any) -> str:
    import json as _json

    return _json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def validate_value(value: Any, schema: Any) -> None:
    """Validate `value` against a JSON Schema document (subset)."""
    root = schema if isinstance(schema, dict) else {}
    _validate(value, schema, "$", root, 0)


def validate_arguments(tool_schema: ToolSchema, arguments: dict[str, Any]) -> None:
    try:
        validate_value(arguments, tool_schema.input_schema)
    except SchemaValidationError as exc:
        raise SchemaValidationError(f"{tool_schema.name}: {exc}") from exc


class EnvironmentRegistry:
    """Register and discover environment packages."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def register(self, manifest: EnvironmentManifest) -> ArtifactRef:
        self.validate_manifest(manifest)
        manifest_json = manifest.model_dump(mode="json", by_alias=True)
        manifest_ref = self.store.put_artifact(manifest_json)
        self.store.register_environment(
            manifest.environment_id,
            manifest.version,
            manifest_ref,
        )
        return manifest_ref

    def get_manifest(self, env_id: str) -> EnvironmentManifest | None:
        row = self.store.get_environment(env_id)
        if not row:
            return None
        manifest_ref = ArtifactRef.model_validate_json(row["manifest_ref"])
        data = self.store.get_artifact(manifest_ref)
        return EnvironmentManifest.model_validate(data)

    def get_tool_schema(self, env_id: str, tool_name: str) -> ToolSchema | None:
        manifest = self.get_manifest(env_id)
        if not manifest:
            return None
        for ts in manifest.tool_schemas:
            if ts.name == tool_name:
                return ts
        return None

    def discover_capabilities(self, env_id: str) -> dict[str, Any]:
        manifest = self.get_manifest(env_id)
        if not manifest:
            return {"error": "environment_not_found"}
        return {
            "environment_id": manifest.environment_id,
            "version": manifest.version,
            "capabilities": manifest.capabilities,
            "tools": [
                {
                    "name": ts.name,
                    "version": ts.version,
                    "effect": ts.effect,
                    "input_schema": ts.input_schema,
                    "output_schema": ts.output_schema,
                }
                for ts in manifest.tool_schemas
            ],
        }

    def validate_manifest(self, manifest: EnvironmentManifest) -> None:
        if manifest.schema_version != 1:
            raise SchemaValidationError("unsupported schema version")
        if not manifest.environment_id:
            raise SchemaValidationError("environment_id is required")
        if not manifest.tool_schemas:
            raise SchemaValidationError("at least one tool schema is required")
        for ref_name, ref in (
            ("policy_ref", manifest.policy_ref),
            ("evaluator_ref", manifest.evaluator_ref),
            ("reset_ref", manifest.reset_ref),
        ):
            if not ref.sha256:
                raise SchemaValidationError(f"{ref_name} missing sha256")
        seen: set[str] = set()
        for ts in manifest.tool_schemas:
            if ts.name in seen:
                raise SchemaValidationError(f"duplicate tool name {ts.name}")
            seen.add(ts.name)
            # Schema self-check: schemas must be objects/booleans, and empty input must validate
            # or fail loudly — never silently accepted.
            if not isinstance(ts.input_schema, (dict, bool)) or not isinstance(ts.output_schema, (dict, bool)):
                raise SchemaValidationError(f"{ts.name}: schemas must be objects or booleans")

    def register_task(self, task: TaskInput) -> ArtifactRef:
        task_json = task.model_dump(mode="json", by_alias=True)
        task_ref = self.store.put_artifact(task_json)
        self.store.register_task(
            task.task_id,
            task.environment_ref.id,
            task.environment_ref.version,
            task_ref.model_dump_json(by_alias=True),
            task.partition,
            task.goal,
        )
        return task_ref

    def get_task(self, task_id: str) -> TaskInput | None:
        row = self.store.get_task(task_id)
        if not row:
            return None
        ref = ArtifactRef.model_validate_json(row["task_ref"])
        data = self.store.get_artifact(ref)
        return TaskInput.model_validate(data)

    def list_tasks_by_partition(self, env_id: str, partition: str) -> list[TaskInput]:
        """Return tasks strictly filtered by environment AND partition."""
        rows = self.store.list_tasks_by_partition(env_id, partition)
        out = []
        for row in rows:
            ref = ArtifactRef.model_validate_json(row["task_ref"])
            out.append(TaskInput.model_validate(self.store.get_artifact(ref)))
        return out
