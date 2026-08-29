#!/usr/bin/env python3
"""Validates `neuriplo-infer --capabilities` against the published schema.

The C++ contract tests assert what the document contains; this asserts that it
still matches `docs/capabilities.schema.json`, which is what consumers read.
Without it the schema and the emitter drift silently, because the schema
forbids unknown properties and nothing was checking.
"""

import json
import subprocess
import sys

try:
    import jsonschema
except ImportError:  # pragma: no cover - environment without the validator
    print("jsonschema is not installed; skipping schema validation")
    sys.exit(0)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <neuriplo-infer> <capabilities.schema.json>")
        return 2

    binary, schema_path = sys.argv[1], sys.argv[2]

    completed = subprocess.run(
        [binary, "--capabilities"], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        print(f"--capabilities exited with {completed.returncode}")
        print(completed.stderr)
        return 1

    with open(schema_path, encoding="utf-8") as handle:
        schema = json.load(handle)

    document = json.loads(completed.stdout)
    jsonschema.validate(document, schema)
    print("capabilities document matches the published schema")
    return 0


if __name__ == "__main__":
    sys.exit(main())
