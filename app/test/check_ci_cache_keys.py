#!/usr/bin/env python3
"""Check that every actions/cache restore-keys entry can match its own key.

restore-keys match by literal prefix against saved cache keys. A restore key
built from a different expression than its primary key (for example a hash of
fewer files) is never a prefix of it, so the cache silently misses whenever the
primary key changes. Usage: check_ci_cache_keys.py WORKFLOW.yml [...]
"""

import os
import re
import sys

KEY = re.compile(r"^\s*key:\s*(\S.*?)\s*$")
RESTORE = re.compile(r"^\s*restore-keys:\s*(\S.*?)\s*$")


def check(path):
    # A missing or empty path is a controlled failure for that input, so the
    # remaining workflows are still checked and reported.
    if not path:
        return 0, ["empty workflow path"]
    if not os.path.isfile(path):
        return 0, [f"{path}: workflow file does not exist"]
    failures = []
    checked = 0
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for index, line in enumerate(lines):
        restore = RESTORE.match(line)
        if not restore:
            continue
        checked += 1
        key = None
        indent = len(line) - len(line.lstrip())
        # The key belongs to the same `with:` block: scan upward through lines at
        # this indentation and stop at the block's parent (a shallower line), so
        # a key from a neighbouring step is never borrowed.
        for previous in range(index - 1, -1, -1):
            text = lines[previous]
            if not text.strip():
                continue
            previous_indent = len(text) - len(text.lstrip())
            if previous_indent < indent:
                break
            match = KEY.match(text)
            if match and previous_indent == indent:
                key = match.group(1)
                break
        prefix = restore.group(1)
        if key is None:
            failures.append(f"{path}:{index + 1}: restore-keys has no key above it")
        elif not key.startswith(prefix):
            failures.append(
                f"{path}:{index + 1}: restore-keys '{prefix}' is not a prefix of key '{key}'"
            )
    return checked, failures


def main(paths):
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    total = 0
    failures = []
    for path in paths:
        checked, found = check(path)
        total += checked
        failures.extend(found)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print(f"ci_cache_restore_keys: ok ({total} restore-keys checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
