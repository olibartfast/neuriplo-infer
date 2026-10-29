#!/usr/bin/env python3
"""Resolve SDK versions from the pinned neuriplo metadata, without defaults."""
from pathlib import Path
import re
import sys


def resolve(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep and key in ('PYTORCH_VERSION', 'OPENVINO_VERSION', 'ONNX_RUNTIME_VERSION'):
            if key in values:
                raise ValueError(f'Duplicate SDK pin: {key}')
            value = value.strip().strip('\"\'')
            if not re.fullmatch(r'\d+\.\d+\.\d+', value):
                raise ValueError(f'Invalid SDK pin: {key}')
            values[key] = value
    return {key.lower(): values[key] for key in ('PYTORCH_VERSION', 'OPENVINO_VERSION', 'ONNX_RUNTIME_VERSION')}


if __name__ == '__main__':
    try:
        for key, value in resolve(sys.argv[1]).items():
            print(f'{key}={value}')
    except (IndexError, KeyError, ValueError, OSError) as error:
        sys.exit(f'CI SDK resolution failed: {error}')
