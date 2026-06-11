#!/usr/bin/env bash
# Print the CHANGELOG.md section for a release version to stdout.
#
# Usage: scripts/extract_changelog_release_notes.sh <version>
# Example: scripts/extract_changelog_release_notes.sh 0.5.0

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <version>" >&2
  exit 2
fi

VERSION_NUM="${1#v}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "${REPO_ROOT}/CHANGELOG.md" ]; then
  echo "Error: CHANGELOG.md not found" >&2
  exit 1
fi

awk -v ver="${VERSION_NUM}" '
  $0 ~ "^## \\[" ver "\\]" { p=1; next }
  /^## \[/ { if (p) exit }
  p { print }
' "${REPO_ROOT}/CHANGELOG.md"
