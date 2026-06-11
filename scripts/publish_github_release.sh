#!/usr/bin/env bash
# Create a GitHub Release for an existing annotated/lightweight tag.
# Pushing a git tag alone does not populate github.com/.../releases — this does.
#
# Usage: scripts/publish_github_release.sh <version>
# Example: scripts/publish_github_release.sh 0.5.0
#
# Requires: gh CLI authenticated (gh auth login), tag already on origin.

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <version>" >&2
  echo "Example: $0 0.5.0" >&2
  exit 2
fi

VERSION_NUM="${1#v}"
TAG="v${VERSION_NUM}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -f CHANGELOG.md ]; then
  echo "Error: CHANGELOG.md not found" >&2
  exit 1
fi

if ! git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "Error: local tag ${TAG} not found" >&2
  exit 1
fi

if ! git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  echo "Error: tag ${TAG} is not on origin; push it first (git push origin ${TAG})" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh CLI not found; install GitHub CLI" >&2
  exit 1
fi

if ! gh api user -q .login >/dev/null 2>&1; then
  echo "Error: gh is not authenticated; run: gh auth login" >&2
  exit 1
fi

if gh release view "${TAG}" >/dev/null 2>&1; then
  echo "GitHub Release ${TAG} already exists: https://github.com/olibartfast/neuriplo-infer/releases/tag/${TAG}"
  exit 0
fi

NOTES_FILE="$(mktemp)"
trap 'rm -f "${NOTES_FILE}"' EXIT

bash scripts/extract_changelog_release_notes.sh "${VERSION_NUM}" > "${NOTES_FILE}"

if [ ! -s "${NOTES_FILE}" ]; then
  echo "Error: no CHANGELOG section for [${VERSION_NUM}]" >&2
  exit 1
fi

echo "==> Creating GitHub Release ${TAG} from CHANGELOG..."
gh release create "${TAG}" \
  --repo olibartfast/neuriplo-infer \
  --title "${TAG}" \
  --notes-file "${NOTES_FILE}"

echo "==> Done: https://github.com/olibartfast/neuriplo-infer/releases/tag/${TAG}"
