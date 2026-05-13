#!/usr/bin/env bash
# Validate that versions.env at the current checkout pins all three siblings
# (vision-core, neuriplo, videocapture) to the given tag, and that each sibling
# repo has a matching remote tag. Used by:
#   - .githooks/pre-push (blocks pushes of unpinned vision-inference release tags)
#   - .github/workflows/release-guard.yml (server-side enforcement on tag push)
#   - scripts/cut_release.sh (sanity check before staging release changes)
#
# Usage: scripts/validate_release_pins.sh <tag>
# Example: scripts/validate_release_pins.sh v0.3.0
#
# Exit codes: 0 = pins look good, 1 = misconfigured

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <tag>" >&2
  exit 2
fi

TAG="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSIONS_ENV="${REPO_ROOT}/versions.env"

if [ ! -f "${VERSIONS_ENV}" ]; then
  echo "::error::versions.env not found at ${VERSIONS_ENV}" >&2
  exit 1
fi

extract_pin() {
  local key="$1"
  awk -F= -v k="^${key}=" '$0 ~ k {gsub(/[ \t\r"]/,"",$2); print $2; exit}' "${VERSIONS_ENV}"
}

NEURIPLO=$(extract_pin NEURIPLO_VERSION)
VIDEOCAPTURE=$(extract_pin VIDEOCAPTURE_VERSION)
VISION_CORE=$(extract_pin VISION_CORE_VERSION)

fail=0
for entry in "NEURIPLO_VERSION=${NEURIPLO}" \
             "VIDEOCAPTURE_VERSION=${VIDEOCAPTURE}" \
             "VISION_CORE_VERSION=${VISION_CORE}"; do
  key="${entry%%=*}"
  val="${entry#*=}"
  if [ -z "${val}" ]; then
    echo "::error::versions.env is missing ${key}. Set it to ${TAG} before tagging." >&2
    fail=1
  elif [ "${val}" != "${TAG}" ]; then
    echo "::error::versions.env has ${key}=${val} but the release tag is ${TAG}." >&2
    fail=1
  fi
done

if [ "${fail}" -ne 0 ]; then
  echo "" >&2
  echo "Run: scripts/cut_release.sh ${TAG#v}" >&2
  echo "to update VERSION and versions.env consistently." >&2
  exit 1
fi

echo "==> versions.env pins look correct (all three sibling refs = ${TAG})"

echo "==> Verifying sibling repos have matching ${TAG} tag..."
for repo in vision-core neuriplo videocapture; do
  if git ls-remote --tags "https://github.com/olibartfast/${repo}.git" "refs/tags/${TAG}" 2>/dev/null \
       | grep -q "refs/tags/${TAG}$"; then
    echo "  ok ${repo}@${TAG}"
  else
    echo "::error::Sibling ${repo} has no tag ${TAG}. Tag it before pushing vision-inference ${TAG}." >&2
    fail=1
  fi
done

exit "${fail}"
