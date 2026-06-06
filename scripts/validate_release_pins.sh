#!/usr/bin/env bash
# Validate that versions.env at the current checkout pins every sibling
# (neuriplo-tasks, neuriplo, videocapture) to a real release tag, and that each
# pinned tag actually exists on the sibling's remote. Used by:
#   - .githooks/pre-push (blocks pushes of unpinned neuriplo-infer release tags)
#   - .github/workflows/release-guard.yml (server-side enforcement on tag push)
#   - scripts/cut_release.sh (sanity check before staging release changes)
#
# Siblings version independently: neuriplo-tasks and neuriplo track the
# neuriplo-infer release cadence, but videocapture may lag (e.g. stay at
# v0.2.0 while the others move to v0.3.0). So this does NOT require the three
# pins to be equal -- it only requires each to be a concrete semver tag
# (vX.Y.Z), never a branch name like 'master' or 'develop', and to exist on
# the sibling remote. That is what makes a tag checkout reproducible.
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

semver_tag_regex='^v[0-9]+\.[0-9]+\.[0-9]+$'

fail=0
echo "==> Validating sibling pins in versions.env for ${TAG}..."
# Each entry maps a versions.env key to the sibling repository it pins.
for entry in "NEURIPLO_VERSION=neuriplo" \
             "VIDEOCAPTURE_VERSION=videocapture" \
             "NEURIPLO_TASKS_VERSION=neuriplo-tasks"; do
  key="${entry%%=*}"
  repo="${entry#*=}"
  val="$(extract_pin "${key}")"

  if [ -z "${val}" ]; then
    echo "::error::versions.env is missing ${key}. Pin it to a ${repo} release tag (vX.Y.Z) before tagging." >&2
    fail=1
    continue
  fi
  if ! [[ "${val}" =~ ${semver_tag_regex} ]]; then
    echo "::error::${key}=${val} is not a release tag. It must be a concrete vX.Y.Z tag, never a branch like 'master' or 'develop'." >&2
    fail=1
    continue
  fi
  if git ls-remote --tags "https://github.com/olibartfast/${repo}.git" "refs/tags/${val}" 2>/dev/null \
       | grep -q "refs/tags/${val}$"; then
    echo "  ok ${key}=${val} (${repo} tag exists)"
  elif [ "${repo}" = "neuriplo-tasks" ] && git ls-remote --tags "https://github.com/olibartfast/vision-core.git" "refs/tags/${val}" 2>/dev/null \
       | grep -q "refs/tags/${val}$"; then
    echo "  ok ${key}=${val} (vision-core tag exists; neuriplo-tasks redirect pending)"
  else
    echo "::error::${key}=${val} but ${repo} has no tag ${val}. Tag it before pushing neuriplo-infer ${TAG}." >&2
    fail=1
  fi
done

if [ "${fail}" -ne 0 ]; then
  echo "" >&2
  echo "Run: scripts/cut_release.sh ${TAG#v}" >&2
  echo "to re-pin versions.env to each sibling's current release tag." >&2
  exit 1
fi

echo "==> versions.env pins look correct (each sibling pinned to an existing release tag)"
exit 0
