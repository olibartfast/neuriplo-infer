#!/usr/bin/env bash
# Prepare a neuriplo-infer release with correctly pinned sibling refs.
#
# Usage: scripts/cut_release.sh <version>
# Example: scripts/cut_release.sh 0.3.0
#
# What this does:
#   1. Validates the version is a clean semver.
#   2. Detects each sibling's current release tag (neuriplo-tasks, neuriplo,
#      videocapture) from its remote -- the latest vX.Y.Z tag it has.
#   3. Writes VERSION = <version>.
#   4. Replaces NEURIPLO_VERSION / VIDEOCAPTURE_VERSION / NEURIPLO_TASKS_VERSION
#      in versions.env with each sibling's detected release tag.
#   5. Stages VERSION and versions.env. Does NOT commit, tag, or push.
#
# Siblings version independently: neuriplo-tasks and neuriplo usually move with
# neuriplo-infer, but videocapture may lag (e.g. stay at v0.2.0). Each pin is
# whatever release tag that sibling currently has -- they need NOT be equal.
# The only hard rule, enforced by validate_release_pins.sh, is that every pin
# is a concrete vX.Y.Z tag, never a branch like 'master' or 'develop'.
#
# After this script, you still need to: update CHANGELOG.md, commit, merge the
# release branch to master, and tag.

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <version>" >&2
  echo "Example: $0 0.3.0" >&2
  exit 2
fi

VERSION_NUM="${1#v}"
TAG="v${VERSION_NUM}"

if ! [[ "${VERSION_NUM}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must match MAJOR.MINOR.PATCH (got '${VERSION_NUM}')" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -f versions.env ] || [ ! -f VERSION ]; then
  echo "Error: must run from a neuriplo-infer checkout (VERSION + versions.env missing)" >&2
  exit 1
fi

# Print the latest vX.Y.Z release tag a sibling repo has on its remote.
latest_remote_tag() {
  local repo="$1"
  git ls-remote --tags --refs "https://github.com/olibartfast/${repo}.git" 2>/dev/null \
    | sed 's#.*refs/tags/##' \
    | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
    | sort -V \
    | tail -n1
}

echo "==> Detecting sibling release tags from their remotes..."
fail=0
NEURIPLO_PIN="$(latest_remote_tag neuriplo)"
VIDEOCAPTURE_PIN="$(latest_remote_tag videocapture)"
NEURIPLO_TASKS_PIN="$(latest_remote_tag neuriplo-tasks)"
NEURIPLO_KSERVE_CLIENT_PIN="$(latest_remote_tag neuriplo-kserve-client)"

for entry in "neuriplo=${NEURIPLO_PIN}" \
             "videocapture=${VIDEOCAPTURE_PIN}" \
             "neuriplo-tasks=${NEURIPLO_TASKS_PIN}" \
             "neuriplo-kserve-client=${NEURIPLO_KSERVE_CLIENT_PIN}"; do
  repo="${entry%%=*}"
  pin="${entry#*=}"
  if [ -z "${pin}" ]; then
    echo "  missing ${repo} -- it has no vX.Y.Z release tag; tag it first" >&2
    fail=1
  else
    echo "  ${repo} -> ${pin}"
  fi
done
[ "${fail}" -eq 0 ] || exit 1

echo "==> Updating VERSION..."
printf '%s\n' "${VERSION_NUM}" > VERSION

echo "==> Updating versions.env pins..."
# Remove any existing pin lines (commented or active) for the three sibling vars.
awk -v IGNORECASE=0 '
  /^[[:space:]]*#?[[:space:]]*(NEURIPLO|VIDEOCAPTURE|NEURIPLO_TASKS|NEURIPLO_KSERVE_CLIENT)_VERSION=/ { next }
  { print }
' versions.env > versions.env.tmp
# Strip trailing blank lines.
awk 'NF { blank=0; for (i=0;i<n;i++) print buf[i]; n=0; print; next } { buf[n++]=$0 }' \
  versions.env.tmp > versions.env
rm -f versions.env.tmp

cat >> versions.env <<EOF

# Sibling repository refs, each pinned to that sibling's current release tag
# so a checkout of the ${TAG} tag rebuilds against the exact same sibling code.
# Siblings version independently -- these need not be equal.
NEURIPLO_VERSION=${NEURIPLO_PIN}
VIDEOCAPTURE_VERSION=${VIDEOCAPTURE_PIN}
NEURIPLO_TASKS_VERSION=${NEURIPLO_TASKS_PIN}
NEURIPLO_KSERVE_CLIENT_VERSION=${NEURIPLO_KSERVE_CLIENT_PIN}
EOF

echo "==> Staging changes..."
git add VERSION versions.env

echo ""
echo "==> Done. Next steps:"
echo "  1. Update CHANGELOG.md for ${TAG}."
echo "  2. git commit -m 'release: ${TAG}'"
echo "  3. Merge the release branch to master and tag ${TAG}."
echo "  4. git push origin master ${TAG}"
echo ""
echo "Current versions.env:"
cat versions.env
