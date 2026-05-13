#!/usr/bin/env bash
# Prepare a vision-inference release with correctly pinned sibling refs.
#
# Usage: scripts/cut_release.sh <version>
# Example: scripts/cut_release.sh 0.3.0
#
# What this does:
#   1. Validates the version is a clean semver.
#   2. Confirms vision-core, neuriplo, and videocapture have matching tags.
#   3. Writes VERSION = <version>.
#   4. Replaces NEURIPLO_VERSION / VIDEOCAPTURE_VERSION / VISION_CORE_VERSION
#      in versions.env with v<version>.
#   5. Stages VERSION and versions.env. Does NOT commit, tag, or push.
#
# After this script, you still need to: update CHANGELOG.md, commit, open the
# release PR to master, and tag after merge.

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
  echo "Error: must run from a vision-inference checkout (VERSION + versions.env missing)" >&2
  exit 1
fi

echo "==> Verifying sibling tags exist on remote..."
fail=0
for repo in vision-core neuriplo videocapture; do
  if git ls-remote --tags "https://github.com/olibartfast/${repo}.git" "refs/tags/${TAG}" 2>/dev/null \
       | grep -q "refs/tags/${TAG}$"; then
    echo "  ok ${repo}@${TAG}"
  else
    echo "  missing ${repo}@${TAG} -- tag the matching sibling commit first" >&2
    fail=1
  fi
done
[ "${fail}" -eq 0 ] || exit 1

echo "==> Updating VERSION..."
printf '%s\n' "${VERSION_NUM}" > VERSION

echo "==> Updating versions.env pins..."
# Remove any existing pin lines (commented or active) for the three sibling vars.
awk -v IGNORECASE=0 '
  /^[[:space:]]*#?[[:space:]]*(NEURIPLO|VIDEOCAPTURE|VISION_CORE)_VERSION=/ { next }
  { print }
' versions.env > versions.env.tmp
# Strip trailing blank lines.
awk 'NF { blank=0; for (i=0;i<n;i++) print buf[i]; n=0; print; next } { buf[n++]=$0 }' \
  versions.env.tmp > versions.env
rm -f versions.env.tmp

cat >> versions.env <<EOF

# Sibling repository refs pinned for ${TAG} release.
NEURIPLO_VERSION=${TAG}
VIDEOCAPTURE_VERSION=${TAG}
VISION_CORE_VERSION=${TAG}
EOF

echo "==> Staging changes..."
git add VERSION versions.env

echo ""
echo "==> Done. Next steps:"
echo "  1. Update CHANGELOG.md for ${TAG}."
echo "  2. git commit -m 'release: ${TAG}'"
echo "  3. Push the commit and open a PR to master."
echo "  4. After merge: git tag ${TAG} && git push origin ${TAG}"
echo ""
echo "Current versions.env:"
cat versions.env
