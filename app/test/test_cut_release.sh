#!/bin/bash
# Regression test for scripts/cut_release.sh against a fixture versions.env.
# Repeated cuts must keep exactly one sibling-pin comment block, leave the rest
# of the file alone, and be idempotent. The sibling tag lookup is stubbed so the
# test runs offline.
set -euo pipefail

ROOT_DIR="${1:?repo root required}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

REPO="${TMP_DIR}/repo"
BIN="${TMP_DIR}/bin"
mkdir -p "${REPO}/scripts" "${BIN}"
cp "${ROOT_DIR}/scripts/cut_release.sh" "${REPO}/scripts/"

REAL_GIT="$(command -v git)"
cat > "${BIN}/git" <<EOF
#!/bin/bash
# Offline stand-in for the sibling tag lookup; everything else is real git.
if [ "\$1" = "ls-remote" ]; then
  if [ -n "\${STUB_NO_TAGS_FOR:-}" ] && [[ "\$*" == *"/\${STUB_NO_TAGS_FOR}.git"* ]]; then
    exit 0
  fi
  printf 'abc\trefs/tags/v0.1.0\nabc\trefs/tags/v9.9.9\nabc\trefs/tags/v10.0.0-rc1\n'
  exit 0
fi
exec "${REAL_GIT}" "\$@"
EOF
chmod +x "${BIN}/git"

printf '0.9.1\n' > "${REPO}/VERSION"
# The stacked blocks are what earlier releases left behind.
cat > "${REPO}/versions.env" <<'EOF'
# System Dependencies (minimum versions)
OPENCV_MIN_VERSION=4.6.0
CMAKE_MIN_VERSION=3.20

# Sibling repository refs, each pinned to that sibling's current release tag
# so a checkout of the v0.9.0 tag rebuilds against the exact same sibling code.
# Siblings version independently -- these need not be equal.

# Sibling repository refs, each pinned to that sibling's current release tag
# so a checkout of the v0.9.1 tag rebuilds against the exact same sibling code.
# Siblings version independently -- these need not be equal.
NEURIPLO_VERSION=v0.9.1
VIDEOCAPTURE_VERSION=v0.5.0
NEURIPLO_TASKS_VERSION=v0.8.2
NEURIPLO_KSERVE_CLIENT_VERSION=v0.4.0
EOF
"${REAL_GIT}" -C "${REPO}" init -q

cut() {
  PATH="${BIN}:${PATH}" bash "${REPO}/scripts/cut_release.sh" 1.2.3 > /dev/null
}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

ENV_FILE="${REPO}/versions.env"
cut
cp "${ENV_FILE}" "${TMP_DIR}/first-cut.env"
cut

[ "$(grep -c '^# Sibling repository refs' "${ENV_FILE}")" -eq 1 ] ||
  fail "expected exactly one sibling-pin comment block"
grep -q 'checkout of the v1.2.3 tag' "${ENV_FILE}" ||
  fail "pin comment does not name the new tag"
for line in '# System Dependencies (minimum versions)' \
  'OPENCV_MIN_VERSION=4.6.0' 'CMAKE_MIN_VERSION=3.20'; do
  grep -qxF "${line}" "${ENV_FILE}" || fail "unrelated line lost: ${line}"
done
for var in NEURIPLO VIDEOCAPTURE NEURIPLO_TASKS NEURIPLO_KSERVE_CLIENT; do
  [ "$(grep -c "^${var}_VERSION=" "${ENV_FILE}")" -eq 1 ] ||
    fail "expected one ${var}_VERSION line"
  grep -qx "${var}_VERSION=v9.9.9" "${ENV_FILE}" ||
    fail "${var}_VERSION not pinned to the latest release tag"
done
cmp -s "${TMP_DIR}/first-cut.env" "${ENV_FILE}" ||
  fail "a second cut of the same version changed versions.env"
[ "$(cat "${REPO}/VERSION")" = "1.2.3" ] || fail "VERSION not bumped"

# A sibling with no release tag must stop the cut with a message naming it,
# not end silently inside a failed pipeline under `set -e`.
if MISSING_OUTPUT="$(STUB_NO_TAGS_FOR=videocapture PATH="${BIN}:${PATH}" \
  bash "${REPO}/scripts/cut_release.sh" 1.2.4 2>&1)"; then
  fail "a sibling without a release tag must fail the cut"
fi
grep -q "missing videocapture" <<<"${MISSING_OUTPUT}" ||
  fail "no message naming the sibling without a release tag"

echo "cut_release_versions_env: ok"
