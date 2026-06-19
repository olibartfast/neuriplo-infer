# Feature Branch Merge Roadmap

Atomic step-by-step plan to merge `feature/*` or `feat/*` → `develop` (GitFlow).

---

## 0. Pre-flight

- [ ] Working tree clean (`git status` — no uncommitted changes).
- [ ] On the feature branch (`git rev-parse --abbrev-ref HEAD`).
- [ ] All local commits pushed to `origin/feature/<name>`.
- [ ] CI green on the feature branch PR/diff (if pushed).
- [ ] Neuriplo-infer contract checks pass (task routing, supported model types, etc. per `AGENTS.md`).

## 1. Sync feature branch with latest develop

```bash
git fetch origin
git merge origin/develop --no-edit
```

- [ ] Resolve conflicts (if any); commit the merge.
- [ ] **Re-run local checks** after the merge (build, tests, lint) — the merge commit can break things.

```bash
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release
cmake --build build

cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-test
ctest --test-dir build-test --output-on-failure
```

- [ ] Build + tests pass with zero failures.

## 2. Open / update PR: `feature/<name>` → `develop`

- [ ] PR targets `develop` (never `master`).
- [ ] PR title and body match the repo's conventional commit / changelog style.
- [ ] If `AGENTS.md` checklists apply (new task type, model type, docs sync), all items are ticked.

## 3. Merge the PR

- [ ] Merge via **merge commit** (not squash, not rebase) to preserve the feature branch identity in `develop` history.
- [ ] Confirm CI passes on the merge result (protect-branch checks).

## 4. Post-merge cleanup

```bash
git checkout develop
git pull origin develop
```

- [ ] `develop` now contains the merge commit.
- [ ] Delete local feature branch:

```bash
git branch -d feature/<name>
```

- [ ] Delete remote feature branch:

```bash
git push origin --delete feature/<name>
```

- [ ] Confirm gone: `git branch -a | grep feature/<name>` returns nothing.

## 5. Convergence check

```bash
git rev-list --left-right --count origin/develop...origin/master
# Must be >0 commits ahead on develop, 0 behind master
```

- [ ] `develop` is ahead of `master` (features accumulate) and never behind.
- [ ] `master` → `develop` back-merge is NOT needed (feature merges don't touch `master`).

---

## Quick reference (feature merge in 6 commands)

```bash
# 1. Sync (use feature/<name> or feat/<name>)
git checkout feature/<name> && git fetch origin && git merge origin/develop

# 2. Verify
cmake --build build && ctest --test-dir build-test --output-on-failure

# 3. Push & PR
git push origin feature/<name>
# → open/merge PR feature/<name> → develop via GitHub

# 4. Cleanup
git checkout develop && git pull
git branch -d feature/<name>
git push origin --delete feature/<name>
```
